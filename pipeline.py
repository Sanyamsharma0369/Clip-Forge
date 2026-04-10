from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, parse, request

OLLAMA_MODEL = "mistral"
WHISPER_MODEL = "base"
MAX_CLIPS = 5
MIN_CLIP_SEC = 30
MAX_CLIP_SEC = 90
OUTPUT_DIR = Path("outputs/clips")
TEMP_DIR = Path("temp")
LOG_FILE = Path("pipeline.log")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENCODER_PREFER_QUALITY = True

PROMPT_TEMPLATE = """You are a viral video editor. Analyze this transcript chunk and find the BEST moments
for YouTube Shorts or Instagram Reels ({min_sec}-{max_sec} seconds each).

Look for:
- Emotional peaks (anger, joy, surprise, inspiration)
- Surprising facts or shocking statements
- Powerful quotes or one-liners
- Story climaxes or dramatic turns
- Actionable tips that stand alone

TRANSCRIPT:
{chunk}

Return ONLY a valid JSON array (no text before or after). Max {max_clips} items.
Format:
[
  {{
    "start": <start_seconds as number>,
    "end": <end_seconds as number>,
    "title": "<catchy short title under 8 words>",
    "hook": "<first sentence to grab attention>",
    "reason": "<why this moment is viral-worthy>"
  }}
]

Rules:
- Each clip must be {min_sec}-{max_sec} seconds long
- Extend end time if needed to complete a thought
- start and end must be numbers (no timestamps, no strings)
- Return empty array [] if no good moments found
"""

LOGGER = logging.getLogger("clipforge")

PROGRESS_PREFIX = "PROGRESS"


class PipelineError(RuntimeError):
    """Represent a recoverable pipeline failure."""


class GPUOrchestrator:
    """Manage GPU VRAM allocation across pipeline stages."""

    REALESRGAN_GPU_ID = "1"
    REALESRGAN_EXE = "realesrgan-ncnn-vulkan"
    REALESRGAN_MODEL = "realesrgan-x4plus"
    _ollama_stopped_for_whisper = False

    @staticmethod
    def free_vram_gb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()[0] / 1024**3
        except Exception:
            pass
        return 0.0

    @staticmethod
    def cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def stop_ollama() -> None:
        """Kill Ollama to free VRAM before Whisper runs."""
        subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], capture_output=True)
        GPUOrchestrator._ollama_stopped_for_whisper = True
        time.sleep(2)
        LOGGER.info("  Ollama paused - VRAM freed for Whisper")

    @staticmethod
    def start_ollama() -> None:
        """Restart Ollama after Whisper completes."""
        if not GPUOrchestrator._ollama_stopped_for_whisper:
            return
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(15)
        GPUOrchestrator._ollama_stopped_for_whisper = False
        LOGGER.info("  Ollama restarted on GPU")

    @staticmethod
    def best_whisper_model() -> str:
        """
        Stop Ollama, measure free VRAM, and return the best Whisper model.
        large  -> >=4.5GB
        medium -> >=2.0GB
        small  -> >=1.0GB
        base   -> fallback
        """
        if not GPUOrchestrator.cuda_available():
            LOGGER.info("  No CUDA - using Whisper base on CPU")
            return "base"

        GPUOrchestrator.stop_ollama()
        time.sleep(1)
        free = GPUOrchestrator.free_vram_gb()
        LOGGER.info("  Free VRAM after stopping Ollama: %.2f GB", free)

        if free >= 4.5:
            model = "large"
        elif free >= 2.0:
            model = "medium"
        elif free >= 1.0:
            model = "small"
        else:
            model = "base"

        LOGGER.info("  Auto-selected Whisper model: %s", model)
        return model

    @staticmethod
    def realesrgan_available() -> bool:
        exe = GPUOrchestrator.REALESRGAN_EXE
        return shutil.which(exe) is not None or Path(f"{exe}.exe").exists()


def setup_logging() -> logging.Logger:
    """Configure console and file logging once."""
    if LOGGER.handlers:
        return LOGGER
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
    except OSError as exc:
        LOGGER.warning("Could not open log file %s: %s", LOG_FILE, exc)
    return LOGGER


def log_progress(stage: str, detail: str, *, message: str | None = None) -> None:
    """Emit a human-readable progress log plus a structured marker for the UI."""
    if message:
        LOGGER.info("%s | %s:%s:%s", message, PROGRESS_PREFIX, stage, detail)
    else:
        LOGGER.info("%s:%s:%s", PROGRESS_PREFIX, stage, detail)


def is_url(value: str) -> bool:
    """Return True when the input looks like a URL."""
    parsed = parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def ts_to_sec(value: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds."""
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def sec_to_ts(value: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    total_ms = int(round(value * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def sec_to_srt_ts(value: float) -> str:
    """Convert seconds to SRT timestamp format."""
    return sec_to_ts(value).replace(".", ",")


def safe_filename(value: str, max_len: int = 60) -> str:
    """Sanitize a string for safe filesystem use."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or "clip")[:max_len]


def estimate_tokens(text: str) -> int:
    """Approximate token count from text length."""
    return max(1, math.ceil(len(text) / 4))


def detect_device() -> str:
    """Return the best available torch device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_ffmpeg_binary() -> str:
    """Locate the FFmpeg executable on PATH or in common Windows install locations."""
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    if sys.platform.startswith("win"):
        package_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        for pattern in ("Gyan.FFmpeg.Essentials*", "Gyan.FFmpeg*", "BtbN.FFmpeg*"):
            for package_dir in package_root.glob(pattern):
                matches = sorted(package_dir.glob("**/ffmpeg.exe"))
                if matches:
                    return str(matches[0])

    raise PipelineError("Missing command: ffmpeg")


def ensure_ffmpeg_on_path() -> str:
    """Ensure FFmpeg's directory is available in this process PATH."""
    ffmpeg_binary = find_ffmpeg_binary()
    ffmpeg_dir = str(Path(ffmpeg_binary).resolve().parent)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in parts:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path if current_path else ffmpeg_dir
    return ffmpeg_binary


def _cpu_encoder() -> tuple[str, list[str]]:
    """Return the standard CPU encoder flags."""
    return "libx264", [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-b:v",
        "14000k",
        "-maxrate",
        "16000k",
        "-bufsize",
        "30000k",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-pix_fmt",
        "yuv420p",
    ]


def _test_encoder(name: str) -> bool:
    """Return True when FFmpeg can use the named encoder."""
    ffmpeg_binary = ensure_ffmpeg_on_path()
    try:
        result = subprocess.run(
            [
                ffmpeg_binary,
                "-f",
                "lavfi",
                "-i",
                "nullsrc=s=64x64",
                "-t",
                "0.1",
                "-c:v",
                name,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def detect_encoder(prefer_quality: bool = True) -> tuple[str, list[str]]:
    """
    Auto-detect best video encoder.
    prefer_quality=True -> always libx264 slow (default, best output)
    prefer_quality=False -> NVENC > AMF > QSV > libx264 (fastest)
    """
    if prefer_quality:
        return _cpu_encoder()

    if _test_encoder("h264_nvenc"):
        LOGGER.info("  Using NVENC (RTX 4050)")
        return "h264_nvenc", [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "14000k",
            "-maxrate",
            "16000k",
            "-bufsize",
            "30000k",
            "-profile:v",
            "high",
            "-level",
            "4.2",
            "-pix_fmt",
            "yuv420p",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
        ]

    if _test_encoder("h264_amf"):
        LOGGER.info("  Using AMF (AMD GPU)")
        return "h264_amf", [
            "-c:v",
            "h264_amf",
            "-quality",
            "quality",
            "-rc",
            "vbr_peak",
            "-b:v",
            "14000k",
            "-maxrate",
            "16000k",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ]

    if _test_encoder("h264_qsv"):
        LOGGER.info("  Using QuickSync (Intel GPU)")
        return "h264_qsv", [
            "-c:v",
            "h264_qsv",
            "-global_quality",
            "18",
            "-look_ahead",
            "1",
            "-b:v",
            "14000k",
            "-maxrate",
            "16000k",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ]

    LOGGER.info("  No GPU encoder - using libx264")
    return _cpu_encoder()


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with readable error handling."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PipelineError(f"Missing command: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "Unknown subprocess error."
        raise PipelineError(stderr) from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"Command timed out after {timeout} seconds: {' '.join(command)}") from exc


def get_video(source: str, job_id: str = "") -> Path:
    """Return a local video path, downloading when needed."""
    path = Path(source).expanduser()
    if not is_url(source):
        if not path.exists():
            raise FileNotFoundError(f"Local video file was not found: {path}")
        return path.resolve()

    safe_id = safe_filename(job_id or uuid.uuid4().hex[:8], max_len=32)
    download_dir = TEMP_DIR / f"download_{safe_id}"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / "video.mp4"
    temp_output_path = download_dir / "video.temp.mp4"
    LOGGER.info("Downloading source video with yt-dlp...")

    def _ydl_progress_hook(data: dict[str, Any]) -> None:
        status = str(data.get("status", ""))
        if status == "downloading":
            pct = str(data.get("_percent_str", "?")).strip()
            speed = str(data.get("_speed_str", "?")).strip()
            eta = str(data.get("_eta_str", "?")).strip()
            log_progress("download", f"{pct} @ {speed} ETA {eta}", message=f"Download progress: {pct} speed: {speed} ETA: {eta}")
        elif status == "finished":
            filename = str(data.get("filename", "")).strip()
            log_progress("download", "100% complete", message=f"Download finished: {filename or output_path.name}")

    try:
        import yt_dlp

        ydl_opts: dict[str, Any] = {
            "remote_components": ["ejs:github"],
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "noplaylist": True,
            "nopart": True,
            "merge_output_format": "mp4",
            "overwrites": True,
            "outtmpl": str(output_path),
            "progress_hooks": [_ydl_progress_hook],
            "noprogress": False,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source])
        if temp_output_path.exists() and not output_path.exists():
            for _ in range(5):
                try:
                    temp_output_path.replace(output_path)
                    break
                except OSError:
                    time.sleep(1)
    except PipelineError as exc:
        LOGGER.error("yt-dlp failed: %s", exc)
        raise PipelineError(f"yt-dlp failed: {exc}\nSuggestion: pip install -U yt-dlp") from exc
    except FileNotFoundError as exc:
        raise PipelineError("Missing command: yt-dlp") from exc
    except Exception as exc:
        stderr = str(exc).strip() or "Unknown yt-dlp error."
        if temp_output_path.exists():
            for _ in range(5):
                try:
                    temp_output_path.replace(output_path)
                    LOGGER.warning("yt-dlp rename failed, but recovered completed download: %s", output_path)
                    return output_path.resolve()
                except OSError:
                    time.sleep(1)
            LOGGER.warning("yt-dlp rename failed; using completed temp video directly: %s", temp_output_path)
            return temp_output_path.resolve()
        if output_path.exists():
            LOGGER.warning("yt-dlp reported failure, but output video exists: %s", output_path)
            return output_path.resolve()
        LOGGER.error("yt-dlp failed: %s", stderr)
        raise PipelineError(f"yt-dlp failed: {stderr}\nSuggestion: pip install -U yt-dlp") from exc
    if not output_path.exists():
        raise PipelineError("yt-dlp completed without producing a video file.")
    return output_path.resolve()


def _collect_transcript_segments(raw_segments: Any, info: Any) -> tuple[list[dict[str, Any]], str]:
    """Consume faster-whisper's generator while emitting periodic progress."""
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    total_duration = float(getattr(info, "duration", 0) or 0)
    last_logged_pct = -1

    for segment in raw_segments:
        text = str(segment.text).strip()
        if not text:
            continue

        end_time = float(segment.end)
        segments.append(
            {
                "start": float(segment.start),
                "end": end_time,
                "text": text,
            }
        )
        text_parts.append(text)

        if total_duration > 0:
            pct = min(int((end_time / total_duration) * 100), 99)
            if pct >= last_logged_pct + 5:
                last_logged_pct = pct
                elapsed_str = f"{int(end_time // 60)}m {int(end_time % 60)}s"
                total_str = f"{int(total_duration // 60)}m {int(total_duration % 60)}s"
                log_progress(
                    "transcribe",
                    str(pct),
                    message=f"Transcribing... {pct}% [{elapsed_str} / {total_str}]",
                )

    LOGGER.info("Transcription complete - %s segments", len(segments))
    return segments, " ".join(text_parts).strip()


def _transcribe_in_process(video_path: Path, model_name: str) -> dict[str, Any]:
    """Run faster-whisper transcription inside the current process."""
    ensure_ffmpeg_on_path()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("faster-whisper is not installed. Run: pip install faster-whisper") from exc

    device = "cuda" if GPUOrchestrator.cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    LOGGER.info("Loading faster-whisper %s on %s...", model_name, device.upper())
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    LOGGER.info("Transcribing %s...", video_path.name)
    raw_segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments, text = _collect_transcript_segments(raw_segments, info)
    return {
        "text": text,
        "segments": segments,
        "language": getattr(info, "language", "en") or "en",
    }


def _transcribe_worker_to_file(video_path: Path, model_name: str, output_path: Path) -> None:
    """Transcribe and persist output before hard-exiting the worker process."""
    ensure_ffmpeg_on_path()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("faster-whisper is not installed. Run: pip install faster-whisper") from exc

    device = "cuda" if GPUOrchestrator.cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    LOGGER.info("Loading faster-whisper %s on %s...", model_name, device.upper())
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    LOGGER.info("Transcribing %s...", video_path.name)
    raw_segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments, text = _collect_transcript_segments(raw_segments, info)
    transcript = {
        "text": text,
        "segments": segments,
        "language": getattr(info, "language", "en") or "en",
    }
    save_json(output_path, transcript)
    LOGGER.info("Transcription worker saved transcript: %s", output_path)
    for handler in LOGGER.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os._exit(0)


def transcribe(video_path: Path, model_name: str | None = None) -> dict[str, Any]:
    """Transcribe audio using a worker process for CUDA isolation."""
    if model_name is None:
        model_name = GPUOrchestrator.best_whisper_model()
    elif GPUOrchestrator.cuda_available():
        GPUOrchestrator.stop_ollama()

    if not GPUOrchestrator.cuda_available():
        return _transcribe_in_process(video_path, model_name)

    worker_output = TEMP_DIR / f"transcript_worker_{uuid.uuid4().hex[:8]}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(video_path),
        "--whisper",
        model_name,
        "--transcribe-worker",
        "--transcribe-output",
        str(worker_output),
    ]
    LOGGER.info("Launching transcription worker process...")
    try:
        result = subprocess.run(command, text=True, timeout=7200)
        if worker_output.exists():
            if result.returncode != 0:
                LOGGER.warning("Transcription worker exited with code %s after writing output.", result.returncode)
            return json.loads(worker_output.read_text(encoding="utf-8"))
        if result.returncode != 0:
            raise PipelineError(f"Transcription worker failed with exit code {result.returncode}.")
        raise PipelineError("Transcription worker completed without producing output.")
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Transcription worker timed out after 7200 seconds.") from exc
    finally:
        if not remove_path_with_retries(worker_output, attempts=8, delay_sec=1.0):
            LOGGER.warning("Could not remove transcription worker output: %s", worker_output)


def chunk_transcript(segments: list[dict[str, Any]], limit: int = 1400) -> list[str]:
    """Split transcript segments into prompt-sized chunks."""
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0
    for segment in segments:
        line = f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}"
        line_tokens = estimate_tokens(line)
        if current_lines and current_tokens + line_tokens > limit:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0
        if line_tokens > limit:
            wrapped = textwrap.wrap(segment["text"], width=280) or [segment["text"]]
            start = float(segment["start"])
            end = float(segment["end"])
            step = max((end - start) / max(len(wrapped), 1), 0.01)
            for index, piece in enumerate(wrapped):
                piece_start = start + index * step
                piece_end = min(end, piece_start + step)
                piece_line = f"[{piece_start:.1f}s -> {piece_end:.1f}s] {piece}"
                if current_lines and current_tokens + estimate_tokens(piece_line) > limit:
                    chunks.append("\n".join(current_lines))
                    current_lines = []
                    current_tokens = 0
                current_lines.append(piece_line)
                current_tokens += estimate_tokens(piece_line)
            continue
        current_lines.append(line)
        current_tokens += line_tokens
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def build_prompt(chunk: str, max_clips: int) -> str:
    """Render the clip-selection prompt for one transcript chunk."""
    return PROMPT_TEMPLATE.format(chunk=chunk, max_clips=max_clips, min_sec=MIN_CLIP_SEC, max_sec=MAX_CLIP_SEC)


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST JSON and parse the JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise PipelineError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise PipelineError("Received invalid JSON from remote service.") from exc


def call_ollama(prompt: str) -> str:
    """Generate clip candidates from a local Ollama model."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024},
    }
    try:
        response = post_json("http://localhost:11434/api/generate", payload, timeout=120)
    except PipelineError as exc:
        raise RuntimeError("Ollama is not reachable. Run: ollama serve") from exc
    text = str(response.get("response", "")).strip()
    if not text:
        raise PipelineError("Ollama returned an empty response.")
    return text


def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Generate clip candidates from Gemini."""
    if not GEMINI_API_KEY:
        raise PipelineError("Gemini requested but GEMINI_API_KEY is empty.")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={parse.quote(GEMINI_API_KEY)}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = post_json(url, payload, timeout=120)
    candidates = response.get("candidates") or []
    if not candidates:
        raise PipelineError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(str(part.get("text", "")).strip() for part in parts if part.get("text"))
    if not text:
        raise PipelineError("Gemini returned an empty response.")
    return text


def parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Extract and parse the first JSON array found in model output."""
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model output.")
    snippet = match.group(0)
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse LLM JSON: {exc.msg}") from exc
    if not isinstance(data, list):
        raise ValueError("LLM output was not a JSON array.")
    return data


def validate_clip(clip: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized clip dict when it meets duration constraints."""
    try:
        start = float(clip["start"])
        end = float(clip["end"])
    except (KeyError, TypeError, ValueError):
        LOGGER.warning("Skipping clip with invalid timestamps: %s", clip)
        return None
    duration = end - start
    if start < 0 or end <= start:
        LOGGER.warning("Skipping clip with non-positive duration: %s", clip)
        return None
    if duration < MIN_CLIP_SEC or duration > MAX_CLIP_SEC:
        LOGGER.warning("Skipping clip outside duration limits (%ss): %s", duration, clip)
        return None
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "title": str(clip.get("title", "Untitled Clip")).strip() or "Untitled Clip",
        "hook": str(clip.get("hook", "")).strip(),
        "reason": str(clip.get("reason", "")).strip(),
    }


def overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Compute overlap as a fraction of the shorter clip."""
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    shorter = min(first["end"] - first["start"], second["end"] - second["start"])
    return overlap / shorter if shorter else 0.0


def clip_score(clip: dict[str, Any]) -> float:
    """Rank clips by descriptive richness and duration."""
    return len(clip.get("hook", "")) + len(clip.get("reason", "")) + len(clip.get("title", "")) * 2 + (clip["end"] - clip["start"])


def deduplicate_clips(clips: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the best non-overlapping clips up to the requested limit."""
    selected: list[dict[str, Any]] = []
    for clip in sorted(clips, key=clip_score, reverse=True):
        if any(overlap_ratio(clip, existing) > 0.25 for existing in selected):
            continue
        selected.append(clip)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: item["start"])


def generate_srt(segments: list[dict[str, Any]], start_sec: float, end_sec: float, out_path: Path) -> Path:
    """Create an SRT file for the clip window."""
    captions: list[str] = []
    caption_index = 1
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end <= start_sec or seg_start >= end_sec:
            continue
        start = max(0.0, seg_start - start_sec)
        end = max(start + 0.05, min(end_sec, seg_end) - start_sec)
        text = "\n".join(textwrap.wrap(str(segment["text"]).strip(), width=42)) or "..."
        captions.append("\n".join([str(caption_index), f"{sec_to_srt_ts(start)} --> {sec_to_srt_ts(end)}", text, ""]))
        caption_index += 1
    out_path.write_text("\n".join(captions), encoding="utf-8")
    return out_path


def escape_ffmpeg_subtitles_path(path: Path) -> str:
    """Escape a Windows path for FFmpeg subtitle filters."""
    resolved = path.resolve().as_posix()
    resolved = resolved.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")
    return resolved


def cut_and_format_clip(
    video_path: Path,
    clip: dict[str, Any],
    srt_path: Path,
    output_path: Path,
    encoder_flags: list[str] | None = None,
) -> Path:
    """Cut, crop, subtitle, and encode a vertical short clip with FFmpeg."""
    duration = round(clip["end"] - clip["start"], 3)
    subtitles_path = escape_ffmpeg_subtitles_path(srt_path)
    ffmpeg_binary = ensure_ffmpeg_on_path()
    if encoder_flags is None:
        _, encoder_flags = detect_encoder(prefer_quality=True)
    vf = (
        "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"subtitles='{subtitles_path}':force_style='FontSize=18,PrimaryColour=&HFFFFFF,"
        "OutlineColour=&H000000,Outline=2,Alignment=2,MarginV=60'"
    )
    command = [
        ffmpeg_binary,
        "-y",
        "-ss",
        str(clip["start"]),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-vf",
        vf,
        *encoder_flags,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        run_command(command, timeout=3600)
    except PipelineError as exc:
        LOGGER.error("FFmpeg failed for clip '%s': %s", clip.get("title", "untitled"), exc)
        raise
    return output_path


def analyze_chunks(chunks: list[str], use_gemini: bool) -> list[dict[str, Any]]:
    """Analyze transcript chunks and collect candidate clips."""
    all_candidates: list[dict[str, Any]] = []
    llm_fn = call_gemini if use_gemini else call_ollama
    for index, chunk in enumerate(chunks, start=1):
        LOGGER.info("Analyzing transcript chunk %s/%s...", index, len(chunks))
        prompt = build_prompt(chunk, MAX_CLIPS)
        try:
            raw = llm_fn(prompt)
        except RuntimeError as exc:
            LOGGER.warning("Ollama not ready, retrying in 10s... (%s)", exc)
            time.sleep(10)
            try:
                raw = llm_fn(prompt)
            except RuntimeError as exc2:
                LOGGER.error("Ollama failed after retry: %s", exc2)
                raise SystemExit(1) from exc2
        except PipelineError as exc:
            LOGGER.warning("LLM call failed for chunk %s: %s", index, exc)
            continue
        try:
            parsed = parse_llm_json(raw)
        except ValueError as exc:
            LOGGER.warning("Skipping chunk %s due to JSON parse failure: %s", index, exc)
            continue
        for candidate in parsed:
            normalized = validate_clip(candidate)
            if normalized:
                all_candidates.append(normalized)
    return all_candidates


def save_json(path: Path, payload: Any) -> None:
    """Write a JSON document with stable formatting."""
    import math

    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        try:
            value = float(obj)
        except (TypeError, ValueError):
            return str(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default_serializer),
        encoding="utf-8",
    )


def remove_path_with_retries(path: Path, attempts: int = 5, delay_sec: float = 1.0) -> bool:
    """Remove a file or directory, retrying briefly for Windows file locks."""
    for attempt in range(attempts):
        try:
            if not path.exists():
                return True
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True
        except OSError:
            if attempt == attempts - 1:
                break
            time.sleep(delay_sec)
    return not path.exists()


def cleanup_temp_files(paths: list[Path]) -> None:
    """Delete temporary files when they are no longer needed."""
    for path in paths:
        if not remove_path_with_retries(path, attempts=8, delay_sec=1.0):
            LOGGER.warning("Could not remove temp file: %s", path)


def process_clip_batch(
    video_path: Path,
    transcript: dict[str, Any],
    clips: list[dict[str, Any]],
    encoder_flags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Render the final MP4 clips and save metadata."""
    results: list[dict[str, Any]] = []
    segments = transcript["segments"]
    for index, clip in enumerate(clips, start=1):
        filename_root = f"{index:02d}_{safe_filename(clip['title'])}_{int(clip['start'])}s"
        srt_path = TEMP_DIR / f"{filename_root}.srt"
        output_path = OUTPUT_DIR / f"{filename_root}.mp4"
        metadata_path = OUTPUT_DIR / f"{filename_root}.json"
        try:
            generate_srt(segments, clip["start"], clip["end"], srt_path)
            cut_and_format_clip(video_path, clip, srt_path, output_path, encoder_flags=encoder_flags)
        except PipelineError:
            LOGGER.warning("Skipping clip after FFmpeg failure: %s", clip["title"])
            continue
        record = {
            **clip,
            "duration": round(clip["end"] - clip["start"], 3),
            "video_path": str(output_path),
            "subtitle_path": str(srt_path),
        }
        save_json(metadata_path, record)
        results.append(record)
        LOGGER.info("Saved clip: %s", output_path)
    save_json(OUTPUT_DIR / "clips_metadata.json", results)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="ClipForge AI video clipping pipeline")
    parser.add_argument("source", help="YouTube URL or local video file path")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--whisper", default=WHISPER_MODEL, choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--clips", type=int, default=MAX_CLIPS, help="Maximum clips to generate")
    parser.add_argument("--min", dest="min_sec", type=int, default=MIN_CLIP_SEC, help="Minimum clip seconds")
    parser.add_argument("--max", dest="max_sec", type=int, default=MAX_CLIP_SEC, help="Maximum clip seconds")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini instead of Ollama")
    parser.add_argument("--job-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--transcribe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transcribe-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Execute the full clip generation workflow."""
    LOGGER.info("Starting ClipForge...")
    job_id = safe_filename(args.job_id or uuid.uuid4().hex[:8], max_len=32)
    temp_files_to_cleanup: list[Path] = []
    transcript_path = TEMP_DIR / f"transcript_{job_id}.json"
    try:
        LOGGER.info("Stage 1/5: Downloading video...")
        video_path = get_video(args.source, job_id=job_id)
        if is_url(args.source):
            temp_files_to_cleanup.append(video_path.parent)
        LOGGER.info("Using video: %s", video_path)

        LOGGER.info("Stage 2/5: Starting transcription (GPU)...")
        transcript = transcribe(video_path)
        LOGGER.info("Transcription returned, saving to disk...")
        try:
            save_json(transcript_path, transcript)
            LOGGER.info("Transcript saved OK: %s", transcript_path)
        except Exception as exc:
            LOGGER.error("save_json FAILED: %s", exc)
            raise
        temp_files_to_cleanup.append(transcript_path)

        if not args.gemini:
            LOGGER.info("Stage 3/5: Restarting Ollama for analysis...")
            GPUOrchestrator.start_ollama()
        chunks = chunk_transcript(transcript["segments"])
        LOGGER.info("Built %s transcript chunk(s).", len(chunks))
        LOGGER.info("Stage 3/5: Analyzing viral moments with AI...")
        candidates = analyze_chunks(chunks, use_gemini=args.gemini)
        deduped = deduplicate_clips(candidates, args.clips)
        LOGGER.info("Selected %s clip(s) after validation and deduplication.", len(deduped))
        if not deduped:
            LOGGER.warning("No valid clips were found.")
            return []

        LOGGER.info("Stage 4/5: Cutting clips...")
        _, encoder_flags = detect_encoder(prefer_quality=True)
        results = process_clip_batch(video_path, transcript, deduped, encoder_flags=encoder_flags)
        LOGGER.info("Stage 5/5: Pipeline complete!")
        return results
    finally:
        cleanup_temp_files(temp_files_to_cleanup)


def _run_tests() -> None:
    """Run lightweight internal tests for helpers."""
    assert ts_to_sec("00:01:30") == 90.0
    assert sec_to_ts(90.5) == "00:01:30.500"
    name = safe_filename("this/is a noisy:title*with spaces and symbols?" * 2)
    assert "/" not in name and len(name) <= 60
    parsed = parse_llm_json('before [{"start": 0, "end": 30, "title": "A", "hook": "B", "reason": "C"}] after')
    assert parsed[0]["start"] == 0
    try:
        parse_llm_json("no array here")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_llm_json should raise ValueError when no JSON array exists.")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, update config, and run the pipeline."""
    global OLLAMA_MODEL, WHISPER_MODEL, MAX_CLIPS, MIN_CLIP_SEC, MAX_CLIP_SEC
    setup_logging()
    args = parse_args(argv)
    OLLAMA_MODEL = args.model
    WHISPER_MODEL = args.whisper
    MAX_CLIPS = args.clips
    MIN_CLIP_SEC = args.min_sec
    MAX_CLIP_SEC = args.max_sec
    if args.transcribe_worker:
        output_path = Path(args.transcribe_output)
        _transcribe_worker_to_file(Path(args.source), args.whisper, output_path)
        return 0
    if MIN_CLIP_SEC <= 0 or MAX_CLIP_SEC <= 0 or MAX_CLIP_SEC < MIN_CLIP_SEC:
        LOGGER.error("Clip duration settings are invalid. Ensure 0 < min <= max.")
        return 2
    try:
        results = run_pipeline(args)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1
    except PipelineError as exc:
        LOGGER.error(str(exc))
        return 1
    LOGGER.info("Done. Generated %s clip(s).", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
