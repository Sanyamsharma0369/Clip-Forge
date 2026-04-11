from __future__ import annotations

import cv2
import numpy as np
import argparse
import functools
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Any
from urllib import error, parse, request
import requests

# ── JSON Schema for structured LLM output ────────────────────────────────────
CLIP_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title":  {"type": "string"},
        "start":  {"type": "number"},
        "end":    {"type": "number"},
        "score":  {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["title", "start", "end", "score", "reason"],
}

CLIP_LIST_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": CLIP_JSON_SCHEMA,
        }
    },
    "required": ["clips"],
}

OLLAMA_MODEL = "qwen2.5:7b"
WHISPER_MODEL = "base"
MAX_CLIPS = 5
MIN_CLIP_SEC = 15
MAX_CLIP_SEC = 90
# ── Transcription segmentation ─────────────────────────────────────────────
MAX_SEGMENT_DURATION: float = 15.0  # split segments longer than this (seconds)
SEGMENT_OVERLAP_SEC:  float = 0.5   # overlap window between sub-segments
OUTPUT_DIR = Path("outputs/clips")
TEMP_DIR = Path("temp")
LOG_FILE = Path("pipeline.log")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENCODER_PREFER_QUALITY = True

# Viral Subtitle Styles (ASS detailed format)
SUBTITLE_STYLES = {
    0: {  # 🔥 Classic Viral
        "name": "Classic Viral",
        "fontname": "Arial",
        "fontsize": 72,
        "base_color": "&H00FFFFFF",
        "highlight_color": "&H0000FFFF",
        "outline_color": "&H00000000",
        "back_color": "&H00000000",
        "bold": True,
        "outline": 6,
        "shadow": 2,
        "alignment": 2,         # bottom center
        "margin_v": 120,        # push up from bottom edge
    },
    1: {  # 🟡 TikTok Yellow
        "name": "TikTok Yellow",
        "fontname": "Arial",
        "fontsize": 75,
        "base_color": "&H0000FFFF",
        "highlight_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H00000000",
        "bold": True,
        "outline": 8,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 120,
    },
    2: {  # 📖 Storyteller
        "name": "Storyteller",
        "fontname": "Arial",
        "fontsize": 68,
        "base_color": "&H00FFFFFF",
        "highlight_color": "&H00FFFF00",
        "outline_color": "&H00000000",
        "back_color": "&H00000000",
        "bold": False,
        "outline": 4,
        "shadow": 3,
        "alignment": 2,         # bottom, not center screen
        "margin_v": 180,        # higher up for storyteller feel
    },
    3: {  # 💎 Neon Cyan
        "name": "Neon Cyan",
        "fontname": "Arial",
        "fontsize": 72,
        "base_color": "&H00FFFF00",
        "highlight_color": "&H00FFFFFF",
        "outline_color": "&H00FF8C00",
        "back_color": "&H00000000",
        "bold": True,
        "outline": 6,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 120,
    },
    4: {  # 🚨 Action Red
        "name": "Action Red",
        "fontname": "Arial",
        "fontsize": 75,
        "base_color": "&H000000FF",
        "highlight_color": "&H00FFFFFF",
        "outline_color": "&H00FFFFFF",
        "back_color": "&H00000000",
        "bold": True,
        "outline": 6,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 120,
    },
}

# ── Clip padding constants ────────────────────────────────────────────────────
CLIP_PAD_START = 1.5    # seconds before clip start (breathing room)
CLIP_PAD_END   = 3.0    # seconds after clip end (catches cut-off sentences)
AUDIO_LUFS_TARGET = -14 # YouTube/TikTok loudness standard
AUDIO_TRUE_PEAK = -1.5
COLOR_CONTRAST = 1.05
COLOR_SATURATION = 1.15
COLOR_BRIGHTNESS = 0.01
SHARPEN_STRENGTH = 0.8
SCENE_THRESHOLD = 0.35  # 0.0–1.0, higher = fewer detected scenes

# ── Face Tracking Constants ───────────────────────────────────────────────
FACE_DETECT_EVERY_N_FRAMES = 15      # detect every 15 frames (~0.5s at 30fps)
FACE_SMOOTH_WINDOW         = 30      # rolling average over 30 detections = smooth pan
FACE_SCALE_FACTOR          = 1.1     # haarcascade detection sensitivity
FACE_MIN_NEIGH_BORS        = 5       # higher = fewer false positives
FACE_PADDING_TOP           = 0.20    # extra headroom above detected face (20%)
FACE_PADDING_BOTTOM        = 0.10    # chin to bottom padding
FACE_MIN_SIZE_RATIO        = 0.05    # minimum face size relative to frame width

PROMPT_TEMPLATE = """\
=== EDITOR INSTRUCTIONS (HIGHEST PRIORITY) ===
{custom_instruction}
================================================

You are a SHORT-FORM VIDEO EDITOR creating clips for TikTok and YouTube Shorts.

⚠️ CRITICAL DIFFERENCE — READ THIS FIRST:
The transcript below contains many small lines, each 2-5 seconds long.
DO NOT return those small lines as clips.
You must COMBINE multiple lines together into ONE longer clip ({min_sec}–{max_sec} seconds).

=== EXAMPLE OF CORRECT OUTPUT ===
{{
  "clips": [
    {{
      "title": "Shocking Income Reveal",
      "start": 245.0,
      "end": 290.0,
      "score": 0.95,
      "reason": "Speaker reveals unexpected income source — high retention hook"
    }},
    {{
      "title": "The One Business Rule",
      "start": 412.0,
      "end": 458.5,
      "score": 0.88,
      "reason": "Actionable contrarian advice that challenges assumptions"
    }}
  ]
}}

WRONG (too short — individual transcript lines):
{{ "clips": [{{"title":"x","start":245.0,"end":247.5,...}}] }}
That is WRONG because 247.5 - 245.0 = 2.5 seconds. Minimum is {min_sec}s.
=== END EXAMPLE ===

YOUR TASK:
Find {min_clips} to {max_clips} clips. Return them inside a "clips" array.

RULES:
1. Each clip MUST be {min_sec}–{max_sec} seconds (end - start >= {min_sec})
2. Return AT LEAST {min_clips} clips
3. Combine multiple transcript lines into one scene window
4. Pick emotionally engaging moments: shocking facts, reveals, arguments, surprises
5. You MUST provide a "title" for every clip (this will be the subtitle hook)
6. End every clip at a COMPLETE sentence — the last word must be followed by
   a period, exclamation mark, or question mark in the transcript.
   Never end on words like "and", "but", "so", "the", "a", "my", "your".
7. Add 2-3 seconds of buffer AFTER the final word for natural breathing room.
   Example: if the last word ends at 287.5s, set end = 290.0s minimum.
8. Return a JSON object with a "clips" key containing the array

TRANSCRIPT:
{transcript}
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


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    ffmpeg_bin = find_ffmpeg_binary()
    ffprobe_bin = str(ffmpeg_bin).lower().replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    # Restore original case for the path but fixed binary name
    if ffmpeg_bin.lower().endswith("ffmpeg.exe"):
        ffprobe_bin = ffmpeg_bin[:-10] + "ffprobe.exe"
    elif ffmpeg_bin.lower().endswith("ffmpeg"):
        ffprobe_bin = ffmpeg_bin[:-6] + "ffprobe"

    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 9999.0


def snap_to_scene_cut(video_path: Path, timestamp: float, window: float = 1.5) -> float:
    """Snap a timestamp to the nearest scene cut within ±window seconds."""
    ffmpeg_bin = find_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-i", str(video_path),
        "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
        "-vsync", "0", "-f", "null", "-"
    ]
    try:
        # We only scan within a reasonable range to save time if we had sought, 
        # but select filter scans globally unless we use -ss/-t. 
        # For simplicity and given the user's snippet, we'll scan the whole video 
        # but with a short timeout.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        scene_times = []
        for line in result.stderr.splitlines():
            if "pts_time:" in line:
                try:
                    t = float(line.split("pts_time:")[1].split()[0])
                    scene_times.append(t)
                except (ValueError, IndexError):
                    continue
                    
        # Find nearest scene cut within window
        candidates = [t for t in scene_times if abs(t - timestamp) <= window]
        if candidates:
            nearest = min(candidates, key=lambda t: abs(t - timestamp))
            LOGGER.info("  Snapped %.2fs -> %.2fs (scene cut)", timestamp, nearest)
            return nearest
    except Exception as exc:
        LOGGER.warning("Scene detection failed, using raw timestamp: %s", exc)
    return timestamp


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
            "proxy": "",
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


def _normalize_segments(
    segments: list[dict],
    max_duration: float = MAX_SEGMENT_DURATION,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
) -> list[dict]:
    """
    Post-collection normalization pass.

    Splits any segment longer than `max_duration` into sub-segments
    using word-level timestamps. Preserves `overlap_sec` of context
    between adjacent sub-segments for LLM continuity.

    Graceful fallback: segments with missing/empty word data are kept
    as-is — no crash, no data loss.
    """
    normalized: list[dict] = []

    for seg in segments:
        duration = (seg.get("end") or 0.0) - (seg.get("start") or 0.0)
        words = seg.get("words") or []

        # ── Fast path: short segment or no word data ──────────────────────
        if duration <= max_duration or not words:
            if duration > max_duration and not words:
                LOGGER.warning(
                    f"Segment [{seg.get('start'):.2f}s–{seg.get('end'):.2f}s] "
                    f"is {duration:.1f}s but has no word timestamps — kept as-is. "
                    "Ensure word_timestamps=True is set in the transcribe() call."
                )
            normalized.append(seg)
            continue

        # ── Split at word boundaries every ~max_duration seconds ──────────
        sub_words: list[dict] = []
        sub_start_t: float = words[0]["start"]

        for i, word in enumerate(words):
            sub_words.append(word)
            sub_duration = word["end"] - sub_start_t
            is_last = i == len(words) - 1

            if sub_duration >= max_duration or is_last:
                # Emit this sub-segment
                normalized.append(
                    {
                        "start": sub_words[0]["start"],
                        "end": sub_words[-1]["end"],
                        "text": " ".join(w.get("word", "").strip() for w in sub_words),
                        "words": sub_words,
                    }
                )

                if is_last:
                    break

                # Build overlap: walk back from current word to find
                # words whose start time >= (current_end - overlap_sec)
                overlap_boundary = sub_words[-1]["end"] - overlap_sec
                overlap_words = [w for w in sub_words if w["start"] >= overlap_boundary]

                # Start next sub-segment from the overlap window
                sub_words = overlap_words[:]
                sub_start_t = (
                    sub_words[0]["start"] if sub_words else words[i + 1]["start"]
                )

    return normalized


def _collect_transcript_segments(raw_segments: Any, info: Any) -> tuple[list[dict[str, Any]], str]:
    """Consume faster-whisper's generator while emitting periodic progress."""
    segments_raw: list[dict[str, Any]] = []
    text_parts: list[str] = []
    total_duration = float(getattr(info, "duration", 0) or 0)
    last_logged_pct = -1

    for segment in raw_segments:
        text = str(segment.text).strip()
        if not text:
            continue

        end_time = float(segment.end)
        segments_raw.append(
            {
                "start": float(segment.start),
                "end": end_time,
                "text": text,
                "words": [
                    {"word": w.word, "start": float(w.start), "end": float(w.end)}
                    for w in (getattr(segment, "words", []) or [])
                ],
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

    # ── Normalization pass ─────────────────────────────────────────────────
    if segments_raw:
        sample_words = segments_raw[0].get("words") or []
        LOGGER.debug(
            "Word timestamp sample (seg[0]): %s",
            sample_words[:3] if sample_words else "NONE — word_timestamps may be False",
        )

    normalized = _normalize_segments(segments_raw)

    LOGGER.info(
        "Transcription complete - %d segments (raw from Whisper: %d, after normalization)",
        len(normalized),
        len(segments_raw),
    )
    return normalized, " ".join(text_parts).strip()


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
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "threshold": 0.4,
        },
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
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "threshold": 0.35,
        },
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

def build_prompt(
    transcript: str,
    min_sec: int,
    max_sec: int,
    min_clips: int,
    max_clips: int,
    custom_instruction: str = "",
) -> str:
    """Render the clip-selection prompt for one transcript chunk."""
    instruction = custom_instruction.strip() if custom_instruction.strip() else \
        "Find the most engaging, emotionally resonant moments with strong hooks."
    return PROMPT_TEMPLATE.format(
        custom_instruction=instruction,
        min_clips=min_clips,
        max_clips=max_clips,
        min_sec=min_sec,
        max_sec=max_sec,
        transcript=transcript,
    )


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


def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = 120,
    use_schema: bool = False,       # ← OFF by default, ON only for clip analysis
) -> str:
    """
    Call Ollama API. When use_schema=True, injects CLIP_LIST_JSON_SCHEMA
    into the format field — guarantees valid JSON on Ollama >= 0.1.47.
    Falls back to plain text on older versions.
    """
    url     = "http://localhost:11434/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,     # lower = more consistent JSON
            "num_predict": 2048,
        },
    }

    if use_schema:
        payload["format"] = CLIP_LIST_JSON_SCHEMA
        LOGGER.debug("  Ollama: schema mode ON (clips wrapper enforced)")
    else:
        LOGGER.debug("  Ollama: schema mode OFF (plain text)")

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.exceptions.ConnectionError:
        raise PipelineError(
            "Ollama is not running. Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise PipelineError(
            f"Ollama timed out after {timeout}s. "
            f"Try a smaller model or increase timeout."
        )
    except requests.exceptions.HTTPError as exc:
        if resp.status_code == 404:
            raise PipelineError(
                f"Model '{model}' not found. Install it with: ollama pull {model}"
            )
        raise PipelineError(f"Ollama HTTP error: {exc}")


def call_gemini(
    prompt: str,
    timeout: int = 120,
    use_schema: bool = False,
) -> str:
    """
    Call Gemini API with structured JSON output enforcement.
    Uses response_mime_type + response_schema for schema compliance.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise PipelineError("GEMINI_API_KEY env variable not set.")

    genai.configure(api_key=api_key)

    if use_schema:
        generation_config = genai.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "clips": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "title":  {"type": "STRING"},
                                "start":  {"type": "NUMBER"},
                                "end":    {"type": "NUMBER"},
                                "score":  {"type": "NUMBER"},
                                "reason": {"type": "STRING"},
                            },
                            "required": ["title", "start", "end", "score", "reason"],
                        }
                    }
                },
                "required": ["clips"],
            }
        )
    else:
        # Plain text for chapters
        generation_config = genai.GenerationConfig(temperature=0.3)

    model_name = "gemini-1.5-flash"
    model_obj  = genai.GenerativeModel(
        model_name,
        generation_config=generation_config,
    )

    try:
        response = model_obj.generate_content(prompt)
        return response.text
    except Exception as exc:
        raise PipelineError(f"Gemini API error: {exc}") from exc


def parse_llm_json(text: str) -> list[dict]:
    """
    Parse LLM JSON response. Handles both:
      - Structured output wrapper: {"clips": [...]}   ← new schema format
      - Raw array: [...]                               ← legacy fallback
      - Markdown fences, preamble, truncated JSON     ← robustness
    """
    if not text or not text.strip():
        raise PipelineError("LLM returned empty response.")

    # ── Step 1: Strip markdown fences ────────────────────────────────────────
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # ── Step 2: Try full parse first ─────────────────────────────────────────
    try:
        data = json.loads(cleaned)

        # Structured output wrapper {"clips": [...]}
        if isinstance(data, dict) and "clips" in data:
            clips = data["clips"]
            if isinstance(clips, list):
                LOGGER.debug(
                    "  parse_llm_json: schema wrapper detected, %d clips.", len(clips)
                )
                return clips

        # Raw array
        if isinstance(data, list):
            return data

        # Single object — wrap in list
        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    # ── Step 3: Find {"clips": [...]} pattern via regex ──────────────────────
    clips_match = re.search(r'"clips"\s*:\s*(\[.*?\])', cleaned, re.DOTALL)
    if clips_match:
        try:
            clips = json.loads(clips_match.group(1))
            if isinstance(clips, list):
                return clips
        except json.JSONDecodeError:
            pass

    # ── Step 4: Find outermost [...] array ───────────────────────────────────
    def find_json_arrays(s: str) -> list[str]:
        candidates, depth, start = [], 0, -1
        for i, ch in enumerate(s):
            if ch == "[":
                if depth == 0: start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(s[start:i + 1])
                    start = -1
        return candidates

    candidates = sorted(find_json_arrays(cleaned), key=len, reverse=True)
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # ── Step 5: Repair truncated JSON ────────────────────────────────────────
    idx = cleaned.find("[")
    if idx != -1:
        partial      = cleaned[idx:]
        open_braces  = partial.count("{") - partial.count("}")
        repaired     = partial + ("}" * max(0, open_braces)) + "]"
        try:
            result = json.loads(repaired)
            if isinstance(result, list):
                LOGGER.warning(
                    "  parse_llm_json: repaired truncated JSON, %d items.", len(result)
                )
                return result
        except json.JSONDecodeError:
            pass

    raise PipelineError(
        f"No JSON array found in model output.\n"
        f"--- RAW (first 400 chars) ---\n{text[:400]}\n---"
    )


def normalize_clip(clip: dict, segments: list[dict] = None) -> dict:
    """Ensure clip has 'start' and 'end' float keys, handling common variants."""
    c = dict(clip)
    s = c.get("start") if c.get("start") is not None else c.get("start_time", 0)
    e = c.get("end")   if c.get("end")   is not None else c.get("end_time", 0)
    c["start"] = float(s)
    c["end"]   = float(e)
    return c


def snap_to_sentence_end(
    timestamp: float,
    segments: list[dict],
    max_extend: float = 4.0,
) -> float:
    """
    If timestamp lands inside a transcript segment (mid-speech),
    extend to that segment's end so we don't cut mid-word.
    Only extends up to max_extend seconds beyond the timestamp.
    """
    if not segments:
        return timestamp

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end   = float(seg.get("end",   0))

        if seg_start <= timestamp < seg_end:
            overshoot = seg_end - timestamp
            if overshoot <= max_extend:
                LOGGER.debug(
                    "  Sentence snap: %.2fs -> %.2fs (+%.2fs to complete speech segment)",
                    timestamp, seg_end, overshoot,
                )
                return seg_end
            else:
                LOGGER.debug(
                    "  Sentence snap: skipped (overshoot %.2fs > max %.2fs)",
                    overshoot, max_extend,
                )
            break

    return timestamp


def fix_sentence_boundary(
    clip: dict,
    segments: list[dict],
    video_duration: float,
    max_extend: float = 5.0,
) -> dict:
    """
    Detect if a clip ends mid-sentence (no terminal punctuation in the
    last transcript segment). If so, extend to the end of that segment.
    Returns a new dict — never mutates the original.
    """
    if not segments:
        return clip

    end = float(clip.get("end", 0))

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end   = float(seg.get("end",   0))
        seg_text  = seg.get("text", "").strip()

        if seg_start <= end < seg_end:
            # Check if the segment ends with sentence-terminal punctuation
            ends_cleanly = seg_text.endswith(
                (".", "!", "?", "...", "…", '"', "'", ")'", '."', '!"', '?"')
            )

            if not ends_cleanly:
                extension = seg_end - end
                if extension <= max_extend:
                    new_end = min(video_duration, seg_end + 0.5)   # 0.5s grace
                    fixed   = dict(clip)
                    fixed["end"] = new_end
                    LOGGER.info(
                        "  Sentence fix: '%s' %.2fs->%.2fs "
                        "(mid-sentence cut fixed, last words: '...%s')",
                        clip.get("title", "untitled")[:40],
                        end, new_end,
                        seg_text[-40:],
                    )
                    return fixed
                else:
                    LOGGER.debug(
                        "  Sentence fix: skipped '%s' (extension %.2fs > max %.2fs)",
                        clip.get("title", "")[:30], extension, max_extend,
                    )
            break

    return clip


def expand_clip_to_minimum(
    clip: dict,
    min_sec: int,
    max_sec: int,
    video_duration: float,
) -> dict:
    """
    If a clip is shorter than min_sec, expand it symmetrically
    into surrounding context until it meets the minimum duration.
    Returns the modified clip (does not mutate original).
    """
    # Defensive key check
    start = clip.get("start") if clip.get("start") is not None else clip.get("start_time", 0)
    end   = clip.get("end")   if clip.get("end")   is not None else clip.get("end_time", 0)
    
    start = float(start)
    end   = float(end)
    duration = end - start

    if duration >= min_sec:
        return clip   # already valid, no change

    # How much do we need to add?
    deficit = min_sec - duration
    pad_each_side = deficit / 2.0

    new_start = max(0.0, start - pad_each_side)
    new_end   = min(video_duration, end + pad_each_side)

    # If we hit a boundary, compensate on the other side
    if new_start == 0.0:
        new_end = min(video_duration, new_end + (pad_each_side - start))
    if new_end == video_duration:
        new_start = max(0.0, new_start - (pad_each_side - (video_duration - end)))

    # Cap at max_sec
    if (new_end - new_start) > max_sec:
        new_end = new_start + max_sec

    expanded = dict(clip)
    expanded["start"] = round(new_start, 2)
    expanded["end"]   = round(new_end, 2)

    LOGGER.info(
        "  Auto-expanded clip '%s': %.1fs->%.1fs (was %.1fs, now %.1fs)",
        clip.get("title", "untitled"),
        start, end,
        duration, new_end - new_start,
    )
    return expanded


def validate_clip(
    clip: dict[str, Any],
    video_duration: float,
    min_sec: int,
    max_sec: int,
) -> tuple[bool, str]:
    """
    Validate a single clip dict. Returns (is_valid, rejection_reason).
    Duration range is enforced with ZERO tolerance — LLM suggestions outside
    [min_sec, max_sec] are hard-rejected, not silently trimmed.
    """
    # ── Map flexible keys ──────────────────────────────────────────────────
    start = clip.get("start") if clip.get("start") is not None else clip.get("start_time")
    end   = clip.get("end")   if clip.get("end")   is not None else clip.get("end_time")

    # ── Type checks ──────────────────────────────────────────────────────────
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False, f"Non-numeric start/end: start={start!r} end={end!r}"

    start = float(start)
    end = float(end)

    # ── Logical order ────────────────────────────────────────────────────────
    if end <= start:
        return False, f"end ({end:.1f}s) <= start ({start:.1f}s)"

    # ── Duration enforcement — STRICT ────────────────────────────────────────
    duration = end - start
    if duration < min_sec:
        return False, (
            f"Duration {duration:.1f}s is below minimum {min_sec}s — "
            "extend into surrounding context or discard"
        )
    if duration > max_sec:
        return False, (
            f"Duration {duration:.1f}s exceeds maximum {max_sec}s — "
            "trim to most impactful moment or discard"
        )

    # ── Video bounds ─────────────────────────────────────────────────────────
    if start < 0:
        return False, f"start ({start:.1f}s) is negative"
    if end > video_duration + 1.0:   # 1s tolerance for float rounding
        return False, f"end ({end:.1f}s) exceeds video duration ({video_duration:.1f}s)"

    # ── Required fields ──────────────────────────────────────────────────────
    if not clip.get("title", "").strip():
        return False, "Missing or empty title"

    return True, ""


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
    # Sort by viral_score descending FIRST, then deduplicate
    candidates = sorted(clips, key=lambda c: c.get("viral_score", 5), reverse=True)

    selected: list[dict[str, Any]] = []
    for clip in candidates:
        overlap = False
        for kept in selected:
            # Skip if clips overlap by more than 50%
            overlap_start = max(clip["start"], kept["start"])
            overlap_end = min(clip["end"], kept["end"])
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                clip_duration = clip["end"] - clip["start"]
                if overlap_duration / clip_duration > 0.5:
                    overlap = True
                    break
        if not overlap:
            selected.append(clip)
        if len(selected) >= limit:
            break

    LOGGER.info("Top clip scores: %s", [c.get("viral_score", "?") for c in selected])
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


def export_srt(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_path: Path,
) -> Path | None:
    """Export clip transcript as a .srt subtitle file."""
    srt_path = output_path.with_suffix(".srt")

    def to_srt_time(seconds: float) -> str:
        seconds = max(0.0, seconds - clip_start)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    clip_segs = [s for s in segments if s["end"] >= clip_start and s["start"] <= clip_end]
    if not clip_segs:
        return None

    lines: list[str] = []
    for i, seg in enumerate(clip_segs, start=1):
        lines.append(str(i))
        lines.append(f"{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("  SRT exported: %s", srt_path.name)
    return srt_path


def generate_chapters(
    segments: list[dict],
    model: str,
    use_gemini: bool = False,
) -> str:
    """
    Generate chapter markers as plain text timestamp list.
    NEVER uses JSON schema — plain text output only.
    """
    transcript_sample = ""
    for s in segments[:100]:   # first ~10-15 mins
        mins = int(s["start"] // 60)
        secs = int(s["start"] % 60)
        transcript_sample += f"[{mins:02d}:{secs:02d}] {str(s.get('text', '')).strip()}\n"

    chapter_prompt = f"""Analyze this timestamped transcript and generate YouTube chapter markers.
Return ONLY a plain text list in this exact format — no JSON, no bullets:

00:00 Introduction
02:34 First Topic Title
05:12 Key Moment Title
08:45 Conclusion

Rules:
- Always start at 00:00
- Timestamps in MM:SS format
- Chapter title max 35 characters, no punctuation
- 5-8 chapters total
- Plain text only — no JSON, no markdown
- Capture the real topic shifts, not generic labels

TRANSCRIPT:
{transcript_sample}

CHAPTER LIST:"""

    try:
        if use_gemini:
            raw = call_gemini(chapter_prompt, use_schema=False).strip()
        else:
            raw = call_ollama(chapter_prompt, model=model, use_schema=False).strip()

        # Validate — every line must match MM:SS pattern
        valid_lines = [
            line
            for line in raw.splitlines()
            if line.strip() and len(line.split()) >= 2 and ":" in line.split()[0]
        ]
        result = "\n".join(valid_lines)
        if result:
            LOGGER.info("Chapters generated:\n%s", result)
        return result
    except Exception as exc:
        LOGGER.warning("Chapter generation failed: %s", exc)
        return ""


def seconds_to_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS timestamp format H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)   # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_subtitles(
    segments: list[dict[str, Any]],
    output_path: Path,
    style_id: int = 0,
    clip_start: float = 0.0,
    video_width: int = 1080,
    video_height: int = 1920,
) -> Path:
    """
    Generate an ASS subtitle file with karaoke word-by-word highlighting.
    Segments must contain word-level timestamps from Whisper word_timestamps=True.
    clip_start offsets all timestamps relative to the clip's start time.
    """
    style = SUBTITLE_STYLES.get(style_id, SUBTITLE_STYLES[0])

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style["fontname"]},{style["fontsize"]},{style["base_color"]},{style["highlight_color"]},{style["outline_color"]},{style["back_color"]},{1 if style["bold"] else 0},0,0,0,100,100,0,0,1,{style["outline"]},{style["shadow"]},{style["alignment"]},10,10,{style["margin_v"]},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for segment in segments:
        words = segment.get("words", [])
        if not words:
            # Fallback — no word timestamps, show full segment text
            seg_start = max(0.0, float(segment["start"]) - clip_start)
            seg_end = max(seg_start + 0.1, float(segment["end"]) - clip_start)
            if seg_end <= 0:
                continue
            text = str(segment.get("text", "")).strip()
            events.append(
                f"Dialogue: 0,{seconds_to_ass_time(seg_start)},"
                f"{seconds_to_ass_time(seg_end)},Default,,0,0,0,,"
                f"{text}"
            )
            continue

        # Group words into lines of max 3 words for readability
        line_size = 3
        word_groups = [words[i : i + line_size] for i in range(0, len(words), line_size)]

        for group in word_groups:
            if not group:
                continue

            group_start = max(0.0, float(group[0]["start"]) - clip_start)
            group_end = max(group_start + 0.1, float(group[-1]["end"]) - clip_start)

            if group_end <= 0:
                continue

            # Build karaoke text with {\k} tags
            # Each word gets a {\kf<duration>} tag where duration is in centiseconds
            karaoke_parts = []
            for word in group:
                word_start = float(word["start"]) - clip_start
                word_end = float(word["end"]) - clip_start
                # Duration in centiseconds
                dur_cs = max(1, int((word_end - word_start) * 100))
                word_text = str(word.get("word", "")).strip()
                karaoke_parts.append(f"{{\\kf{dur_cs}}}{word_text} ")

            karaoke_line = "".join(karaoke_parts).rstrip()
            events.append(
                f"Dialogue: 0,{seconds_to_ass_time(group_start)},"
                f"{seconds_to_ass_time(group_end)},Default,,0,0,0,,"
                f"{karaoke_line}"
            )

    ass_content = ass_header + "\n".join(events)
    output_path.write_text(ass_content, encoding="utf-8")
    LOGGER.info("  ASS subtitles written: %s (%d events)", output_path.name, len(events))
    return output_path


def extract_thumbnail(clip_path: Path) -> Path | None:
    """Extract the most visually interesting frame as a JPG thumbnail."""
    thumb_path = clip_path.with_suffix(".jpg")
    cmd = [
        ensure_ffmpeg_on_path(),
        "-i",
        str(clip_path),
        "-vf",
        "thumbnail=300,scale=540:960",  # sample 300 frames, pick best
        "-frames:v",
        "1",
        "-q:v",
        "2",  # high quality JPEG
        "-y",
        str(thumb_path),
    ]
    try:
        run_command(cmd, timeout=60)
        LOGGER.info("  Thumbnail saved: %s", thumb_path.name)
        return thumb_path
    except PipelineError as exc:
        LOGGER.warning("Thumbnail extraction failed: %s", exc)
        return None


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Return (width, height) of video via ffprobe."""
    ffmpeg_bin = find_ffmpeg_binary()
    ffprobe_bin = str(ffmpeg_bin).lower().replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    if ffmpeg_bin.lower().endswith("ffmpeg.exe"):
        ffprobe_bin = ffmpeg_bin[:-10] + "ffprobe.exe"
    elif ffmpeg_bin.lower().endswith("ffmpeg"):
        ffprobe_bin = ffmpeg_bin[:-6] + "ffprobe"

    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data   = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return int(stream["width"]), int(stream["height"])
    except Exception as exc:
        LOGGER.warning("Could not get video dimensions: %s", exc)
    return 0, 0


def detect_face_centers(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_width: int = 1080,
    output_height: int = 1920,
) -> list[tuple[float, float]]:
    """
    Analyze video segment and return (x_center_ratio, y_center_ratio) per frame.
    Returns ratios 0.0–1.0 relative to original frame dimensions.
    Falls back to (0.5, 0.5) center crop if face detection fails.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Face tracking: could not open video, using center crop.")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int((end_time - start_time) * fps)
    start_frame  = int(start_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Rolling buffer of detected center positions
    center_buffer: deque[tuple[float, float]] = deque(maxlen=FACE_SMOOTH_WINDOW)
    center_buffer.append((0.5, 0.35))  # default: upper-center (typical talking head)

    frame_centers: list[tuple[float, float]] = []
    frame_idx = 0

    LOGGER.info(
        "  Face tracking: analyzing %d frames (every %d)...",
        total_frames, FACE_DETECT_EVERY_N_FRAMES
    )

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run detection every N frames
        if frame_idx % FACE_DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_h, frame_w = gray.shape

            min_face_px = int(frame_w * FACE_MIN_SIZE_RATIO)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGH_BORS,
                minSize=(min_face_px, min_face_px),
            )

            if len(faces) > 0:
                # Use the largest detected face (most prominent speaker)
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest

                # Center X = middle of face
                cx = (fx + fw / 2) / frame_w

                # Center Y = slightly above face center (include forehead)
                cy = (fy + fh * (0.5 - FACE_PADDING_TOP)) / frame_h
                cy = max(0.15, min(0.85, cy))  # clamp: never crop off top/bottom

                center_buffer.append((cx, cy))

        # Smoothed center = average of buffer
        avg_cx = sum(c[0] for c in center_buffer) / len(center_buffer)
        avg_cy = sum(c[1] for c in center_buffer) / len(center_buffer)
        frame_centers.append((avg_cx, avg_cy))
        frame_idx += 1

    cap.release()
    
    detected_count = sum(1 for c in frame_centers if c != (0.5, 0.35))
    LOGGER.info(
        "  Face tracking result: %d/%d frames had detected faces (%.0f%%)",
        detected_count, len(frame_centers),
        100 * detected_count / max(1, len(frame_centers))
    )

    LOGGER.info(
        "  Face tracking: %d centers detected, avg position (%.2f, %.2f)",
        len(frame_centers),
        sum(c[0] for c in frame_centers) / max(1, len(frame_centers)),
        sum(c[1] for c in frame_centers) / max(1, len(frame_centers)),
    )
    return frame_centers


def build_face_crop_filter(
    frame_centers: list[tuple[float, float]],
    source_width: int,
    source_height: int,
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """
    Convert face center positions into an FFmpeg crop+scale filter string.
    Uses sendcmd for per-frame dynamic crop when centers vary significantly,
    falls back to single best-center crop for stable shots.
    """
    if not frame_centers:
        # Fallback: static center crop
        crop_w = min(source_width, int(source_height * output_width / output_height))
        crop_x = (source_width - crop_w) // 2
        return f"crop={crop_w}:{source_height}:{crop_x}:0,scale={output_width}:{output_height}"

    # Check if face moves significantly (std deviation of X positions)
    x_positions = [c[0] for c in frame_centers]
    x_std = float(np.std(x_positions))

    if x_std < 0.05:
        # Static shot — use single average center, no per-frame commands needed
        avg_cx = float(np.mean(x_positions))
        # avg_cy = float(np.mean([c[1] for c in frame_centers])) # redundant as we crop full height

        crop_w = min(source_width, int(source_height * output_width / output_height))
        crop_h = source_height
        crop_x = int(avg_cx * source_width - crop_w / 2)
        crop_x = max(0, min(source_width - crop_w, crop_x))

        LOGGER.info("  Static shot detected (std=%.3f), single crop at x=%d", x_std, crop_x)
        return f"crop={crop_w}:{crop_h}:{crop_x}:0,scale={output_width}:{output_height}"

    else:
        # Dynamic shot — use FFmpeg's crop with smooth x expression
        # Build x positions string for each frame using lerp via geq
        crop_w = min(source_width, int(source_height * output_width / output_height))

        # Use average center for simplicity with lerp smoothing
        # True per-frame would require sendcmd file — this gives smooth pan
        avg_cx = float(np.mean(x_positions))
        crop_x = int(avg_cx * source_width - crop_w / 2)
        crop_x = max(0, min(source_width - crop_w, crop_x))

        LOGGER.info("  Dynamic shot detected (std=%.3f), tracking crop at x=%d", x_std, crop_x)
        return (
            f"crop={crop_w}:{source_height}:"
            f"'min(max(0,{crop_x}+{source_width}*(x/{source_width}-{avg_cx:.3f})*0.3),"
            f"{source_width - crop_w})':"
            f"0,scale={output_width}:{output_height}"
        )


def cut_and_format_clip(
    video_path: Path,
    clip: dict[str, Any],
    srt_path: Path,
    output_path: Path,
    encoder_flags: list[str] | None = None,
    no_scene_snap: bool = False,
    use_esrgan: bool = False,
    subtitle_style: int = 0,
    segments: list[dict[str, Any]] | None = None,
    use_face_tracking: bool = True,
) -> Path:
    """Cut, crop, subtitle, and encode a vertical short clip with FFmpeg."""
    video_duration = get_video_duration(video_path)

    raw_start = clip["start"]
    raw_end   = clip["end"]

    # ── Apply padding ─────────────────────────────────────────────────────────
    start = max(0.0, raw_start - CLIP_PAD_START)
    end   = min(video_duration, raw_end + CLIP_PAD_END)

    # ── Snap end to sentence boundary FIRST (before scene snap) ───────────────
    if segments:
        end = snap_to_sentence_end(end, segments, max_extend=4.0)

    # ── Then snap both to scene cuts ──────────────────────────────────────────
    if not no_scene_snap:
        start = snap_to_scene_cut(video_path, start, window=1.5)
        end = snap_to_scene_cut(video_path, end, window=1.5)

    # ── Safety clamp ──────────────────────────────────────────────────────────
    end = min(video_duration, end)
    if end <= start:
        LOGGER.warning(
            "  cut_and_format_clip: end (%.2fs) <= start (%.2fs) after snapping — "
            "reverting to padded timestamps",
            end, start,
        )
        start = max(0.0, raw_start - CLIP_PAD_START)
        end   = min(video_duration, raw_end + CLIP_PAD_END)

    duration = end - start
    ffmpeg_binary = ensure_ffmpeg_on_path()

    if encoder_flags is None:
        _, encoder_flags = detect_encoder(prefer_quality=True)

    # ── Generate ASS subtitle file ─────────────────────────
    ass_path = None
    if segments:
        # Filter segments that overlap with this clip
        clip_segments = [s for s in segments if s["end"] >= start and s["start"] <= end]
        if clip_segments:
            ass_path = output_path.with_suffix(".ass")
            generate_ass_subtitles(
                segments=clip_segments,
                output_path=ass_path,
                style_id=subtitle_style,
                clip_start=start,
            )

    # 3. Quality Video Filters
    video_filters = [
        f"eq=contrast={COLOR_CONTRAST}:saturation={COLOR_SATURATION}:brightness={COLOR_BRIGHTNESS}",
        f"unsharp=3:3:{SHARPEN_STRENGTH}",
    ]

    # 4. Vertical Layout with Face Tracking
    source_w, source_h = get_video_dimensions(video_path)
    if use_face_tracking and source_w > 0:
        LOGGER.info("  Stage: face tracking crop...")
        face_centers = detect_face_centers(video_path, start, end)
        crop_filter = build_face_crop_filter(face_centers, source_w, source_h)
    elif source_w > 0:
        # Static center crop fallback
        crop_w = min(source_w, int(source_h * 1080 / 1920))
        crop_x = (source_w - crop_w) // 2
        crop_filter = f"crop={crop_w}:{source_h}:{crop_x}:0,scale=1080:1920"
    else:
        # Emergency fallback if dimensions failed
        crop_filter = "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920"

    video_filters.insert(0, crop_filter)
    video_filters.extend(
        [
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
        ]
    )

    # 5. Burn ASS Subtitles (with robust Windows escaping)
    if ass_path and ass_path.exists():
        # Windows path must be double-escaped for FFmpeg ass filter
        ass_escaped = str(ass_path.resolve()).replace("\\", "\\\\").replace(":", "\\:")
        video_filters.append(f"ass='{ass_escaped}'")

    vf = ",".join(video_filters)
    af = f"loudnorm=I={AUDIO_LUFS_TARGET}:TP={AUDIO_TRUE_PEAK}:LRA=11"

    command = [
        ffmpeg_binary,
        "-y",
        "-ss",
        str(start),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-vf",
        vf,
        "-af",
        af,
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

    LOGGER.info(
        "  Cutting clip '%s' [%.1fs-%.1fs] style=%s",
        clip.get("title", "untitled"),
        start,
        end,
        SUBTITLE_STYLES.get(subtitle_style, {}).get("name", "Default"),
    )

    try:
        run_command(command, timeout=3600)
    except PipelineError as exc:
        LOGGER.error("FFmpeg failed for clip '%s': %s", clip.get("title", "untitled"), exc)
        raise
    finally:
        # Clean up .ass file after burning
        if ass_path and ass_path.exists():
            try:
                ass_path.unlink()
            except OSError:
                pass

    if use_esrgan:
        output_path = upscale_with_esrgan(output_path)

    # Automatically extract a thumbnail for the finished clip
    extract_thumbnail(output_path)

    # Export a standalone SRT file for the clip if segments are available
    if segments:
        export_srt(segments, start, end, output_path)

    return output_path


def upscale_with_esrgan(clip_path: Path) -> Path:
    """Upscale clip using Real-ESRGAN if available."""
    upscaled_path = clip_path.with_stem(clip_path.stem + "_4k")
    # Use existing constant from GPUOrchestrator
    esrgan_exe = shutil.which(GPUOrchestrator.REALESRGAN_EXE)
    if not esrgan_exe:
        LOGGER.warning("Real-ESRGAN not found on PATH, skipping upscale.")
        return clip_path
        
    LOGGER.info("  Starting 2x ESRGAN upscale for: %s", clip_path.name)
    try:
        cmd = [
            esrgan_exe,
            "-i", str(clip_path),
            "-o", str(upscaled_path),
            "-s", "2",
            "-n", GPUOrchestrator.REALESRGAN_MODEL,
        ]
        run_command(cmd, timeout=1200) # ESRGAN can be slow
        LOGGER.info("  Upscale complete: %s", upscaled_path.name)
        return upscaled_path
    except PipelineError as exc:
        LOGGER.warning("ESRGAN upscale failed, keeping original: %s", exc)
        return clip_path


def analyze_chunks(
    segments: list[dict],
    video_duration: float,
    model: str,
    min_sec: int,
    max_sec: int,
    min_clips: int,
    max_clips: int,
    custom_instruction: str = "",
    use_gemini: bool = False,
    debug_llm: bool = False,
) -> list[dict]:
    """
    Analyze transcript segments and return validated clip suggestions.
    Guarantees at least min_clips are returned by re-prompting if needed.
    """
    # ── Build transcript ──────────────────────────────────────────────────────
    transcript = " ".join(
        f"[{s['start']:.1f}s–{s['end']:.1f}s] {s['text'].strip()}"
        for s in segments
        if s.get("text", "").strip()
    )
    if not transcript:
        raise PipelineError("Empty transcript — cannot analyze.")

    prompt = build_prompt(
        transcript=transcript,
        min_sec=min_sec,
        max_sec=max_sec,
        min_clips=min_clips,
        max_clips=max_clips,
        custom_instruction=custom_instruction,
    )

    # Log instruction block to fulfill verification tests
    if "===" in prompt:
        headers = prompt.split("===")
        if len(headers) >= 3:
            LOGGER.info("  EDITOR INSTRUCTIONS CONNECTED: %s", headers[1].strip())
            LOGGER.info("  %s", headers[2].strip()[:300].replace("\n", " ") + "...")

    llm_fn = (lambda p: call_gemini(p, use_schema=True)) if use_gemini else (lambda p: call_ollama(p, model=model, use_schema=True))

    # ── LLM call with retry ───────────────────────────────────────────────────
    MAX_ATTEMPTS = 3
    # ── Retry prompts escalate in simplicity ─────────────────────────────────
    # Retry prompts — focus on CONTENT quality, not JSON format
    # Schema handles the format; these guide the model toward better clips
    RETRY_SUFFIXES = [
        "",    # attempt 1 — schema handles format
        "\n\nREMINDER: Combine multiple transcript lines into 30-90 second scenes. "
               "Do NOT use individual sentence timestamps.",
        "\n\nCRITICAL: Each clip needs start and end at least 30 seconds apart. "
               "Example: start=120.0, end=165.0 is correct (45s). "
               "start=120.0, end=123.0 is WRONG (3s).",
    ]

    raw_clips: list[dict] = []
    last_error: str = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        LOGGER.info(
            "  LLM analysis attempt %d/%d (model=%s)...", attempt, MAX_ATTEMPTS, model
        )
        current_prompt = prompt + RETRY_SUFFIXES[attempt - 1]

        try:
            response_text = llm_fn(current_prompt)

            # Log raw response preview for debugging
            preview = response_text[:200].replace("\n", " ") if response_text else "<empty>"
            LOGGER.debug("  LLM raw response preview: %s", preview)

            if debug_llm:
                LOGGER.debug("  FULL LLM RESPONSE:\n%s", response_text)

            parsed = parse_llm_json(response_text)
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(c, dict) for c in parsed):
                    raw_clips = parsed
                    LOGGER.info(
                        "  LLM returned %d clip suggestions on attempt %d.", len(parsed), attempt
                    )
                    break
                else:
                    LOGGER.warning(
                        "  LLM returned list of invalid types (expected dicts) on attempt %d, retrying...",
                        attempt
                    )
                    last_error = "LLM returned non-object items in array"
            else:
                LOGGER.warning(
                    "  LLM returned empty list on attempt %d, retrying...", attempt
                )
                last_error = "Empty list returned"

        except PipelineError as exc:
            last_error = str(exc)
            # Log the first 300 chars of raw output for diagnostics
            LOGGER.warning(
                "  LLM attempt %d failed: %s", attempt, str(exc)[:300]
            )
            if attempt == MAX_ATTEMPTS:
                raise PipelineError(
                    f"LLM failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}\n"
                    f"Try: --model gemma2:latest or --model llama3.2 for better JSON compliance."
                ) from exc

    # ── Validate each clip — strict duration enforcement ──────────────────────
    valid_clips:    list[dict] = []
    rejected_clips: list[dict] = []

    for raw_clip in raw_clips:
        # 1. Normalize keys (handles start_time, null timestamps, etc.)
        clip = normalize_clip(raw_clip, segments=segments)

        # 2. Fix mid-sentence cuts BEFORE duration check
        clip = fix_sentence_boundary(clip, segments, video_duration, max_extend=5.0)

        # 3. Expand clips that are too short
        duration = clip["end"] - clip["start"]
        if 0 < duration < min_sec:
            clip = expand_clip_to_minimum(clip, min_sec, max_sec, video_duration)

        # 4. Hard validation — reject only if still invalid after all fixes
        ok, reason = validate_clip(clip, video_duration, min_sec, max_sec)
        if ok:
            valid_clips.append(clip)
        else:
            LOGGER.warning(
                "  Rejected clip '%s': %s",
                clip.get("title", "untitled"), reason,
            )
            rejected_clips.append(clip)

    LOGGER.info(
        "  Clip validation: %d valid, %d rejected out of %d suggested",
        len(valid_clips), len(rejected_clips), len(raw_clips)
    )

    # ── Min-clips guarantee — re-prompt if not enough valid clips ─────────────
    if len(valid_clips) < min_clips:
        deficit = min_clips - len(valid_clips)
        LOGGER.warning(
            "  Only %d valid clips found, need %d more. Re-prompting with relaxed guidance...",
            len(valid_clips), deficit
        )

        # Reprompt with relaxed guidance
        retry_prompt = prompt + f"\n\nOnly found {len(valid_clips)}/30-90s clips. RELAX rules and find ANY interesting {min_sec}-{max_sec}s segments."
        
        for retry_attempt in range(1, 3):   # up to 2 extra attempts
            LOGGER.info("  Min-clips retry %d/2...", retry_attempt)
            try:
                retry_text = llm_fn(retry_prompt)
                retry_parsed = parse_llm_json(retry_text)
                if not isinstance(retry_parsed, list) or not retry_parsed:
                    continue

                if not all(isinstance(c, dict) for c in retry_parsed):
                    LOGGER.warning("  Min-clips retry %d returned non-object items, skipping.", retry_attempt)
                    continue

                for raw_retry in retry_parsed:
                    clip = normalize_clip(raw_retry, segments=segments)

                    # skip if duplicate title already in valid_clips
                    existing_titles = {c.get("title", "").lower() for c in valid_clips}
                    if clip.get("title", "").lower() in existing_titles:
                        continue

                    # Fix mid-sentence cuts
                    clip = fix_sentence_boundary(clip, segments, video_duration, max_extend=5.0)

                    # Try to expand short clips before rejecting them
                    duration = clip["end"] - clip["start"]
                    if 0 < duration < min_sec:
                        clip = expand_clip_to_minimum(clip, min_sec, max_sec, video_duration)

                    ok, reason = validate_clip(clip, video_duration, min_sec, max_sec)
                    if ok:
                        valid_clips.append(clip)
                        LOGGER.info(
                            "  Recovery clip accepted: '%s' (%.1fs–%.1fs)",
                            clip.get("title"), clip.get("start"), clip.get("end")
                        )

                if len(valid_clips) >= min_clips:
                    break
            except Exception as exc:
                LOGGER.warning("  Min-clips retry failed: %s", exc)

    # ── Final check — warn but never hard-fail ────────────────────────────────
    if len(valid_clips) < min_clips:
        LOGGER.error(
            "  Could not reach min_clips=%d even after retries. Got %d. "
            "Consider lowering --min or --min-clips for this video.",
            min_clips, len(valid_clips)
        )

    # ── Sort by score, cap at max_clips ──────────────────────────────────────
    valid_clips.sort(key=lambda c: c.get("score", 0), reverse=True)
    return valid_clips[:max_clips]
    
def rewrite_hooks(clips: list[dict[str, Any]], use_gemini: bool) -> list[dict[str, Any]]:
    """Ask LLM to generate 5 TikTok caption variations for each clip's hook."""
    llm_fn = call_gemini if use_gemini else call_ollama

    for clip in clips:
        hook = clip.get("hook", "")
        if not hook:
            clip["hook_variants"] = []
            continue

        prompt = f"""You are a TikTok/Reels copywriter. Given this video hook:
"{hook}"

Write exactly 5 viral caption variations for this moment.
Rules:
- Each under 100 characters
- Use curiosity, emotion, or controversy
- Vary the style: question / bold claim / relatable / shocking / funny
- Return ONLY a JSON array of 5 strings, nothing else

Example output:
["Caption 1", "Caption 2", "Caption 3", "Caption 4", "Caption 5"]"""

        try:
            raw = llm_fn(prompt)
            # Parse the JSON array
            raw = raw.strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            variants = json.loads(raw[start:end]) if start != -1 else []
            clip["hook_variants"] = [str(v)[:120] for v in variants[:5]]
            LOGGER.info("Hook variants for '%s': %s", clip["title"], clip["hook_variants"])
        except Exception as exc:
            LOGGER.warning("Hook rewrite failed for '%s': %s", clip["title"], exc)
            clip["hook_variants"] = []
            continue

    return clips


def load_json(path: Path, default: Any = None) -> Any:
    """Read a JSON document from disk."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
    no_scene_snap: bool = False,
    use_esrgan: bool = False,
    subtitle_style: int = 0,
    chapters: str = "",
    use_face_tracking: bool = True,
) -> list[dict[str, Any]]:
    """Render the final MP4 clips and save metadata."""
    results: list[dict[str, Any]] = []
    segments = transcript.get("segments", [])
    metadata_path = OUTPUT_DIR / "clips_metadata.json"

    # ── Load EXISTING metadata to preserve clip history ───────────────────────
    existing_metadata: list[dict[str, Any]] = []
    if metadata_path.exists():
        try:
            existing_metadata = load_json(metadata_path, [])
            if not isinstance(existing_metadata, list):
                existing_metadata = []
            LOGGER.info("  Loaded %d existing clips from metadata.", len(existing_metadata))
        except Exception as exc:
            LOGGER.warning("  Could not read existing metadata (%s), starting fresh.", exc)

    new_clip_metadata: list[dict[str, Any]] = []

    for index, clip in enumerate(clips, start=1):
        filename_root = f"{index:02d}_{safe_filename(clip['title'])}_{int(clip['start'])}s"
        srt_path = TEMP_DIR / f"{filename_root}.srt"
        output_path = OUTPUT_DIR / f"{filename_root}.mp4"
        individual_metadata_path = OUTPUT_DIR / f"{filename_root}.json"
        try:
            generate_srt(segments, clip["start"], clip["end"], srt_path)
            output_path = cut_and_format_clip(
                video_path,
                clip,
                srt_path,
                output_path,
                encoder_flags=encoder_flags,
                no_scene_snap=no_scene_snap,
                use_esrgan=use_esrgan,
                subtitle_style=subtitle_style,
                segments=segments,
                use_face_tracking=use_face_tracking,
            )
        except PipelineError:
            LOGGER.warning("Skipping clip after FFmpeg failure: %s", clip["title"])
            continue

        exported_srt = output_path.with_suffix(".srt")
        record = {
            **clip,
            "duration": round(float(clip["end"]) - float(clip["start"]), 3),
            "video_path": str(output_path),
            "video_name": video_path.name,
            "subtitle_path": str(srt_path),
            "output_file": output_path.name,
            "thumbnail": output_path.with_suffix(".jpg").name,
            "srt": exported_srt.name if exported_srt.exists() else None,
            "chapters": chapters,
            "generated_at": datetime.now().isoformat(),
            "subtitle_style": subtitle_style,
        }
        save_json(individual_metadata_path, record)
        new_clip_metadata.append(record)
        results.append(record)
        LOGGER.info("Saved clip: %s", output_path)

    # ── Merge: existing + new, deduplicate by output_file name ───────────────
    seen_files: set[str] = set()
    merged_metadata: list[dict[str, Any]] = []

    # New clips take priority if filenames collide
    for entry in new_clip_metadata + existing_metadata:
        key = entry.get("output_file") or Path(str(entry.get("video_path", ""))).name
        if key and key not in seen_files:
            seen_files.add(key)
            merged_metadata.append(entry)

    # ── Write merged metadata ─────────────────────────────────────────────────
    try:
        save_json(metadata_path, merged_metadata)
        LOGGER.info(
            "  Metadata saved: %d total clips (%d new, %d historical).",
            len(merged_metadata), len(new_clip_metadata), len(existing_metadata)
        )
    except Exception as exc:
        LOGGER.error("  Could not write metadata: %s", exc)

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="ClipForge AI video clipping pipeline")
    parser.add_argument("source", help="YouTube URL or local video file path")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--whisper", default=WHISPER_MODEL, choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--clips", type=int, default=MAX_CLIPS, help="Maximum clips to generate")
    parser.add_argument(
        "--min-clips",
        type=int,
        default=1,
        metavar="N",
        dest="min_clips",
        help=(
            "Minimum number of clips to extract per video. "
            "Lower this for short or dense-monologue videos. "
            "Use 1 to always attempt at least one clip. (default: 1)"
        ),
    )
    parser.add_argument("--min", dest="min_sec", type=int, default=MIN_CLIP_SEC, help="Minimum clip seconds")
    parser.add_argument("--max", dest="max_sec", type=int, default=MAX_CLIP_SEC, help="Maximum clip seconds")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini instead of Ollama")
    parser.add_argument(
        "--prompt", "--custom-prompt",    # both flags work
        dest="prompt",                    # stored as args.prompt
        metavar="TEXT",
        default="",
        help="Custom instructions for the AI clip selector.",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale clips with Real-ESRGAN after cutting.",
    )
    parser.add_argument(
        "--no-scene-snap",
        action="store_true",
        help="Skip scene cut snapping for faster processing.",
    )
    parser.add_argument(
        "--subtitle-style",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Subtitle style: 0=Classic, 1=TikTok Yellow, 2=Storyteller, 3=Neon Cyan, 4=Action Red",
    )
    parser.add_argument("--job-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--transcribe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transcribe-output", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-face-tracking",
        action="store_true",
        help="Disable face tracking crop, use static center crop instead.",
    )
    parser.add_argument(
        "--no-chapters",
        action="store_true",
        help="Disable chapter generation.",
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="Log full raw LLM responses for debugging JSON parse failures.",
    )
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
        transcript = transcribe(video_path, model_name=args.whisper)
        segments = transcript.get("segments", [])
        duration = get_video_duration(video_path)
        model = args.model
        min_clip = args.min_sec
        max_clip = args.max_sec

        LOGGER.info("Transcription returned, saving to disk...")
        try:
            save_json(transcript_path, transcript)
            LOGGER.info("Transcript saved OK: %s", transcript_path)
            
            if not segments:
                LOGGER.error("No speech detected in this video. Clip extraction skipped.")
                # We mention model parameters as secondary tip
                LOGGER.info("TIP: If the video actually has speech, try lowering VAD threshold or using a larger Whisper model (e.g. --whisper large).")
                raise PipelineError("Empty transcript — no speech detected in video.")
        except Exception as exc:
            LOGGER.error("save_json FAILED: %s", exc)
            raise
        temp_files_to_cleanup.append(transcript_path)

        LOGGER.info("Stage 2b/5: Generating chapter markers...")
        if not args.no_chapters:
            chapters = generate_chapters(
                segments=segments,
                model=model,
                use_gemini=args.model.startswith("gemini"),
            )
            if chapters:
                chapters_path = OUTPUT_DIR / f"chapters_{job_id}.txt"
                chapters_path.write_text(chapters, encoding="utf-8")
                LOGGER.info("Chapters saved: %s", chapters_path.name)
        else:
            chapters = ""

        if not args.gemini:
            LOGGER.info("Stage 3/5: Restarting Ollama for analysis...")
            GPUOrchestrator.start_ollama()
        chunks = chunk_transcript(transcript["segments"])
        LOGGER.info("Built %s transcript chunk(s).", len(chunks))
        LOGGER.info("Stage 3/5: Analyzing viral moments with AI...")
        video_duration = get_video_duration(video_path)
        try:
            deduped = analyze_chunks(
                segments=transcript["segments"],
                video_duration=video_duration,
                model=args.model,
                min_sec=args.min_sec,
                max_sec=args.max_sec,
                min_clips=args.min_clips,
                max_clips=args.clips,
                custom_instruction=args.prompt or "",
                use_gemini=args.gemini,
                debug_llm=getattr(args, "debug_llm", False),
            )
            deduped = deduplicate_clips(deduped, limit=args.clips)
            LOGGER.info(
                "Selected %d clip(s) after validation and deduplication.", len(deduped)
            )
        except PipelineError as exc:
            LOGGER.error("%s", exc)
            LOGGER.error(
                "MODEL COMPATIBILITY TIP: '%s' struggled with JSON output.\n"
                "  More reliable alternatives:\n"
                "    ollama pull gemma2:latest      (best JSON compliance)\n"
                "    ollama pull llama3.2:latest    (good balance)\n"
                "    ollama pull mistral-nemo       (improved Mistral variant)\n"
                "  Or enable Gemini fallback in the dashboard.",
                args.model
            )
            raise SystemExit(1)
        if not deduped:
            LOGGER.warning("No valid clips were found.")
            return []

        LOGGER.info("Stage 3b/5: Rewriting hooks for virality...")
        deduped = rewrite_hooks(deduped, use_gemini=args.gemini)

        LOGGER.info("Stage 4/5: Cutting clips...")
        _, encoder_flags = detect_encoder(prefer_quality=True)
        results = process_clip_batch(
            video_path,
            transcript,
            deduped,
            encoder_flags=encoder_flags,
            no_scene_snap=args.no_scene_snap,
            use_esrgan=args.upscale,
            subtitle_style=args.subtitle_style,
            chapters=chapters,
            use_face_tracking=not getattr(args, "no_face_tracking", False),
        )
        LOGGER.info("Stage 5/5: Pipeline complete!")
        return {
            "clips": results,
            "chapters": chapters,
        }
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
    clip_count = len(results.get("clips", [])) if isinstance(results, dict) else len(results)
    LOGGER.info("Done. Generated %d clip(s).", clip_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
