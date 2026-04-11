from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import platform
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pipeline
import setup_check
import fix_quality

ROOT_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = ROOT_DIR / "dashboard.html"
HISTORY_FILE = pipeline.TEMP_DIR / "dashboard_history.json"
MAX_LOG_LINES = 220
MAX_HISTORY_ITEMS = 12
SETUP_CACHE_TTL = 15.0
MODELS_CACHE_TTL = 20.0
PAYLOAD_CACHE_TTL = 3.0
SETUP_CACHE: dict[str, Any] = {"timestamp": 0.0, "payload": None}
MODELS_CACHE: dict[str, Any] = {"timestamp": 0.0, "payload": []}
PAYLOAD_CACHE: dict[str, Any] = {"timestamp": 0.0, "payload": None}
ENHANCE_JOBS: dict[str, dict[str, Any]] = {}
ENHANCE_LOCK = threading.Lock()
ENHANCED_DIR = ROOT_DIR / "outputs" / "enhanced"
REALESRGAN_EXE = "realesrgan-ncnn-vulkan"


def now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    """Load a JSON file or return a default value."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, payload: Any) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a workspace-relative path safely."""
    target = (ROOT_DIR / relative_path).resolve()
    target.relative_to(ROOT_DIR)
    return target


def safe_output_clip(filename: str) -> Path:
    """Resolve a clip filename inside outputs/clips safely."""
    candidate = Path(filename).name
    path = (pipeline.OUTPUT_DIR / candidate).resolve()
    path.relative_to((ROOT_DIR / pipeline.OUTPUT_DIR).resolve())
    return path


def find_ollama_binary() -> str | None:
    """Locate the Ollama executable."""
    discovered = shutil.which("ollama")
    if discovered:
        return discovered
    if sys.platform.startswith("win"):
        candidate = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
        if candidate.exists():
            return str(candidate)
    return None


def list_ollama_models() -> list[str]:
    """Return installed Ollama model names when available."""
    binary = find_ollama_binary()
    if not binary:
        return []
    try:
        result = subprocess.run(
            [binary, "list"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        models.append(stripped.split()[0])
    return models


def get_cached_setup(force: bool = False) -> dict[str, Any]:
    """Return setup checks with a short-lived cache."""
    now = time.monotonic()
    if (
        not force
        and SETUP_CACHE["payload"] is not None
        and now - float(SETUP_CACHE["timestamp"]) < SETUP_CACHE_TTL
    ):
        return SETUP_CACHE["payload"]
    checks = setup_check.collect_checks()
    payload = {"checks": checks, "all_ok": all(bool(check["ok"]) for check in checks)}
    SETUP_CACHE.update({"timestamp": now, "payload": payload})
    return payload


def get_cached_models(force: bool = False) -> list[str]:
    """Return installed model names with a short-lived cache."""
    now = time.monotonic()
    if (
        not force
        and MODELS_CACHE["payload"]
        and now - float(MODELS_CACHE["timestamp"]) < MODELS_CACHE_TTL
    ):
        return MODELS_CACHE["payload"]
    models = list_ollama_models()
    MODELS_CACHE.update({"timestamp": now, "payload": models})
    return models


def gpu_diagnostics() -> dict[str, Any]:
    """Collect GPU and encoder diagnostics for the dashboard."""
    results: dict[str, Any] = {}

    try:
        import torch  # type: ignore

        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        results["cuda_version"] = torch.version.cuda
        results["torch_version"] = torch.__version__
    except ImportError:
        results["cuda_available"] = False
        results["cuda_device"] = None
        results["cuda_version"] = None
        results["torch_version"] = "torch not installed"

    results["nvenc_available"] = pipeline._test_encoder("h264_nvenc")
    results["qsv_available"] = pipeline._test_encoder("h264_qsv")
    results["amf_available"] = pipeline._test_encoder("h264_amf")

    enhancer_path = shutil.which(REALESRGAN_EXE)
    local_enhancer = ROOT_DIR / f"{REALESRGAN_EXE}.exe"
    results["realesrgan_found"] = bool(enhancer_path or local_enhancer.exists())

    try:
        from urllib import request

        request.urlopen("http://localhost:11434", timeout=2)
        results["ollama_running"] = True
    except Exception:
        results["ollama_running"] = False

    ffmpeg_binary = pipeline.ensure_ffmpeg_on_path()
    ffmpeg_result = subprocess.run(
        [ffmpeg_binary, "-version"], capture_output=True, text=True
    )
    results["ffmpeg_version"] = (
        ffmpeg_result.stdout.splitlines()[0]
        if ffmpeg_result.returncode == 0
        else "not found"
    )

    active_encoder, _ = pipeline.detect_encoder(
        prefer_quality=pipeline.ENCODER_PREFER_QUALITY
    )
    results["active_encoder"] = active_encoder

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if smi.returncode == 0:
        parts = smi.stdout.strip().split(", ")
        results["nvidia_gpu"] = parts[0] if len(parts) > 0 else None
        results["nvidia_driver"] = parts[1] if len(parts) > 1 else None
        results["nvidia_vram"] = parts[2] if len(parts) > 2 else None
    else:
        results["nvidia_gpu"] = None
        results["nvidia_driver"] = "nvidia-smi not found or no GPU"
        results["nvidia_vram"] = None

    return results


def to_relative_url(path: Path) -> str:
    """Convert a workspace path into a served file URL."""
    relative = path.resolve().relative_to(ROOT_DIR).as_posix()
    return f"/files/{relative}"


def load_clips() -> list[dict[str, Any]]:
    """Load clip metadata enriched for the dashboard, rebuilding from sidecars if needed."""
    metadata_path = pipeline.OUTPUT_DIR / "clips_metadata.json"
    records = load_json(metadata_path, [])

    # ── Rebuild from .json sidecars if metadata is missing or empty ──────────
    if not records and pipeline.OUTPUT_DIR.exists():
        records = []
        seen_files = set()
        for json_file in sorted(pipeline.OUTPUT_DIR.glob("*.json")):
            if json_file.name == "clips_metadata.json":
                continue
            try:
                entry = load_json(json_file, {})
                mp4_name = json_file.stem + ".mp4"
                if mp4_name in seen_files:
                    continue
                seen_files.add(mp4_name)
                # Normalize into standard metadata shape
                records.append(
                    {
                        "video_name": entry.get("video_name", "unknown"),
                        "title": entry.get("title", json_file.stem),
                        "start": entry.get("start", 0),
                        "end": entry.get("end", 0),
                        "score": entry.get("score", 0),
                        "reason": entry.get("reason", ""),
                        "output_file": mp4_name,
                        "video_path": str((pipeline.OUTPUT_DIR / mp4_name).resolve()),
                        "subtitle_path": str(
                            (pipeline.TEMP_DIR / (json_file.stem + ".srt")).resolve()
                        ),
                        "generated_at": entry.get("generated_at", ""),
                        "subtitle_style": entry.get("subtitle_style", 0),
                    }
                )
            except Exception:
                continue
        if records:
            save_json(metadata_path, records)

    if not isinstance(records, list):
        records = []

    clips: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        # Determine path to video
        v_name = record.get("output_file") or record.get("video_name") or ""
        if v_name and not v_name.endswith(".mp4") and "." not in v_name:
            v_name += ".mp4"

        video_path = pipeline.OUTPUT_DIR / v_name
        if not video_path.exists():
            # Try absolute path from record
            raw_video_path = Path(str(record.get("video_path", "")))
            video_path = (
                raw_video_path
                if raw_video_path.is_absolute()
                else (ROOT_DIR / raw_video_path)
            )

        if not video_path.exists():
            continue

        raw_subtitle_path = Path(str(record.get("subtitle_path", "")))
        subtitle_path = (
            raw_subtitle_path
            if raw_subtitle_path.is_absolute()
            else (ROOT_DIR / raw_subtitle_path)
        )
        json_path = pipeline.OUTPUT_DIR / f"{video_path.stem}.json"

        clips.append(
            {
                "title": record.get("title", video_path.stem),
                "hook": record.get("hook", ""),
                "reason": record.get("reason", ""),
                "start": record.get("start", 0),
                "end": record.get("end", 0),
                "duration": record.get("duration", 0)
                or round(
                    float(record.get("end", 0)) - float(record.get("start", 0)), 2
                ),
                "video_name": video_path.name,
                "video_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
                "modified_at": record.get("generated_at")
                or datetime.fromtimestamp(
                    video_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "video_url": to_relative_url(video_path),
                "json_url": to_relative_url(json_path) if json_path.exists() else "",
                "subtitle_url": to_relative_url(subtitle_path)
                if subtitle_path.exists()
                else "",
                "thumbnail_url": to_relative_url(video_path.with_suffix(".jpg"))
                if video_path.with_suffix(".jpg").exists()
                else "",
                "video_path": str(video_path.resolve()),
                "video_relative": str(
                    video_path.resolve().relative_to(ROOT_DIR)
                ).replace("\\", "/"),
                "json_path": str(json_path.resolve()) if json_path.exists() else "",
            }
        )

    clips.sort(key=lambda item: item["modified_at"], reverse=True)
    return clips


def update_enhance_job(job_id: str, **changes: Any) -> None:
    """Update an enhancement job entry in a threadsafe way."""
    with ENHANCE_LOCK:
        if job_id not in ENHANCE_JOBS:
            ENHANCE_JOBS[job_id] = {"enhance_job_id": job_id}
        ENHANCE_JOBS[job_id].update(changes)


def get_enhance_job(job_id: str) -> dict[str, Any]:
    """Return a snapshot of an enhancement job."""
    with ENHANCE_LOCK:
        job = ENHANCE_JOBS.get(job_id, {})
        return dict(job)


def extract_src_fps(src: Path) -> float:
    """Read source FPS for AI frame reassembly."""
    info = fix_quality.get_source_info(src)
    fps = float(info.get("fps") or 30.0)
    return fps if fps > 0 else 30.0


def remove_tree(path: Path) -> None:
    """Delete a temporary directory tree if it exists."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)


def _enhance_fast(src: Path, out: Path, job_id: str) -> None:
    """Enhance a clip using FFmpeg-only filters."""
    update_enhance_job(job_id, progress=10, message="Applying filters...")
    ffmpeg_binary = pipeline.ensure_ffmpeg_on_path()
    _, encoder_flags = pipeline.detect_encoder(
        prefer_quality=pipeline.ENCODER_PREFER_QUALITY
    )
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(src),
        "-vf",
        "hqdn3d=3:3:6:6,"
        "unsharp=5:5:1.2:5:5:0.0,"
        "eq=contrast=1.08:brightness=0.02:saturation=1.15,"
        "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1",
        *encoder_flags,
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(out),
    ]
    update_enhance_job(job_id, progress=90, message="Encoding...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "FFmpeg enhancement failed")[-500:]
        )


def _enhance_ai(src: Path, out: Path, scale: int, job_id: str) -> None:
    """Enhance a clip using Real-ESRGAN frame upscaling."""
    enhancer_binary = shutil.which(REALESRGAN_EXE)
    if not enhancer_binary:
        candidate = ROOT_DIR / f"{REALESRGAN_EXE}.exe"
        enhancer_binary = str(candidate) if candidate.exists() else None
    if not enhancer_binary:
        raise RuntimeError(
            "Real-ESRGAN not installed. Download from: github.com/xinntao/Real-ESRGAN/releases "
            "Extract realesrgan-ncnn-vulkan.exe into project folder."
        )

    ffmpeg_binary = pipeline.ensure_ffmpeg_on_path()
    frames_in = pipeline.TEMP_DIR / f"frames_in_{job_id}"
    frames_out = pipeline.TEMP_DIR / f"frames_out_{job_id}"
    remove_tree(frames_in)
    remove_tree(frames_out)
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        update_enhance_job(job_id, progress=10, message="Extracting frames...")
        extract_command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(src),
            "-q:v",
            "1",
            "-vsync",
            "0",
            str(frames_in / "%06d.png"),
        ]
        result = subprocess.run(extract_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "Frame extraction failed")[-500:]
            )

        frame_count = len(list(frames_in.glob("*.png")))
        update_enhance_job(
            job_id,
            progress=30,
            message=f"AI upscaling {frame_count} frames (may take 5-30 min)...",
        )

        upscale_command = [
            enhancer_binary,
            "-i",
            str(frames_in),
            "-o",
            str(frames_out),
            "-n",
            "realesrgan-x4plus",
            "-s",
            str(scale),
            "-f",
            "png",
            "-g",
            pipeline.GPUOrchestrator.REALESRGAN_GPU_ID,
        ]
        update_enhance_job(
            job_id, progress=35, message=f"Running Real-ESRGAN x{scale}..."
        )
        result = subprocess.run(upscale_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "Real-ESRGAN failed")[-500:]
            )

        fps = extract_src_fps(src)
        update_enhance_job(
            job_id, progress=85, message="Reassembling enhanced frames..."
        )
        reassemble_command = [
            ffmpeg_binary,
            "-y",
            "-framerate",
            f"{fps}",
            "-i",
            str(frames_out / "%06d.png"),
            "-i",
            str(src),
            "-vf",
            "scale=1080:1920:flags=lanczos,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            "setsar=1,"
            "unsharp=3:3:0.5:3:3:0.0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-b:v",
            "16000k",
            "-maxrate",
            "20000k",
            "-bufsize",
            "30000k",
            "-profile:v",
            "high",
            "-level",
            "4.2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-movflags",
            "+faststart",
            str(out),
        ]
        result = subprocess.run(reassemble_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "Frame reassembly failed")[-500:]
            )
    finally:
        remove_tree(frames_in)
        remove_tree(frames_out)


def run_enhance(filename: str, mode: str, scale: int, job_id: str) -> None:
    """Run one enhancement job in the background."""
    src = safe_output_clip(filename)
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "AI_" if mode == "ai" else "ENHANCED_"
    out_path = ENHANCED_DIR / f"{prefix}{Path(filename).stem}.mp4"
    update_enhance_job(
        job_id,
        enhance_job_id=job_id,
        filename=Path(filename).name,
        status="running",
        progress=0,
        output_filename=None,
        download_url=None,
        error=None,
        message="Queued...",
    )
    try:
        if mode == "ai":
            _enhance_ai(src, out_path, scale, job_id)
        else:
            _enhance_fast(src, out_path, job_id)

        update_enhance_job(
            job_id,
            status="done",
            progress=100,
            output_filename=out_path.name,
            download_url=f"/download/enhanced/{out_path.name}",
            message="Enhancement complete!",
        )
    except Exception as exc:
        update_enhance_job(job_id, status="error", error=str(exc), message=str(exc))


class JobManager:
    """Manage a single active pipeline job for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._job: dict[str, Any] | None = None
        self._queue: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = load_json(HISTORY_FILE, [])

    def _infer_stage(self, line: str) -> str:
        """Map a log line to a friendly pipeline stage."""
        lowered = line.lower()
        if "progress:download:" in lowered:
            return "Downloading"
        if "progress:transcribe:" in lowered:
            return "Transcribing"
        if "stage 1/5" in lowered:
            return "Downloading"
        if "stage 2/5" in lowered:
            return "Transcribing"
        if "stage 3/5" in lowered:
            return "Analyzing"
        if "stage 4/5" in lowered:
            return "Cutting Clips"
        if "stage 5/5" in lowered:
            return "Done"
        if "downloading source video" in lowered:
            return "Downloading"
        if "transcribing " in lowered or "detected language" in lowered:
            return "Transcribing"
        if "built " in lowered and "transcript chunk" in lowered:
            return "Analyzing"
        if "analyzing transcript chunk" in lowered:
            return "Analyzing"
        if "selected " in lowered and "clip" in lowered:
            return "Cutting Clips"
        if "saved clip:" in lowered:
            return "Cutting Clips"
        if "done. generated" in lowered:
            return "Done"
        if "[error]" in lowered:
            return "Needs Attention"
        return "Working"

    def _persist_history(self) -> None:
        """Persist recent history to disk."""
        save_json(HISTORY_FILE, self._history[:MAX_HISTORY_ITEMS])

    def _append_history(self, snapshot: dict[str, Any]) -> None:
        """Store a finished job in recent history."""
        entry = {
            "id": snapshot["id"],
            "source": snapshot["source"],
            "status": snapshot["status"],
            "started_at": snapshot["started_at"],
            "finished_at": snapshot.get("finished_at"),
            "clip_count": snapshot.get("clip_count", 0),
            "stage": snapshot.get("stage", "Idle"),
            "model": snapshot["config"].get("model", pipeline.OLLAMA_MODEL),
            "use_gemini": bool(snapshot["config"].get("gemini")),
        }
        self._history = [
            entry,
            *[item for item in self._history if item.get("id") != entry["id"]],
        ][:MAX_HISTORY_ITEMS]
        self._persist_history()

    def start_job(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Launch the pipeline subprocess or queue it if busy."""
        source = str(config.get("source", "")).strip()
        if not source:
            return False, "Add a video URL or local file path first."

        # Read 'prompt' consistently, supporting backward compatibility
        custom_prompt = config.get("prompt") or config.get("custom_prompt") or ""
        custom_prompt = custom_prompt.strip()

        with self._lock:
            # If already running, add to queue instead
            if self._process and self._process.poll() is None:
                self._queue.append(config)
                return True, f"Video added to queue ({len(self._queue)} pending)."

            job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            command = [
                sys.executable,
                "pipeline.py",
                source,
                "--job-id",
                job_id,
                "--model",
                str(config.get("model", pipeline.OLLAMA_MODEL)),
                "--whisper",
                str(config.get("whisper", pipeline.WHISPER_MODEL)),
                "--clips",
                str(config.get("clips", pipeline.MAX_CLIPS)),
                "--min",
                str(config.get("min_sec", pipeline.MIN_CLIP_SEC)),
                "--max",
                str(config.get("max_sec", pipeline.MAX_CLIP_SEC)),
            ]
            if bool(config.get("gemini")):
                command.append("--gemini")
            if custom_prompt:
                command.extend(["--prompt", custom_prompt])
            if bool(config.get("upscale")):
                command.append("--upscale")
            if bool(config.get("no_scene_snap")):
                command.append("--no-scene-snap")
            if config.get("subtitle_style") is not None:
                command.extend(["--subtitle-style", str(config.get("subtitle_style"))])
            if config.get("face_tracking") is False:
                command.append("--no-face-tracking")
            if config.get("min_clips") is not None:
                command.extend(["--min-clips", str(config.get("min_clips"))])

            self._job = {
                "id": job_id,
                "source": source,
                "status": "running",
                "stage": "Queued",
                "started_at": now_iso(),
                "finished_at": None,
                "clip_count": 0,
                "exit_code": None,
                "config": {
                    "model": str(config.get("model", pipeline.OLLAMA_MODEL)),
                    "whisper": str(config.get("whisper", pipeline.WHISPER_MODEL)),
                    "clips": int(config.get("clips", pipeline.MAX_CLIPS)),
                    "min_sec": int(config.get("min_sec", pipeline.MIN_CLIP_SEC)),
                    "max_sec": int(config.get("max_sec", pipeline.MAX_CLIP_SEC)),
                    "gemini": bool(config.get("gemini")),
                    "prompt": custom_prompt,
                    "upscale": bool(config.get("upscale")),
                    "no_scene_snap": bool(config.get("no_scene_snap")),
                    "subtitle_style": int(config.get("subtitle_style", 0)),
                    "face_tracking": bool(config.get("face_tracking", True)),
                },
                "baseline_videos": [clip["video_name"] for clip in load_clips()],
                "logs": [],
            }
            self._process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(
                target=self._watch_process, args=(self._process, job_id), daemon=True
            ).start()

        return True, "ClipForge run started."

    def _watch_process(self, process: subprocess.Popen[str], job_id: str) -> None:
        """Capture live logs, filter noise, and finalize job state."""
        assert process.stdout is not None

        ffmpeg_noise = re.compile(
            r"(frame=\s*\d+|fps=|bitrate=|speed=|size=\s*\d+kB"
            r"|time=\d+:\d+:\d+|dup=|drop=|Output #|Input #"
            r"|Stream mapping:|Press \[q\]|encoder\s*:|"
            r"video:\d+|audio:\d+|subtitle:\d+|global headers)"
        )

        for raw_line in process.stdout:
            line = raw_line.rstrip()

            # Filter FFmpeg noise
            stripped = line.strip()
            if not stripped:
                continue
            if ffmpeg_noise.search(stripped):
                continue

            # Junk threshold for long lines without log markers
            markers = (
                "[INFO]",
                "[WARNING]",
                "[ERROR]",
                "Stage",
                "Done.",
                "Saved clip",
                "Face tracking",
            )
            if len(stripped) > 200 and not any(m in stripped for m in markers):
                continue

            print(line, flush=True)
            with self._lock:
                if not self._job or self._job.get("id") != job_id:
                    continue

                # Tag for frontend
                if "[ERROR]" in line:
                    tagged = f"[ERROR] {line}"
                elif "[WARNING]" in line:
                    tagged = f"[WARNING] {line}"
                else:
                    tagged = line

                logs = deque(self._job.get("logs", []), maxlen=MAX_LOG_LINES)
                logs.append(tagged)
                self._job["logs"] = list(logs)
                inferred_stage = self._infer_stage(line)
                if inferred_stage != "Working":
                    self._job["stage"] = inferred_stage

        exit_code = process.wait()
        clips = load_clips()
        with self._lock:
            if not self._job or self._job.get("id") != job_id:
                return
            baseline_videos = set(self._job.get("baseline_videos", []))
            new_clips = [
                clip for clip in clips if clip["video_name"] not in baseline_videos
            ]
            self._job["exit_code"] = exit_code
            self._job["finished_at"] = now_iso()
            self._job["clip_count"] = len(new_clips)
            self._job["stage"] = "Completed" if exit_code == 0 else "Failed"
            self._job["status"] = "succeeded" if exit_code == 0 else "failed"
            self._job["results"] = (new_clips or clips)[:6]
            snapshot = dict(self._job)
            self._process = None

        self._append_history(snapshot)
        self._trigger_next_job()

    def _trigger_next_job(self) -> None:
        """Start the next job in the queue if one exists."""
        with self._lock:
            if self._process and self._process.poll() is None:
                return
            if not self._queue:
                return
            config = self._queue.pop(0)

        # Re-call start_job with the next config
        # Note: start_job already uses the lock internally
        self.start_job(config)

    def snapshot(self) -> dict[str, Any]:
        """Return a safe snapshot of the current job."""
        with self._lock:
            if not self._job:
                return {
                    "active": False,
                    "status": "idle",
                    "stage": "Idle",
                    "logs": [],
                    "clip_count": len(load_clips()),
                    "history": self._history,
                }
            snapshot = dict(self._job)
            snapshot["active"] = snapshot["status"] == "running"
            snapshot["history"] = self._history
            snapshot["queue_length"] = len(self._queue)
            return snapshot

    def start_ollama(self) -> tuple[bool, str]:
        """Launch Ollama serve when it is not already listening."""
        if setup_check.check_ollama()[0]:
            return True, "Ollama is already running."
        binary = find_ollama_binary()
        if not binary:
            return False, "Ollama was not found on this machine."

        kwargs: dict[str, Any] = {
            "cwd": str(ROOT_DIR),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([binary, "serve"], **kwargs)
        return True, "Ollama is starting in the background."


JOB_MANAGER = JobManager()


def dashboard_payload() -> dict[str, Any]:
    """Build the full dashboard state payload."""
    now = time.monotonic()
    cached_payload = PAYLOAD_CACHE["payload"]
    if (
        cached_payload is not None
        and now - float(PAYLOAD_CACHE["timestamp"]) < PAYLOAD_CACHE_TTL
    ):
        return cached_payload

    clips = load_clips()
    encoder_name, _ = pipeline.detect_encoder(
        prefer_quality=pipeline.ENCODER_PREFER_QUALITY
    )
    speed_encoder_name, _ = pipeline.detect_encoder(prefer_quality=False)
    payload = {
        "workspace": str(ROOT_DIR),
        "setup": get_cached_setup(),
        "job": JOB_MANAGER.snapshot(),
        "clips": clips,
        "models": get_cached_models(),
        "defaults": {
            "model": pipeline.OLLAMA_MODEL,
            "whisper": pipeline.WHISPER_MODEL,
            "clips": pipeline.MAX_CLIPS,
            "min_sec": pipeline.MIN_CLIP_SEC,
            "max_sec": pipeline.MAX_CLIP_SEC,
        },
        "folders": {
            "outputs": str((ROOT_DIR / pipeline.OUTPUT_DIR).resolve()),
            "temp": str((ROOT_DIR / pipeline.TEMP_DIR).resolve()),
            "log": str((ROOT_DIR / pipeline.LOG_FILE).resolve()),
        },
        "system": {
            "encoder": encoder_name,
            "gpu_ready": encoder_name != "libx264",
            "cuda_ready": pipeline._test_encoder("h264_nvenc"),
            "prefer_quality": pipeline.ENCODER_PREFER_QUALITY,
            "speed_encoder": speed_encoder_name,
        },
    }
    PAYLOAD_CACHE.update({"timestamp": now, "payload": payload})
    return payload


class SilentThreadingHTTPServer(ThreadingHTTPServer):
    """Suppress benign Windows client disconnect tracebacks."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Ignore harmless socket disconnect errors and surface real ones."""
        exc_type = sys.exc_info()[0]
        if exc_type in (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


class ClipForgeHandler(BaseHTTPRequestHandler):
    """Serve the dashboard UI and local control API."""

    server_version = "ClipForge/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default HTTP request logging."""
        return

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        """Send a JSON response."""
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return

    def _send_text_file(self, path: Path, content_type: str) -> None:
        """Send a text file response."""
        try:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return

    def _read_json_body(self) -> dict[str, Any]:
        """Read a JSON request body."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _serve_workspace_file(self, relative_url_path: str) -> None:
        """Serve a workspace file under /files/."""
        relative_path = unquote(relative_url_path).lstrip("/")
        try:
            path = resolve_workspace_path(relative_path)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        """Handle GET routes."""
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_text_file(DASHBOARD_FILE, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/dashboard":
                self._send_json(dashboard_payload())
                return
            if parsed.path == "/api/gpu-diagnostics":
                self._send_json(gpu_diagnostics())
                return
            if parsed.path.startswith("/api/enhance-status/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                job = get_enhance_job(job_id)
                if not job:
                    self._send_json(
                        {
                            "enhance_job_id": job_id,
                            "status": "error",
                            "error": "Unknown enhancement job.",
                        },
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json(job)
                return
            if parsed.path.startswith("/download/enhanced/"):
                filename = Path(unquote(parsed.path.rsplit("/", 1)[-1])).name
                target = (ENHANCED_DIR / filename).resolve()
                target.relative_to(ENHANCED_DIR.resolve())
                if not target.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{target.name}"'
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/files/"):
                self._serve_workspace_file(parsed.path.removeprefix("/files/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        """Handle POST routes."""
        parsed = urlparse(self.path)
        payload = self._read_json_body()

        if parsed.path == "/api/run":
            sources = payload.get("sources") or []
            if not sources and payload.get("source"):
                sources = [payload["source"]]

            if not sources:
                self._send_json(
                    {"ok": False, "message": "No source URLs provided"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            results = []
            for src in sources:
                if not src.strip():
                    continue
                # Create a specific config for this source
                config = {**payload, "source": src.strip()}
                ok, msg = JOB_MANAGER.start_job(config)
                results.append({"source": src, "ok": ok, "message": msg})

            queued_count = sum(1 for r in results if r["ok"])
            self._send_json(
                {
                    "ok": True,
                    "queued": queued_count,
                    "results": results,
                    "job": JOB_MANAGER.snapshot(),
                },
                status=HTTPStatus.OK,
            )
            return

        if parsed.path == "/api/ollama/start":
            ok, message = JOB_MANAGER.start_ollama()
            status = HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST
            self._send_json(
                {"ok": ok, "message": message, "setup": get_cached_setup(force=True)},
                status=status,
            )
            return

        if parsed.path == "/api/open":
            relative_path = str(payload.get("path", "")).strip() or "outputs/clips"
            try:
                target = resolve_workspace_path(relative_path)
            except ValueError:
                self._send_json(
                    {
                        "ok": False,
                        "message": "That path is outside the ClipForge workspace.",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if not target.exists():
                self._send_json(
                    {"ok": False, "message": "That path does not exist yet."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(target))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)], cwd=str(ROOT_DIR))
                else:
                    subprocess.Popen(["xdg-open", str(target)], cwd=str(ROOT_DIR))
            except OSError as exc:
                self._send_json(
                    {"ok": False, "message": f"Could not open path: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_json({"ok": True, "message": f"Opened {target.name}."})
            return

        if parsed.path == "/api/open-folder":
            rel_path = str(payload.get("path", "")).strip()
            if not rel_path:
                self._send_json(
                    {"ok": False, "error": "No path provided"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            abs_path = (ROOT_DIR / rel_path).resolve()
            try:
                abs_path.mkdir(parents=True, exist_ok=True)
                if platform.system() == "Windows":
                    os.startfile(str(abs_path))
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", str(abs_path)])
                else:
                    subprocess.Popen(["xdg-open", str(abs_path)])
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/set-encoder":
            raw_preference = payload.get("prefer_quality", True)
            if isinstance(raw_preference, str):
                prefer_quality = raw_preference.strip().lower() not in {
                    "false",
                    "0",
                    "off",
                    "no",
                }
            else:
                prefer_quality = bool(raw_preference)
            pipeline.ENCODER_PREFER_QUALITY = prefer_quality
            pipeline.detect_encoder.cache_clear()
            PAYLOAD_CACHE.update({"timestamp": 0.0, "payload": None})
            encoder_name, _ = pipeline.detect_encoder(
                prefer_quality=pipeline.ENCODER_PREFER_QUALITY
            )
            self._send_json(
                {
                    "ok": True,
                    "prefer_quality": pipeline.ENCODER_PREFER_QUALITY,
                    "encoder": encoder_name,
                    "speed_encoder": pipeline.detect_encoder(prefer_quality=False)[0],
                }
            )
            return

        if parsed.path == "/api/enhance":
            filename = Path(str(payload.get("filename", "")).strip()).name
            mode = str(payload.get("mode", "fast")).strip().lower()
            scale = int(payload.get("scale", 2) or 2)
            if mode not in {"fast", "ai"}:
                self._send_json(
                    {"ok": False, "message": "Mode must be 'fast' or 'ai'."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if scale not in {2, 4}:
                self._send_json(
                    {"ok": False, "message": "Scale must be 2 or 4."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                src = safe_output_clip(filename)
            except ValueError:
                self._send_json(
                    {"ok": False, "message": "Invalid filename."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if not src.exists():
                self._send_json(
                    {"ok": False, "message": "Clip not found in outputs/clips."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            enhance_job_id = f"enh_{uuid.uuid4().hex[:8]}"
            update_enhance_job(
                enhance_job_id,
                enhance_job_id=enhance_job_id,
                filename=filename,
                status="running",
                progress=0,
                output_filename=None,
                download_url=None,
                error=None,
                message="Starting...",
            )
            thread = threading.Thread(
                target=run_enhance,
                args=(filename, mode, scale, enhance_job_id),
                daemon=True,
            )
            thread.start()
            self._send_json({"enhance_job_id": enhance_job_id, "status": "started"})
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the local web app."""
    parser = argparse.ArgumentParser(description="ClipForge local dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=4173, help="Port to serve on")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the local ClipForge dashboard server."""
    args = parse_args(argv)
    pipeline.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    server = SilentThreadingHTTPServer((args.host, args.port), ClipForgeHandler)
    print(f"ClipForge dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
