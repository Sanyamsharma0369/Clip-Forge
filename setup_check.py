from __future__ import annotations

import importlib
import shutil
import socket
import sys
from pathlib import Path


def status_line(ok: bool, label: str, fix: str = "") -> str:
    """Render a single setup check line."""
    prefix = "OK" if ok else "X"
    suffix = "" if ok or not fix else f" -> Fix: {fix}"
    return f"[{prefix}] {label}{suffix}"


def check_python() -> tuple[bool, str]:
    """Validate the Python version."""
    ok = sys.version_info >= (3, 10)
    return ok, status_line(
        ok, "Python 3.10+", "Install Python 3.10 or higher from python.org/downloads"
    )


def check_binary(name: str, label: str, fix: str) -> tuple[bool, str]:
    """Validate a binary on PATH."""
    ok = shutil.which(name) is not None
    return ok, status_line(ok, label, fix)


def find_ffmpeg() -> bool:
    """Return True when FFmpeg exists on PATH or in common Winget install locations."""
    if shutil.which("ffmpeg") is not None:
        return True
    if sys.platform.startswith("win"):
        package_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        for pattern in ("Gyan.FFmpeg.Essentials*", "Gyan.FFmpeg*", "BtbN.FFmpeg*"):
            for package_dir in package_root.glob(pattern):
                if any(package_dir.glob("**/ffmpeg.exe")):
                    return True
    return False


def check_module(module_name: str, label: str, fix: str) -> tuple[bool, str]:
    """Validate that a Python module imports."""
    try:
        importlib.import_module(module_name)
        return True, status_line(True, label)
    except Exception:
        return False, status_line(False, label, fix)


def check_ollama() -> tuple[bool, str]:
    """Validate that the Ollama server responds on localhost."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.5):
            return True, status_line(True, "ollama running")
    except OSError:
        return False, status_line(False, "ollama running", "Run: ollama serve")


def collect_checks() -> list[dict[str, str | bool]]:
    """Collect setup checks in a machine-readable structure."""
    ffmpeg_ok = find_ffmpeg()
    checks = [
        ("python", check_python()),
        (
            "ffmpeg",
            (
                ffmpeg_ok,
                status_line(
                    ffmpeg_ok, "ffmpeg binary", "Install FFmpeg and add it to PATH"
                ),
            ),
        ),
        ("yt_dlp", check_module("yt_dlp", "yt-dlp", "Run: pip install yt-dlp")),
        (
            "whisper",
            check_module(
                "whisper", "whisper (openai)", "Run: pip install openai-whisper"
            ),
        ),
        ("ollama", check_ollama()),
    ]
    return [
        {"id": check_id, "ok": ok, "message": line} for check_id, (ok, line) in checks
    ]


def main() -> int:
    """Run all setup checks and print the results."""
    checks = collect_checks()
    for check in checks:
        print(check["message"])
    return 0 if all(bool(check["ok"]) for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
