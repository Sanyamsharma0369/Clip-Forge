from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pipeline

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def parse_fps(rate: str) -> float:
    """Convert an ffprobe frame-rate string to a float."""
    if "/" not in rate:
        return float(rate or 0)
    numerator, denominator = rate.split("/", 1)
    return float(numerator) / float(denominator or 1)


def format_size_mb(size_mb: float) -> str:
    """Format a size in megabytes for console output."""
    return f"{size_mb:.1f} MB"


def format_bitrate_kbps(bitrate_kbps: int) -> str:
    """Format a bitrate in kilobits per second for console output."""
    return f"{bitrate_kbps:,} kbps"


def find_ffprobe_binary() -> str:
    """Locate the ffprobe executable beside ffmpeg when possible."""
    try:
        ffmpeg_binary = Path(find_ffmpeg_binary())
    except RuntimeError as exc:
        raise RuntimeError("ffprobe comes with FFmpeg — reinstall it") from exc
    sibling = ffmpeg_binary.with_name(
        "ffprobe.exe" if ffmpeg_binary.suffix.lower() == ".exe" else "ffprobe"
    )
    if sibling.exists():
        return str(sibling)
    raise RuntimeError("ffprobe comes with FFmpeg — reinstall it")


def find_ffmpeg_binary() -> str:
    """Locate FFmpeg with a user-friendly install error."""
    try:
        return pipeline.ensure_ffmpeg_on_path()
    except RuntimeError as exc:
        raise RuntimeError("Install FFmpeg: winget install ffmpeg") from exc


def get_source_info(path: Path) -> dict[str, Any]:
    """Return source media metadata using ffprobe."""
    ffprobe_binary = find_ffprobe_binary()
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,bit_rate,codec_name",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe comes with FFmpeg — reinstall it") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            exc.stderr.strip() or exc.stdout.strip() or "ffprobe failed"
        ) from exc

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), {}
    )
    format_info = payload.get("format", {})

    duration_sec = float(format_info.get("duration") or 0.0)
    size_bytes = int(float(format_info.get("size") or 0))
    video_bitrate = int(round(float(video_stream.get("bit_rate") or 0) / 1000))
    audio_bitrate = int(round(float(audio_stream.get("bit_rate") or 0) / 1000))

    if not video_bitrate and duration_sec > 0:
        video_bitrate = int(round((size_bytes * 8 / duration_sec) / 1000))

    return {
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": round(parse_fps(str(video_stream.get("r_frame_rate") or "0")), 3),
        "duration_sec": round(duration_sec, 3),
        "video_bitrate_kbps": video_bitrate,
        "audio_bitrate_kbps": audio_bitrate,
        "codec": str(video_stream.get("codec_name") or "").lower(),
        "size_mb": round(size_bytes / (1024 * 1024), 1),
    }


def needs_enhancement(info: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a clip misses the target quality and why."""
    reasons: list[str] = []
    if int(info["width"]) < 1080 or int(info["height"]) < 1920:
        reasons.append("resolution below 1080x1920")
    if int(info["video_bitrate_kbps"]) < 10000:
        reasons.append("video bitrate below 10,000 kbps")
    if str(info["codec"]) != "h264":
        reasons.append("codec is not h264")
    if int(info["audio_bitrate_kbps"]) < 160:
        reasons.append("audio bitrate below 160 kbps")
    return bool(reasons), reasons


def build_ffmpeg_command(input_path: Path, output_path: Path) -> list[str]:
    """Build the exact HQ ffmpeg command."""
    ffmpeg_binary = find_ffmpeg_binary()
    return [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1,"
        "unsharp=5:5:0.8:5:5:0.0",
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
        "21000k",
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
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def check_low_resolution(info: dict[str, Any]) -> None:
    """Warn when the source resolution is very low."""
    if min(int(info["width"]), int(info["height"])) < 720:
        print("⚠ Source is low resolution. Enhancement will upscale")
        print("  but cannot restore lost detail. Use a higher quality source.")


def summarize_info(name: str, info: dict[str, Any], reasons: list[str]) -> None:
    """Print the analysis block for a source clip."""
    print(f"🔍 Analyzing: {name}")
    print(
        "   Source: "
        f"{info['width']}x{info['height']} | "
        f"{format_bitrate_kbps(int(info['video_bitrate_kbps']))} | "
        f"{info['codec']} | "
        f"{info['duration_sec']:.1f}s | "
        f"{format_size_mb(float(info['size_mb']))}"
    )
    if reasons:
        print(f"   ⚠  Needs enhancement: {', '.join(reasons)}")
    else:
        print("   ✅ Already upload-ready")


def fix_quality(input_path: str) -> Path:
    """Enhance a single clip to the HQ upload target."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    info = get_source_info(source)
    needs_fix, reasons = needs_enhancement(info)
    summarize_info(source.name, info, reasons)
    check_low_resolution(info)

    if not needs_fix:
        return source

    output_path = source.with_name(f"HQ_{source.name}")
    print()
    print("🔧 Re-encoding with HQ settings...")
    print("   CRF 18 | preset slow | 14 Mbps | lanczos + unsharp")

    command = build_ffmpeg_command(source, output_path)
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install FFmpeg: winget install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "")[-500:]
        raise RuntimeError(f"FFmpeg encode error:\n{tail}") from exc

    original_size = round(source.stat().st_size / (1024 * 1024), 1)
    enhanced_size = round(output_path.stat().st_size / (1024 * 1024), 1)

    print()
    print("✅ Done!")
    print(f"   Original : {format_size_mb(original_size)}")
    print(f"   Enhanced : {format_size_mb(enhanced_size)}")
    print(f"   Saved to : {output_path.name}")
    print()
    print("📤 Ready to upload to Instagram Reels / YouTube Shorts")
    return output_path


def batch_fix(folder: str = "outputs/clips") -> list[Path]:
    """Enhance every non-HQ MP4 clip in a folder."""
    target_folder = Path(folder).expanduser().resolve()
    if not target_folder.exists():
        raise FileNotFoundError(f"File not found: {target_folder}")
    outputs: list[Path] = []
    for clip in sorted(target_folder.glob("*.mp4")):
        if clip.name.startswith("HQ_"):
            continue
        outputs.append(fix_quality(str(clip)))
        print()
    return outputs


def check_quality(input_path: str) -> tuple[bool, list[str]]:
    """Analyze a clip and report whether it needs enhancement."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    info = get_source_info(source)
    needs_fix, reasons = needs_enhancement(info)
    summarize_info(source.name, info, reasons)
    check_low_resolution(info)
    return needs_fix, reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the quality fixer."""
    parser = argparse.ArgumentParser(description="ClipForge video quality fixer")
    parser.add_argument("target", nargs="?", help="Clip path or batch folder")
    parser.add_argument(
        "--batch", action="store_true", help="Process all clips in a folder"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only analyze quality without re-encoding"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the quality fixer CLI."""
    args = parse_args(argv)
    try:
        if args.check:
            if not args.target:
                raise ValueError("Provide a clip path with --check.")
            check_quality(args.target)
            return 0
        if args.batch:
            batch_fix(args.target or "outputs/clips")
            return 0
        if not args.target:
            raise ValueError("Provide a clip path, use --batch, or use --check.")
        fix_quality(args.target)
        return 0
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    except ValueError as exc:
        print(str(exc))
        return 2
    except RuntimeError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
