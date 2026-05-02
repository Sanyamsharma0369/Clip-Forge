from __future__ import annotations

import logging
import shutil
import subprocess
import json
from pathlib import Path

LOGGER = logging.getLogger("clipforge")

# ── Tuning constants ──────────────────────────────────────────────────────────
MUSIC_VOLUME = 0.10  # 10% = -20dB under voice (inaudible but present)
MUSIC_FADE_OUT_SEC = 2.5  # fade music out in last 2.5 seconds
MUSIC_FADE_IN_SEC = 1.0  # fade music in over first 1 second
DEFAULT_MUSIC_DIR = Path("assets/music")  # drop .mp3 files here


def find_music_file(
    music_path: str | Path | None = None,
    music_dir: Path = DEFAULT_MUSIC_DIR,
) -> Path | None:
    """
    Resolve a music file to use.
    Priority:
      1. Explicit path passed by caller
      2. First .mp3 in assets/music/
      3. None (caller should skip music gracefully)
    """
    if music_path:
        p = Path(music_path)
        if p.exists():
            return p
        LOGGER.warning("audio.py: specified music file not found: %s", music_path)

    if music_dir.exists():
        mp3s = sorted(music_dir.glob("*.mp3"))
        if mp3s:
            LOGGER.info("audio.py: auto-selected music: %s", mp3s[0].name)
            return mp3s[0]

    LOGGER.info("audio.py: no music file found in %s — skipping", music_dir)
    return None


def build_music_filter(
    clip_duration: float,
    volume: float = MUSIC_VOLUME,
    fade_in: float = MUSIC_FADE_IN_SEC,
    fade_out: float = MUSIC_FADE_OUT_SEC,
) -> str:
    """
    Build an FFmpeg audio filter string for the background music stream.
    Trims music to clip duration, applies fade in/out and volume.
    """
    fade_out_start = max(0.0, clip_duration - fade_out)
    return (
        f"[1:a]"
        f"atrim=duration={clip_duration:.3f},"  # cut music to clip length
        f"afade=t=in:st=0:d={fade_in},"  # fade in
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out},"  # fade out
        f"volume={volume:.2f}"  # lower to background level
        f"[bg]"
    )


def apply_music(
    clip_path: Path,
    out_path: Path,
    music_dir: Path = Path("assets/music"),
    volume: float = MUSIC_VOLUME,
    fade_in: float = MUSIC_FADE_IN_SEC,
    fade_out: float = MUSIC_FADE_OUT_SEC,
    track_name: str | None = None,
) -> Path:
    """
    Mixes background music into clip.
    track_name: if set, uses that specific file from music_dir.
                if None, auto-selects first .mp3 alphabetically.
    """
    if track_name:
        # Use the specified track
        music_path = music_dir / track_name
        if not music_path.exists():
            available = [f.name for f in sorted(music_dir.glob("*.mp3"))]
            LOGGER.warning(
                f"audio.py: Specified track '{track_name}' not found. Available: {available}"
            )
            return clip_path  # safe fallback
        LOGGER.info(f"audio.py: using specified track: {track_name}")
    else:
        # Auto-select: first .mp3 alphabetically
        tracks = sorted(music_dir.glob("*.mp3"))
        if not tracks:
            LOGGER.warning(
                f"audio.py: No .mp3 files found in {music_dir} — skipping music"
            )
            return clip_path
        music_path = tracks[0]
        LOGGER.info(f"audio.py: auto-selected music: {music_path.name}")

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    # Get clip duration via ffprobe
    clip_duration = _get_duration(clip_path)
    if clip_duration <= 0:
        LOGGER.warning("audio.py: could not read clip duration — skipping music")
        return clip_path

    music_filter = build_music_filter(clip_duration, volume, fade_in, fade_out)

    # We need a temporary output path because ffmpeg cannot typically read and write the same file
    temp_output = out_path.with_suffix(".mix" + out_path.suffix)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(clip_path),  # input 0: rendered clip
        "-stream_loop",
        "-1",  # loop music if shorter than clip
        "-i",
        str(music_path),  # input 1: background music
        "-filter_complex",
        (
            f"{music_filter};"  # process music stream → [bg]
            f"[0:a][bg]amix=inputs=2:"  # mix voice + bg
            f"duration=first:"  # output length = clip length
            f"dropout_transition=2"  # smooth if music ends early
            f"[outa]"
        ),
        "-map",
        "0:v",  # video from clip
        "-map",
        "[outa]",  # mixed audio
        "-c:v",
        "copy",  # stream copy video (no re-encode)
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(temp_output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:])

        # Replace the original with the mixed version
        temp_output.replace(output_path)

        LOGGER.info(
            "audio.py: music mixed → %s (vol=%.0f%% fade_in=%.1fs fade_out=%.1fs)",
            output_path.name,
            volume * 100,
            fade_in,
            fade_out,
        )
        return output_path

    except Exception as exc:
        LOGGER.warning("audio.py: music mix failed (%s) — keeping original audio", exc)
        if temp_output.exists():
            try:
                temp_output.unlink()
            except Exception:
                pass
        return clip_path


def _get_duration(path: Path) -> float:
    """Get media duration in seconds via ffprobe."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = (
        str(ffmpeg_bin)
        .lower()
        .replace("ffmpeg.exe", "ffprobe.exe")
        .replace("ffmpeg", "ffprobe")
    )

    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


# ── Inline tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Audio Module Tests ===\n")

    # Test 1: find_music_file returns None gracefully when dir missing
    result = find_music_file(music_dir=Path("nonexistent_dir"))
    assert result is None
    print("[PASS] Test 1: Missing music dir -> returns None gracefully")

    # Test 2: find_music_file picks up explicit bad path gracefully
    result = find_music_file(music_path="nonexistent.mp3")
    assert result is None
    print("[PASS] Test 2: Bad explicit path -> returns None gracefully")

    # Test 3: build_music_filter contains all required parts
    f = build_music_filter(clip_duration=45.0)
    assert "atrim=duration=45.000" in f
    assert "afade=t=in" in f
    assert "afade=t=out" in f
    assert "volume=0.10" in f
    assert "[bg]" in f
    print(
        "[PASS] Test 3: build_music_filter() contains trim, fade-in, fade-out, volume"
    )

    # Test 4: fade_out_start is correctly calculated
    f = build_music_filter(clip_duration=30.0, fade_out=2.5)
    assert "st=27.500" in f  # 30.0 - 2.5 = 27.5
    print("[PASS] Test 4: fade_out_start = clip_duration - fade_out")

    # Test 5: very short clip — fade_out_start never goes negative
    f = build_music_filter(clip_duration=2.0, fade_out=2.5)
    assert "st=0.000" in f  # max(0, 2.0 - 2.5) = 0
    print("[PASS] Test 5: Short clip -> fade_out_start clamped to 0.0")

    print("\nAll tests passed [OK]")
