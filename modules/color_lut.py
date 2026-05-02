from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger("clipforge")

# ── Tuning constants ──────────────────────────────────────────────────────────
DEFAULT_LUT_PATH = Path("assets/luts/technicolor.cube")
INTERPOLATION = "trilinear"  # or "tetrahedral"
LUT_INTENSITY = 0.6  # 60% intensity (blend with original)


def build_simple_lut_filter(
    lut_path: Path,
    interp: str = INTERPOLATION,
) -> str:
    """
    Simple single-filter version for -vf pipeline.
    Handles Windows drive letter escaping for FFmpeg.
    """
    # Convert to absolute path with forward slashes
    abs_path = str(lut_path.resolve())

    # Windows: D:\path\file.cube → D\:/path/file.cube
    # Step 1: replace backslashes with forward slashes
    abs_path = abs_path.replace("\\", "/")
    # Step 2: escape the colon after drive letter (D: → D\:)
    if len(abs_path) >= 2 and abs_path[1] == ":":
        abs_path = abs_path[0] + "\\:" + abs_path[2:]
    # Step 3: escape spaces (clip forge → clip\ forge)
    abs_path = abs_path.replace(" ", "\\ ")

    return f"lut3d=file='{abs_path}':interp={interp}"


def apply_lut_to_clip(
    clip_path: Path,
    lut_path: Path,
    output_path: Path,
    intensity: float = LUT_INTENSITY,
    vcodec_params: list[str] | None = None,
) -> Path:
    """
    Apply a 3D LUT to a rendered clip for cinematic color grading.
    Returns output_path on success, clip_path on failure.
    vcodec_params: list of encoding flags (e.g., LOSSLESS_ARGS).
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    if not lut_path.exists():
        LOGGER.warning("color_lut.py: LUT file not found: %s", lut_path)
        return clip_path

    if vcodec_params is None:
        vcodec_params = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]

    lut_filter = build_simple_lut_filter(lut_path)

    # We use a mix filter to control intensity (original vs graded)
    # Use [0:v] to create the graded stream, then mix back with [0:v]
    filter_graph = (
        f"[0:v]{lut_filter}[graded];"
        f"[0:v][graded]mix=weights='{1 - intensity} {intensity}'[outv]"
    )

    temp_output = output_path.with_suffix(".lut" + output_path.suffix)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(clip_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[outv]",
        "-map",
        "0:a",  # preserve audio
        *vcodec_params,
        "-c:a",
        "copy",
        str(temp_output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:])

        # Replace the original with the color-graded version
        temp_output.replace(output_path)

        LOGGER.info(
            "color_lut.py: LUT applied → %s (lut=%s intensity=%.0f%%)",
            output_path.name,
            lut_path.name,
            intensity * 100,
        )
        return output_path

    except Exception as exc:
        LOGGER.warning(
            "color_lut.py: color grading failed (%s) — keeping original", exc
        )
        if temp_output.exists():
            try:
                temp_output.unlink()
            except Exception:
                pass
        return clip_path


# ── Inline tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Color LUT Module Tests ===\n")

    # Test 1: Path escaping check (The "Windows Path Problem" fix)
    p = Path("D:/projects/clip forge/assets/luts/technicolor.cube")
    f = build_simple_lut_filter(p)
    print(f"Filter string: {f}")

    assert "lut3d=file='D\\:/projects/clip\\ forge/assets/luts/technicolor.cube'" in f
    assert "interp=trilinear" in f
    print("[PASS] Test 1: Windows Path Escaping (Colon + Spaces)")

    # Test 2: Intensity calculation
    # (Testing apply_lut logic would require a real video, so we skip for unit tests)

    print("\nAll tests passed [OK]")
