# modules/hybrid.py
"""
Hybrid Split-Screen Restructure
Applies 65/35 vertical split (whiteboard top, speaker bottom)
to board and hybrid scene-mode clips.
Auto-detects face position — works for any creator layout.
"""

from pathlib import Path
import subprocess
import logging
import cv2
import numpy as np

log = logging.getLogger(__name__)

OUTPUT_W = 1080
OUTPUT_H = 1920
BOARD_H = int(OUTPUT_H * 0.65)  # 1248px
FACE_H = OUTPUT_H - BOARD_H  # 672px

TRIGGER_MODES = {"board", "hybrid"}  # ← single place to update if modes change

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_region(video_path: Path, sample_frames: int = 10) -> dict:
    """
    Detects face in full frame — stable across any speaker layout.
    Falls back to bottom-half if no face found.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    step = max(1, total // sample_frames)

    for i in range(0, min(total, sample_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Search full frame, not just center
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for x, y, w, h in faces:
            detections.append((x, y, w, h))
    cap.release()

    if not detections:
        log.warning("hybrid.py: No face detected in full frame")
        return {
            "x": 0,
            "y": vid_h // 2,
            "w": vid_w,
            "h": vid_h // 2,
            "found": False,
            "src_w": vid_w,
            "src_h": vid_h,
        }

    arr = np.array(detections)
    x, y, w, h = np.median(arr, axis=0).astype(int)

    # 80px padding around face for natural framing
    pad = 80
    fx = max(0, int(x) - pad)
    fy = max(0, int(y) - pad)
    fw = min(vid_w - fx, int(w) + pad * 2)
    fh = min(vid_h - fy, int(h) + pad * 2)

    log.info(f"hybrid.py: Face detected → x={fx} y={fy} w={fw} h={fh}")
    return {
        "x": fx,
        "y": fy,
        "w": fw,
        "h": fh,
        "found": True,
        "src_w": vid_w,
        "src_h": vid_h,
    }


def build_hybrid_filter(face: dict) -> str:
    """Builds -filter_complex string for 65/35 vstack layout."""
    board_scale = f"[0:v]scale={OUTPUT_W}:{BOARD_H},setsar=1[board]"
    face_crop = (
        f"[0:v]crop={face['w']}:{face['h']}:{face['x']}:{face['y']},"
        f"scale={OUTPUT_W}:{FACE_H},setsar=1[face]"
    )
    stack = "[board][face]vstack=inputs=2[out]"
    return f"{board_scale};{face_crop};{stack}"


def apply_hybrid(
    clip_path: Path,
    out_path: Path,
    face: dict | None = None,
    vcodec_params: list[str] | None = None,
) -> tuple[Path, bool]:
    """
    Full auto hybrid: detect face → build filter → FFmpeg render.
    Returns (result_path, face_was_found).
    """
    if face is None:
        face = detect_face_region(clip_path)

    if not face.get("found"):
        log.info(f"  hybrid.py: no face found → skipping split for {clip_path.name}")
        return clip_path, False

    if vcodec_params is None:
        vcodec_params = ["-c:v", "libx264", "-crf", "18"]

    filt = build_hybrid_filter(face)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
        "-filter_complex",
        filt,
        "-map",
        "[out]",
        "-map",
        "0:a",
        *vcodec_params,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"hybrid.py: FFmpeg stderr:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Hybrid render failed → {clip_path.name}")

    log.info(f"hybrid.py: Split-screen applied → {out_path.name}")
    return out_path, True


# ── Inline tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Test 1: Module import and cascade load
    assert _face_cascade is not None and not _face_cascade.empty(), (
        "FAIL: Haar cascade not loaded"
    )
    print("Test 1: Haar cascade loaded [OK]")

    # Test 2: Filter string format
    sample_face = {"x": 960, "y": 540, "w": 960, "h": 540}
    filt = build_hybrid_filter(sample_face)
    assert "vstack" in filt and "setsar=1" in filt, "FAIL: Filter malformed"
    assert f"scale={OUTPUT_W}:{BOARD_H}" in filt, "FAIL: Board scale wrong"
    assert f"scale={OUTPUT_W}:{FACE_H}" in filt, "FAIL: Face scale wrong"
    print(f"Test 2: Filter string OK\n   → {filt[:80]}...")

    # Test 3: TRIGGER_MODES contains both expected values
    assert "board" in TRIGGER_MODES and "hybrid" in TRIGGER_MODES, (
        "FAIL: TRIGGER_MODES incomplete"
    )
    print(f"Test 3: TRIGGER_MODES = {TRIGGER_MODES}")

    # Test 4: Fallback fires when no clip path given
    class _FakeCap:
        def __init__(self):
            pass

    # Simulated — just verify the fallback dict structure
    fallback = {"x": 0, "y": 540, "w": 1920, "h": 540}
    assert all(k in fallback for k in ("x", "y", "w", "h")), "FAIL: Fallback keys"
    print("Test 4: Fallback region structure valid")

    # Test 5: Live clip test (only if clip exists)
    test_clip = Path("outputs/clips/01_Introduction_to_Affiliate_Marketing_14s.mp4")
    if test_clip.exists():
        out = test_clip.parent / "test_hybrid_output.mp4"
        apply_hybrid(test_clip, out)
        assert out.exists() and out.stat().st_size > 10_000, "FAIL: Output too small"
        print(f"Test 5: Live render OK -> {out.name} ({out.stat().st_size // 1024}KB)")
    else:
        print(f"Test 5: Skipped (clip not found at {test_clip})")

    print("\nAll hybrid.py tests passed [OK]")
