from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

LOGGER = logging.getLogger("clipforge")

# ── Tuning constants ──────────────────────────────────────────────────────────
FACE_DETECT_EVERY_N_FRAMES = 60  # was 15 → 4x fewer samples
ANALYSIS_WIDTH = 640  # downsample to this width before analysis
MAX_SAMPLES = 30  # hard cap regardless of video length
BRIGHT_PIXEL_THRESHOLD = 200  # gray value considered "bright" (whiteboard/screen)
BOARD_BRIGHT_RATIO = 0.25  # caught more board frames
FACE_BRIGHT_HYBRID = 0.15  # trigger hybrid earlier
EDGE_DENSITY_THRESHOLD = 0.05  # was 0.08 — catches sparse writing like single "$"
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGH_BORS = 5
FACE_MIN_SIZE_RATIO = 0.05  # minimum face px relative to frame width
MIN_SCREEN_AREA_RATIO = 0.15  # whiteboard/screen must be at least 15% of frame
SCREEN_BRIGHT_THRESHOLD = 200  # gray value for screen detection


def _downsample_frame(frame: np.ndarray, target_width: int = ANALYSIS_WIDTH):
    """
    Resize frame to target_width preserving aspect ratio.
    Returns (small_frame, scale_x, scale_y) for coordinate mapping back.
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0, 1.0  # already small enough
    scale = target_width / w
    new_h = int(h * scale)
    small = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return small, scale, scale


class FrameAnalysis(NamedTuple):
    mode: str  # 'face' | 'board' | 'hybrid'
    bright_ratio: float
    edge_density: float
    has_face: bool
    board_region: tuple[int, int, int, int] | None  # x,y,w,h of brightest region


def detect_animation(frame: np.ndarray) -> bool:
    """
    Detects cartoon/animation content using outline structure analysis.
    Cartoons have thick, uniform black outlines surrounding flat color fills.
    This works regardless of background color or saturation level.
    """
    if frame is None:
        return False

    # Downsample for speed
    small, _, _ = _downsample_frame(frame, target_width=400)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Signal 1: Strong, thick black outlines (Canny edge detection) ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    # --- Signal 2: Flat color regions (low variance inside filled areas) ---
    # Cartoon fills are uniform — very low local std deviation
    kernel = np.ones((8, 8), np.float32) / 64
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    # Ensure no negative variance due to precision
    local_var = np.maximum(local_var, 0)
    flat_region_ratio = np.sum(local_var < 150) / (h * w)

    # --- Signal 3: Very dark pixels form thin connected lines (outlines) ---
    dark_mask = (gray < 60).astype(np.uint8)
    dark_ratio = np.sum(dark_mask) / (h * w)

    # Animation fingerprint:
    # High edges (outlines) + high flat regions (fills) + moderate dark (outlines not shadows)
    is_animation = (
        edge_density > 0.06  # strong contour lines
        and flat_region_ratio > 0.55  # large flat color fills
        and 0.02 < dark_ratio < 0.25  # outline-level dark pixels (not a dark scene)
    )

    return is_animation


def has_distinct_screen_region(frame: np.ndarray) -> bool:
    """
    Returns True only if the frame contains a distinct rectangular
    bright/white region that looks like a whiteboard, screen, or slide.
    A car interior or garage does NOT qualify.
    """
    if frame is None:
        return False

    # Downsample for speed
    small, _, _ = _downsample_frame(frame, target_width=400)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Threshold for bright rectangular regions (screens/whiteboards are bright)
    _, thresh = cv2.threshold(gray, SCREEN_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Must be at least 15% of frame area to count as a "screen"
        if area < frame_area * MIN_SCREEN_AREA_RATIO:
            continue

        # Must be roughly rectangular (aspect ratio check)
        x, y, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / max(rh, 1)
        if 0.5 < aspect < 3.0:  # plausible screen shape
            return True

    return False


def classify_mode(
    bright: float, edge: float, face: bool, frame: np.ndarray = None
) -> str:
    """
    Core classification logic with screen and animation guards.
    """
    # Animation guard — cartoons often have high edges but should be pure face mode
    if frame is not None and detect_animation(frame):
        return "face"

    # Screen/board guard — only go hybrid/board if actual screen detected
    has_screen = has_distinct_screen_region(frame)

    if not has_screen:
        # No whiteboard/screen visible → always pure face mode
        # Catches car interiors, garages, busy backgrounds
        return "face"

    # Screen IS present — now classify based on features
    # Blackboard check: dark background + ANY bright strokes = hybrid
    if bright < 0.35 and edge > 0.03 and face:
        return "hybrid"

    if face:
        if edge < 0.05:
            return "face"  # clean background, pure talking head
        return "hybrid"  # face + screen content = hybrid

    # No face detected:
    if edge > 0.05 or bright > 0.25:
        return "board"  # board/screen content only

    if bright < 0.10 and edge < 0.05:
        return "face"  # dark vlog fallback

    return "face"


def _load_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Haar cascade not found — check OpenCV installation.")
    return cascade


def analyse_frame(
    frame: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> FrameAnalysis:
    """
    Classify a single BGR frame as face / board / hybrid.
    Runs on DOWNSAMPLED copy for speed.
    """
    small, sx, sy = _downsample_frame(frame)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # ── Face detection ────────────────────────────────────────────────────────
    min_face_px = int(w * FACE_MIN_SIZE_RATIO)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGH_BORS,
        minSize=(min_face_px, min_face_px),
    )
    has_face = len(faces) > 0

    # ── Bright region detection (whiteboard / screen / slide) ─────────────────
    bright_mask = gray > BRIGHT_PIXEL_THRESHOLD
    bright_ratio = float(np.sum(bright_mask)) / gray.size

    # Find largest contiguous bright region (the board/screen rectangle)
    board_region = None
    bright_uint8 = bright_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        bright_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        region_area = cv2.contourArea(largest)
        if region_area > (w * h * 0.05):  # ignore noise < 5% of frame
            bx, by, bw, bh = cv2.boundingRect(largest)
            board_region = (bx, by, bw, bh)

    # ── Edge density (screens/slides have sharp uniform edges) ────────────────
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / gray.size

    # ── Classify mode ─────────────────────────────────────────────────────────
    mode = classify_mode(bright_ratio, edge_density, has_face, frame=frame)

    return FrameAnalysis(
        mode=mode,
        bright_ratio=round(bright_ratio, 3),
        edge_density=round(edge_density, 3),
        has_face=has_face,
        board_region=board_region,
    )


def detect_clip_mode(
    video_path: Path,
    start_sec: float,
    end_sec: float,
) -> str:
    """
    Sample frames across a clip window and return the dominant mode.
    Returns: 'face' | 'board' | 'hybrid'
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("scene_crop: could not open video, defaulting to face mode.")
        return "face"

    cascade = _load_cascade()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mode_counts: dict[str, int] = {"face": 0, "board": 0, "hybrid": 0}
    frame_idx = start_frame
    total_frames = end_frame - start_frame
    step = max(FACE_DETECT_EVERY_N_FRAMES, total_frames // MAX_SAMPLES)

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame at current index
        analysis = analyse_frame(frame, cascade)
        mode_counts[analysis.mode] += 1
        LOGGER.debug(
            "  scene_crop frame %d: mode=%s bright=%.2f edge=%.2f face=%s",
            frame_idx,
            analysis.mode,
            analysis.bright_ratio,
            analysis.edge_density,
            analysis.has_face,
        )

        # Advance multiple frames
        frame_idx += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()

    total = sum(mode_counts.values()) or 1
    dominant = max(mode_counts, key=lambda m: mode_counts[m])

    LOGGER.info(
        "  scene_crop clip %.1fs-%.1fs → dominant mode: %s "
        "(face=%d board=%d hybrid=%d / %d samples)",
        start_sec,
        end_sec,
        dominant,
        mode_counts["face"],
        mode_counts["board"],
        mode_counts["hybrid"],
        total,
    )
    return dominant


def build_smart_crop_filter(
    mode: str,
    frame_centers: list[tuple[float, float]],
    source_width: int,
    source_height: int,
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """
    Return the correct FFmpeg -vf crop/scale string based on detected mode.

    face   → follow detected face center (existing behavior)
    board  → letterbox full frame, preserve all content
    hybrid → top 65% shows board, bottom 35% shows face (stacked)
    """
    if mode == "face":
        # Delegate to existing face crop logic — return sentinel
        return "__USE_FACE_CROP__"

    if mode == "board":
        # Letterbox: scale to fit full frame, pad black bars
        return (
            f"scale={output_width}:{output_height}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:"
            f"(ow-iw)/2:(oh-ih)/2:color=black"
        )

    if mode == "hybrid":
        # Hybrid mode: Signal pipeline to preserve full 16:9 frame for apply_hybrid
        return "__PRESERVE_16_9__"

    return "__USE_FACE_CROP__"  # safe fallback


# ── Inline tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Scene Crop Module Tests ===\n")

    cascade = _load_cascade()

    # Test 1: Bright white frame → board mode
    white_frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    result = analyse_frame(white_frame, cascade)
    assert result.mode == "board", f"Expected board, got {result.mode}"
    assert result.bright_ratio > BOARD_BRIGHT_RATIO
    print(
        f"[PASS] Test 1: White frame → mode='{result.mode}' bright={result.bright_ratio}"
    )

    # Test 2: Dark frame, no face → face mode (vlog fallback)
    dark_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = analyse_frame(dark_frame, cascade)
    assert result.mode == "face"
    print(f"[PASS] Test 2: Dark frame → mode='{result.mode}' face={result.has_face}")

    # Test 3: Mixed frame (top half bright, bottom half dark) → board
    mixed_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mixed_frame[:540, :] = 230  # top half bright (whiteboard)
    result = analyse_frame(mixed_frame, cascade)
    assert result.mode == "board"
    print(f"[PASS] Test 3: Half-bright frame → mode='{result.mode}'")

    # Test 4: board filter string — must contain letterbox keywords
    board_filter = build_smart_crop_filter("board", [], 1920, 1080)
    assert "force_original_aspect_ratio=decrease" in board_filter
    assert "pad=" in board_filter
    print("[PASS] Test 4: board filter contains letterbox logic")

    # Test 5: face mode returns sentinel
    face_filter = build_smart_crop_filter("face", [(0.5, 0.35)], 1920, 1080)
    assert face_filter == "__USE_FACE_CROP__"
    print("[PASS] Test 5: face mode returns __USE_FACE_CROP__ sentinel")

    print("\nAll tests passed [OK]")
    print("\n=== Sample Filter Outputs ===")
    print(f"\n[BOARD]\n{build_smart_crop_filter('board', [], 1920, 1080)}")
