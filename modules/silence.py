"""
silence.py — Conservative and Aggressive silence/filler removal.
"""

from __future__ import annotations
import re
import subprocess
import json
import logging
from pathlib import Path
from typing import Literal, List, NamedTuple

log = logging.getLogger(__name__)

# ── Mode Constants ─────────────────────────────────────────────────────────────
CONSERVATIVE = {"threshold_db": -40, "min_duration": 1.5, "max_cuts": 6}
AGGRESSIVE = {"threshold_db": -30, "min_duration": 0.4, "max_cuts": 14}

# Filler words to detect and remove in aggressive mode
FILLER_WORDS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bright\b",
    r"\bokay so\b",
    r"\bso yeah\b",
    r"\bi mean\b",
    r"\bkind of\b",
    r"\bsort of\b",
]


class SilenceSegment(NamedTuple):
    start: float
    end: float
    reason: str  # "silence" | "filler" | "pause"


# ── Core Public API ────────────────────────────────────────────────────────────
def remove_silences(
    src: Path,
    start: float,
    end: float,
    out: Path,
    encoder_flags: list[str] | None = None,
    mode: Literal["conservative", "aggressive"] = "conservative",
    transcript_segments: list | None = None,
    extra_vf: str = "",
) -> tuple[Path, list[tuple[float, float]]]:
    """
    Remove silences (and fillers in aggressive mode) from src[start:end] → out.
    Returns (output_path, kept_segments_list).
    """
    cfg = AGGRESSIVE if mode == "aggressive" else CONSERVATIVE

    # Pre-extract the segment to a temp file for silence detection
    # (Or we can run silencedetect on the full file with -ss -t)

    # Step 1: Detect silence segments
    # We run detection on the specific range
    silence_segs = _detect_silence_in_range(
        src, start, end, cfg["threshold_db"], cfg["min_duration"]
    )

    # Step 2: In aggressive mode, add filler word segments
    if mode == "aggressive" and transcript_segments:
        filler_segs = _detect_fillers(transcript_segments, start, end)
        silence_segs = _merge_segments(silence_segs + filler_segs)

    # Step 3: Cap cuts to max_cuts to avoid over-editing
    silence_segs = silence_segs[: cfg["max_cuts"]]

    if not silence_segs:
        return src, [(start, end)]

    # Step 4: Build FFmpeg select filter to keep non-silent parts
    # We need to invert these to get "keep" segments
    keep_segs = _invert_segments(silence_segs, start, end)

    if len(keep_segs) <= 1:
        return src, [(start, end)]

    out_path = _apply_cuts(src, out, keep_segs, encoder_flags, extra_vf)

    return out_path, keep_segs


# ── Silence Detection ──────────────────────────────────────────────────────────
def _detect_silence_in_range(
    src: Path, start: float, end: float, threshold_db: int, min_duration: float
) -> List[SilenceSegment]:
    """Run FFmpeg silencedetect on a specific range and parse output into segments."""
    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src),
        "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The output timestamps will be relative to 0 (the start of the slice)
    # We need to shift them back by 'start'
    rel_segs = _parse_silence_output(result.stderr)
    abs_segs = []
    for s in rel_segs:
        abs_segs.append(SilenceSegment(s.start + start, s.end + start, s.reason))
    return abs_segs


def _parse_silence_output(stderr: str) -> List[SilenceSegment]:
    segments = []
    starts = re.findall(r"silence_start: (\d+\.?\d*)", stderr)
    ends = re.findall(r"silence_end: (\d+\.?\d*)", stderr)
    for s, e in zip(starts, ends):
        segments.append(SilenceSegment(float(s), float(e), "silence"))
    return segments


# ── Filler Word Detection ──────────────────────────────────────────────────────
def _detect_fillers(
    transcript_segments: list, start: float, end: float
) -> List[SilenceSegment]:
    """
    Cross-reference transcript words with FILLER_WORDS list within a range.
    """
    filler_segs = []
    pattern = re.compile("|".join(FILLER_WORDS), re.IGNORECASE)

    for seg in transcript_segments:
        # Check if segment overlaps with our range
        if seg["end"] < start or seg["start"] > end:
            continue

        text = seg.get("text", "").strip()
        if pattern.search(text):
            # Add a small padding: don't cut the entire segment, just trim the filler
            duration = seg["end"] - seg["start"]
            if duration <= 0.8:  # short filler word → remove entirely
                filler_segs.append(SilenceSegment(seg["start"], seg["end"], "filler"))
    return filler_segs


# ── Segment Merging ────────────────────────────────────────────────────────────
def _merge_segments(
    segs: List[SilenceSegment], gap: float = 0.1
) -> List[SilenceSegment]:
    """Merge overlapping or near-adjacent segments."""
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda s: s.start)
    merged = [sorted_segs[0]]
    for curr in sorted_segs[1:]:
        prev = merged[-1]
        if curr.start <= prev.end + gap:
            merged[-1] = SilenceSegment(
                prev.start, max(prev.end, curr.end), prev.reason
            )
        else:
            merged.append(curr)
    return merged


# ── FFmpeg Cut Application ─────────────────────────────────────────────────────
def _apply_cuts(
    src: Path,
    out: Path,
    keep_segs: List[tuple[float, float]],
    encoder_flags: list[str] | None = None,
    extra_vf: str = "",
) -> Path:
    """
    Build an FFmpeg complex filter that removes the silence segments
    and concatenates the remaining audio+video parts seamlessly.
    """
    if not keep_segs:
        return src

    if encoder_flags is None:
        encoder_flags = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]

    # Build select + concat filter
    v_parts, a_parts = [], []
    filter_parts = []
    interleaved = []  # Interleaved [v0][a0][v1][a1]...

    # If extra_vf is provided (e.g. crop, eq), we apply it to the input stream 0:v
    # BEFORE the trim filters.
    base_v = "0:v"
    if extra_vf:
        filter_parts.append(f"[0:v]{extra_vf},setsar=1[vprep];")
        base_v = "vprep"

    for i, (s, e) in enumerate(keep_segs):
        filter_parts.append(
            f"[{base_v}]trim={s:.4f}:{e:.4f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS[a{i}];"
        )
        interleaved.append(f"[v{i}][a{i}]")

    n = len(keep_segs)
    concat_filter = (
        "".join(filter_parts)
        + "".join(interleaved)
        + f"concat=n={n}:v=1:a=1[vout][aout]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-filter_complex",
        concat_filter,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        *encoder_flags,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _invert_segments(
    silence_segs: List[SilenceSegment], start: float, end: float
) -> List[tuple[float, float]]:
    """Convert silence segments to keep segments within [start, end]."""
    keep = []
    cursor = start
    for seg in sorted(silence_segs, key=lambda s: s.start):
        if seg.start > cursor + 0.05:  # min 50ms keep segment
            keep.append((cursor, seg.start))
        cursor = seg.end
    if end - cursor > 0.05:
        keep.append((cursor, end))
    return keep


def _get_duration(src: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(src)],
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# ── Inline Tests ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test 1: Segment merging
    segs = [
        SilenceSegment(1.0, 2.0, "silence"),
        SilenceSegment(1.8, 3.0, "silence"),  # overlaps → should merge
        SilenceSegment(5.0, 6.0, "silence"),
    ]
    merged = _merge_segments(segs)
    assert len(merged) == 2, f"FAIL merge: expected 2 got {len(merged)}"
    assert merged[0].end == 3.0, "FAIL merge end"
    print("✅ Test 1 PASS: segment merging")

    # Test 2: Invert segments
    keep = _invert_segments([SilenceSegment(2.0, 4.0, "silence")], start=0.0, end=10.0)
    assert keep == [(0.0, 2.0), (4.0, 10.0)], f"FAIL invert: {keep}"
    print("✅ Test 2 PASS: segment inversion")

    # Test 3: Filler detection
    transcript = [
        {"start": 1.0, "end": 1.4, "text": "um"},
        {"start": 2.0, "end": 5.0, "text": "this is a real sentence"},
        {"start": 6.0, "end": 6.6, "text": "like"},
    ]
    fillers = _detect_fillers(transcript, start=0.0, end=10.0)
    assert len(fillers) == 2, f"FAIL fillers: expected 2 got {len(fillers)}"
    print("✅ Test 3 PASS: filler detection")

    # Test 4: Conservative config has longer min_duration
    assert CONSERVATIVE["min_duration"] > AGGRESSIVE["min_duration"]
    print("✅ Test 4 PASS: mode constants")

    # Test 5: Merge with gap tolerance
    close_segs = [
        SilenceSegment(1.0, 2.0, "silence"),
        SilenceSegment(2.05, 3.0, "silence"),  # 50ms gap → should merge
    ]
    merged2 = _merge_segments(close_segs, gap=0.1)
    assert len(merged2) == 1, f"FAIL gap merge: expected 1 got {len(merged2)}"
    print("✅ Test 5 PASS: gap tolerance merging")

    print("\n✅ All tests passed.")
