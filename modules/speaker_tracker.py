"""
speaker_tracker.py — Speaker diarization using pyannote-audio.
Detects who is speaking when, and maps speaker turns to transcript segments.
Runs AFTER Whisper to avoid VRAM conflicts (sequential GPU use).
"""

from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────
class SpeakerTurn(NamedTuple):
    start: float
    end: float
    speaker_id: str  # "SPEAKER_00", "SPEAKER_01", etc.


class AnnotatedSegment(NamedTuple):
    start: float
    end: float
    text: str
    speaker_id: str


# ── Core API ───────────────────────────────────────────────────────────────────
def diarize(
    video_path: Path,
    transcript_segments: list[dict],
    hf_token: str,
    device: str = "cpu",
) -> list[AnnotatedSegment]:
    """
    Run speaker diarization on video_path audio, then annotate
    each transcript segment with its speaker_id.

    Args:
        video_path:           Path to source video/audio
        transcript_segments:  List of {start, end, text} from Whisper
        hf_token:             HuggingFace token (required by pyannote)
        device:               "cuda" or "cpu"

    Returns:
        List of AnnotatedSegment with speaker_id attached to each segment
    """
    # Step 1: Extract audio to WAV (pyannote requires 16kHz mono WAV)
    wav_path = video_path.parent / f"{video_path.stem}_diarize.wav"
    _extract_audio(video_path, wav_path)

    # Step 2: Run pyannote diarization
    turns = _run_pyannote(wav_path, hf_token, device)

    # Step 3: Annotate transcript segments with speaker IDs
    annotated = _annotate_segments(transcript_segments, turns)

    # Cleanup temp WAV
    wav_path.unlink(missing_ok=True)

    log.info(
        f"  speaker_tracker: {len(set(s.speaker_id for s in annotated))} speakers "
        f"detected across {len(annotated)} segments"
    )
    return annotated


# ── Audio Extraction ───────────────────────────────────────────────────────────
def _extract_audio(video_path: Path, out_wav: Path) -> None:
    """Extract 16kHz mono WAV — required format for pyannote."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16kHz
        "-vn",  # no video
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ── Pyannote Diarization ───────────────────────────────────────────────────────
def _run_pyannote(wav_path: Path, hf_token: str, device: str) -> list[SpeakerTurn]:
    """
    Load pyannote speaker-diarization-3.1 pipeline and run inference.
    Lazily imported so the rest of ClipForge works without pyannote installed.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        raise RuntimeError(
            "pyannote-audio not installed.\n"
            "Run: pip install pyannote-audio\n"
            "And set HF_TOKEN in your .env file."
        )

    log.info("  speaker_tracker: loading pyannote/speaker-diarization-3.1 ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    log.info(
        "  speaker_tracker: running diarization (this takes ~30s per hour of audio) ..."
    )
    diarization = pipeline(str(wav_path))

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker_id=speaker,
            )
        )

    log.info(f"  speaker_tracker: {len(turns)} speaker turns detected")
    return turns


# ── Segment Annotation ─────────────────────────────────────────────────────────
def _annotate_segments(
    segments: list[dict],
    turns: list[SpeakerTurn],
) -> list[AnnotatedSegment]:
    """
    For each transcript segment, find the dominant speaker
    (the one with most overlap) from the diarization turns.
    """
    annotated = []
    for seg in segments:
        speaker = _dominant_speaker(seg["start"], seg["end"], turns)
        annotated.append(
            AnnotatedSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker_id=speaker,
            )
        )
    return annotated


def _dominant_speaker(
    seg_start: float,
    seg_end: float,
    turns: list[SpeakerTurn],
) -> str:
    """Return the speaker_id with the most overlap with [seg_start, seg_end]."""
    overlap: dict[str, float] = {}
    for turn in turns:
        o = _overlap(seg_start, seg_end, turn.start, turn.end)
        if o > 0:
            overlap[turn.speaker_id] = overlap.get(turn.speaker_id, 0) + o
    if not overlap:
        return "SPEAKER_00"  # default if no turn overlaps
    return max(overlap, key=overlap.get)


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    """Compute overlap duration between two time intervals."""
    return max(0.0, min(a2, b2) - max(a1, b1))


# ── Face Crop Router ───────────────────────────────────────────────────────────
def build_speaker_crop_filter(
    annotated_segments: list[AnnotatedSegment],
    face_positions: dict[str, tuple],  # speaker_id → (x, y, w, h) of their face
    output_w: int = 1080,
    output_h: int = 1920,
) -> str:
    """
    Build an FFmpeg filter_complex that switches face crop target
    based on active speaker at each timestamp.

    face_positions: pre-computed per-speaker face bounding boxes
    Returns: vf filter string ready for -vf argument
    """
    if not face_positions or len(face_positions) < 2:
        log.info(
            "  speaker_tracker: <2 face positions — using standard single-face crop"
        )
        return _single_face_crop(output_w, output_h)

    # Build zoompan keyframe expression switching between speaker crops
    # Each speaker has a target (x, y) center
    segments_sorted = sorted(annotated_segments, key=lambda s: s.start)

    # Convert face boxes to crop centers
    centers = {
        sid: (fx + fw // 2, fy + fh // 2)
        for sid, (fx, fy, fw, fh) in face_positions.items()
    }

    # Build per-segment crop list for FFmpeg trim+crop+concat approach
    filter_parts = []
    v_labels = []
    a_labels = []

    for i, seg in enumerate(segments_sorted):
        speaker = seg.speaker_id
        cx, cy = centers.get(speaker, centers.get("SPEAKER_00", (960, 540)))

        # Crop window: 1080 wide centered on speaker face
        src_w = 1920
        crop_x = max(0, min(cx - output_w // 2, src_w - output_w))
        crop_h = int(output_w * output_h / output_w)  # = output_h for 9:16

        filter_parts.append(
            f"[0:v]trim={seg.start:.3f}:{seg.end:.3f},setpts=PTS-STARTPTS,"
            f"crop={output_w}:{output_h}:{crop_x}:0[v{i}];"
            f"[0:a]atrim={seg.start:.3f}:{seg.end:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    n = len(segments_sorted)
    concat = (
        "".join(filter_parts)
        + "".join(v_labels)
        + "".join(a_labels)
        + f"concat=n={n}:v=1:a=1[vout][aout]"
    )
    return concat


def _single_face_crop(w: int, h: int) -> str:
    return f"crop={w}:{h}:(iw-{w})/2:0"


# ── Inline Tests ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test 1: Overlap calculation
    assert _overlap(0, 5, 3, 8) == 2.0
    assert _overlap(0, 5, 6, 8) == 0.0
    assert _overlap(2, 4, 1, 5) == 2.0
    print("✅ Test 1 PASS: overlap calculation")

    # Test 2: Dominant speaker selection
    turns = [
        SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
        SpeakerTurn(3.0, 6.0, "SPEAKER_01"),
        SpeakerTurn(4.5, 5.5, "SPEAKER_00"),  # short overlap
    ]
    # Segment 4.0-5.0: SPEAKER_01 has 1.0s overlap, SPEAKER_00 has 0.5s → SPEAKER_01 wins
    result = _dominant_speaker(4.0, 5.0, turns)
    assert result == "SPEAKER_01", f"FAIL: got {result}"
    print("✅ Test 2 PASS: dominant speaker with overlap tie-breaking")

    # Test 3: Segment annotation
    segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
        {"start": 3.5, "end": 5.5, "text": "Thanks for having me"},
    ]
    turns2 = [
        SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
        SpeakerTurn(3.0, 8.0, "SPEAKER_01"),
    ]
    annotated = _annotate_segments(segments, turns2)
    assert annotated[0].speaker_id == "SPEAKER_00"
    assert annotated[1].speaker_id == "SPEAKER_01"
    print("✅ Test 3 PASS: segment annotation")

    # Test 4: Default speaker when no overlap
    result2 = _dominant_speaker(10.0, 12.0, turns)  # no turns in this range
    assert result2 == "SPEAKER_00"
    print("✅ Test 4 PASS: default speaker fallback")

    # Test 5: Speaker crop filter with single speaker → simple crop
    result3 = build_speaker_crop_filter([], {}, 1080, 1920)
    assert "crop=1080:1920" in result3
    print("✅ Test 5 PASS: single-speaker fallback crop filter")

    print("\n✅ All tests passed.")
