"""
modules/speaker_tracker.py
Multi-speaker diarization using pyannote/speaker-diarization-3.1
Maps speaker segments to Whisper transcript timestamps and assigns
face crop regions per speaker for automated camera switching.
"""

from __future__ import annotations
import os
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    speaker: str        # e.g. "SPEAKER_00", "SPEAKER_01"
    start:   float      # seconds
    end:     float      # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class WordSpeaker:
    word:    str
    start:   float
    end:     float
    speaker: str        # assigned speaker label


# ─────────────────────────────────────────────
# DIARIZATION CORE
# ─────────────────────────────────────────────

def diarize(audio_path: str, hf_token: str, num_speakers: Optional[int] = None) -> list[SpeakerSegment]:
    """
    Run pyannote speaker diarization on an audio file.

    Args:
        audio_path:   Path to .wav or .mp3 file (mono, any sample rate)
        hf_token:     HuggingFace access token (from .env HF_TOKEN)
        num_speakers: Force exact speaker count (None = auto-detect)

    Returns:
        List of SpeakerSegment sorted by start time
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "pyannote.audio not installed. Run:\n"
            "pip install pyannote.audio --upgrade"
        )

    log.info("Loading pyannote/speaker-diarization-community-1...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )

    # Push to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    log.info("Diarization running on: %s", device.upper())

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **kwargs)

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=round(turn.start, 3),
            end=round(turn.end, 3),
        ))

    segments.sort(key=lambda s: s.start)
    log.info("Diarization complete: %d segments, %d unique speakers",
             len(segments),
             len({s.speaker for s in segments}))
    return segments


# ─────────────────────────────────────────────
# SPEAKER → WORD ASSIGNMENT
# ─────────────────────────────────────────────

def assign_speakers_to_words(
    whisper_segments: list[dict],
    diarization:      list[SpeakerSegment],
    clip_start:       float = 0.0,
) -> list[WordSpeaker]:
    """
    Assigns a speaker label to each Whisper word by finding
    the diarization segment with maximum overlap.

    Args:
        whisper_segments: Raw Whisper output with word-level timestamps
        diarization:      Output of diarize()
        clip_start:       Clip start offset in the source video (seconds)

    Returns:
        List of WordSpeaker with absolute timestamps
    """
    result: list[WordSpeaker] = []

    for seg in whisper_segments:
        for w in seg.get("words", []):
            ws = w.get("start", seg["start"]) + clip_start
            we = w.get("end",   seg["end"])   + clip_start
            word = w.get("word", "").strip()
            if not word:
                continue

            # Find diarization segment with maximum overlap
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0
            for d in diarization:
                overlap = max(0.0, min(we, d.end) - max(ws, d.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d.speaker

            result.append(WordSpeaker(
                word=word,
                start=round(ws - clip_start, 3),
                end=round(we - clip_start, 3),
                speaker=best_speaker,
            ))

    return result


# ─────────────────────────────────────────────
# CROP REGION MAPPING
# ─────────────────────────────────────────────

def build_crop_timeline(
    diarization: list[SpeakerSegment],
    face_regions: dict[str, tuple[int, int, int, int]],
    clip_start:   float,
    clip_end:     float,
) -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """
    Builds a timeline of (start, end, crop_box) for FFmpeg.
    Each entry defines which face crop region is active for that time window.

    Args:
        diarization:  Speaker segments
        face_regions: Dict mapping speaker label → (x, y, w, h) crop box
                      e.g. {"SPEAKER_00": (0, 0, 540, 1920),
                             "SPEAKER_01": (540, 0, 540, 1920)}
        clip_start:   Clip start in source video
        clip_end:     Clip end in source video

    Returns:
        List of (rel_start, rel_end, crop_box) tuples
    """
    timeline = []
    default_box = list(face_regions.values())[0] if face_regions else (0, 0, 1080, 1920)

    for seg in diarization:
        seg_start = seg.start - clip_start
        seg_end   = seg.end   - clip_start

        # Clamp to clip window
        if seg_end <= 0 or seg_start >= (clip_end - clip_start):
            continue
        seg_start = max(0.0, seg_start)
        seg_end   = min(clip_end - clip_start, seg_end)

        box = face_regions.get(seg.speaker, default_box)
        timeline.append((round(seg_start, 3), round(seg_end, 3), box))

    return timeline


# ─────────────────────────────────────────────
# SPEAKER SUMMARY HELPER
# ─────────────────────────────────────────────

def speaker_summary(segments: list[SpeakerSegment]) -> dict[str, float]:
    """Returns total speaking time per speaker in seconds."""
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg.speaker] = round(totals.get(seg.speaker, 0.0) + seg.duration, 2)
    return dict(sorted(totals.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test 1: WordSpeaker assignment logic (no HF token needed)
    fake_diar = [
        SpeakerSegment("SPEAKER_00", 0.0, 5.0),
        SpeakerSegment("SPEAKER_01", 5.0, 10.0),
        SpeakerSegment("SPEAKER_00", 10.0, 15.0),
    ]
    fake_whisper = [{
        "start": 0.0, "end": 15.0,
        "words": [
            {"word": "Hello",   "start": 0.5,  "end": 1.0},
            {"word": "world",   "start": 1.0,  "end": 1.5},
            {"word": "this",    "start": 5.5,  "end": 6.0},
            {"word": "is",      "start": 6.0,  "end": 6.3},
            {"word": "speaker", "start": 6.3,  "end": 6.8},
            {"word": "two",     "start": 6.8,  "end": 7.2},
            {"word": "back",    "start": 10.5, "end": 11.0},
        ]
    }]

    words = assign_speakers_to_words(fake_whisper, fake_diar)
    assert words[0].speaker == "SPEAKER_00", "Word 0 should be SPEAKER_00"
    assert words[2].speaker == "SPEAKER_01", "Word 2 should be SPEAKER_01"
    assert words[6].speaker == "SPEAKER_00", "Word 6 should be SPEAKER_00"
    print("  Test 1 \u2014 speaker word assignment:  \u2705 PASS")

    # Test 2: Speaker summary
    summary = speaker_summary(fake_diar)
    assert summary["SPEAKER_00"] == 10.0
    assert summary["SPEAKER_01"] == 5.0
    print("  Test 2 \u2014 speaker summary totals:   \u2705 PASS")

    # Test 3: Crop timeline builder
    regions = {"SPEAKER_00": (0, 0, 540, 1920), "SPEAKER_01": (540, 0, 540, 1920)}
    timeline = build_crop_timeline(fake_diar, regions, clip_start=0.0, clip_end=15.0)
    assert len(timeline) == 3
    assert timeline[1][2] == (540, 0, 540, 1920), "Second segment should be SPEAKER_01 crop"
    print("  Test 3 \u2014 crop timeline builder:    \u2705 PASS")

    # Test 4: HF_TOKEN presence check
    token = os.getenv("HF_TOKEN", "")
    if token.startswith("hf_"):
        print("  Test 4 \u2014 HF_TOKEN in environment:  \u2705 PASS")
    else:
        print("  Test 4 \u2014 HF_TOKEN in environment:  \u26A0  NOT SET (add to .env)")

    print("\nAll offline tests passed. Run with --multi-speaker to test live diarization.")
