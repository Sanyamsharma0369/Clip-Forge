# tests/test_pipeline.py
import sys
import os

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import _normalize_segments, VAD_PARAMETERS, MAX_SEGMENT_DURATION


# ── Test 1: Short segment is never split ─────────────────────────────────────
def test_short_segment_unchanged():
    seg = [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
        }
    ]
    result = _normalize_segments(seg)
    assert len(result) == 1, "Short segment should not be split"


# ── Test 2: Long segment is split ────────────────────────────────────────────
def test_long_segment_splits():
    words = [
        {"word": f"w{i}", "start": float(i * 2), "end": float(i * 2 + 1.9)}
        for i in range(10)
    ]  # 20s total
    seg = [{"start": 0.0, "end": 19.0, "text": "...", "words": words}]
    result = _normalize_segments(seg)
    assert len(result) > 1, f"20s segment must be split (MAX={MAX_SEGMENT_DURATION})"
    for s in result:
        assert s["end"] - s["start"] <= MAX_SEGMENT_DURATION + 2.0


# ── Test 3: No word data = graceful fallback, no crash ───────────────────────
def test_no_words_fallback():
    seg = [{"start": 0.0, "end": 30.0, "text": "long segment", "words": []}]
    result = _normalize_segments(seg)
    assert len(result) == 1, "Missing word data should keep segment as-is"
    assert result[0]["text"] == "long segment"


# ── Test 4: VAD params are identical (the bug we just fixed) ─────────────────
def test_vad_params_constant_exists():
    assert "min_silence_duration_ms" in VAD_PARAMETERS
    assert "threshold" in VAD_PARAMETERS
    assert VAD_PARAMETERS["min_silence_duration_ms"] == 300
    assert VAD_PARAMETERS["threshold"] == 0.4


# ── Test 5: Segment text reconstructed from words correctly ──────────────────
def test_segment_text_from_words():
    words = [
        {"word": " Hello", "start": 0.0, "end": 0.5},
        {"word": " world", "start": 0.5, "end": 1.0},
    ]
    seg = [
        {"start": 0.0, "end": 20.0, "text": "Hello world", "words": words * 12}
    ]  # repeat to force split
    result = _normalize_segments(seg)
    for s in result:
        assert s["text"].strip() != "", "Sub-segment text must not be empty"
        assert "words" in s, "Sub-segment must retain word data"
