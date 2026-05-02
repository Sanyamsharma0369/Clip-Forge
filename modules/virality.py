from __future__ import annotations

import re
from typing import Any

# ── Weighted signal patterns ──────────────────────────────────────────────────
VIRALITY_SIGNALS: list[tuple[str, float]] = [
    # Money / numbers (highest weight)
    (r"\$[\d,]+", 3.0),
    (r"\b\d+[kK]\s*(views|followers|dollars|subscribers)", 2.5),
    (r"\b\d{4,}\b", 1.5),  # large raw numbers
    # Pattern interrupts
    (r"\b(secret|nobody|lie|lied|wrong|mistake|truth|exposed|shocking)\b", 2.0),
    (r"\b(they don.t want|what they hide|hidden|real reason)\b", 2.0),
    # Urgency / scarcity
    (r"\b(now|today|limited|stop|quit|immediately|urgent)\b", 1.8),
    # Questions (curiosity gap)
    (r"\?", 1.5),
    (r"\b(how|why|what if|what happens|did you know)\b", 1.4),
    # Emotional peaks
    (r"\b(never|always|every|fail|failed|fired|broke|lost|won|made)\b", 1.2),
    # Contrarian / bold claim
    (r"\b(actually|in reality|the truth is|most people|nobody tells)\b", 1.5),
    # Direct address
    (r"\b(you|your|yourself)\b", 0.8),
    # Social proof
    (r"\b(million|billion|thousand|viral|trending)\b", 1.3),
    # Domain-specific (Marketing/Business)
    (r"\b(affiliate|commission|passive|income|earn|earning|revenue)\b", 1.5),
    (r"\b(marketing|funnel|clickfunnels|bootcamp|russell)\b", 1.0),
    (r"\b(free|course|training|system|strategy|formula)\b", 1.2),
]

# Bonus: clip starts with a question or bold statement (hook quality)
HOOK_STARTERS: list[str] = [
    "what if",
    "did you know",
    "the truth",
    "nobody tells",
    "stop doing",
    "most people",
    "here's why",
    "i made",
    "i lost",
    "i quit",
    "the secret",
    "how i",
]


def score_virality(text: str) -> float:
    """
    Score a text snippet for viral potential.
    Returns a float 0.0–10.0. Higher = more likely to stop scroll.
    """
    if not text or not text.strip():
        return 0.0

    text_lower = text.lower()
    score = 0.0

    for pattern, weight in VIRALITY_SIGNALS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        score += len(matches) * weight

    # Bonus: strong hook starter in first 15 words
    first_words = " ".join(text_lower.split()[:15])
    for starter in HOOK_STARTERS:
        if starter in first_words:
            score += 2.0
            break

    # Bonus: short punchy sentence (under 12 words = high retention)
    word_count = len(text.split())
    if word_count <= 12:
        score += 1.0

    return round(min(score, 10.0), 2)


def score_clips(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Inject viral_score into each clip dict.
    Uses clip title + reason for scoring.
    Does NOT mutate originals — returns new dicts.
    """
    scored = []
    for clip in clips:
        # Include hook if available from earlier pipeline stages
        text = (
            f"{clip.get('title', '')} {clip.get('reason', '')} {clip.get('hook', '')}"
        )
        v_score = score_virality(text)
        llm_score = float(clip.get("score", 0.5))
        # Normalize scores > 1.0 (e.g. 9.5 -> 0.95)
        if llm_score > 1.0:
            llm_score /= 10.0

        # Weighted blend: 60% LLM content score + 40% virality heuristic
        blended = round(0.6 * llm_score + 0.4 * (v_score / 10.0), 4)
        updated = dict(clip)
        updated["viral_score"] = v_score
        updated["blended_score"] = blended
        updated["score"] = llm_score
        scored.append(updated)
    # Re-sort by blended score descending
    return sorted(scored, key=lambda c: c["blended_score"], reverse=True)


# ── Inline tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tests = [
        # (text, expected_min_score)
        ("I made $10,000 in 30 days and nobody told me this secret", 7.0),
        ("The truth is most people never learn this mistake", 5.0),
        ("Did you know? Stop doing this immediately", 5.0),
        ("Good morning everyone", 0.0),
        ("What if you could make $50k without a job?", 6.0),
    ]

    print("=== Virality Scorer Tests ===\n")
    all_passed = True
    for text, min_score in tests:
        result = score_virality(text)
        passed = result >= min_score
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} | Score: {result:5.2f} (min {min_score}) | '{text[:60]}'")
        if not passed:
            all_passed = False

    print(f"\n{'All tests passed [OK]' if all_passed else 'Some tests failed [ERR]'}")

    # Test score_clips batch
    sample_clips = [
        {
            "title": "How I made $5000 this week",
            "score": 0.9,
            "reason": "income reveal",
        },
        {"title": "Morning routine tips", "score": 0.85, "reason": "lifestyle content"},
        {
            "title": "The secret nobody tells you about affiliate marketing",
            "score": 0.7,
            "reason": "hidden truth",
        },
    ]
    print("\n=== Batch Clip Scoring ===\n")
    for c in score_clips(sample_clips):
        print(f"  blended={c['blended_score']} viral={c['viral_score']} | {c['title']}")
