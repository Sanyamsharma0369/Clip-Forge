"""
modules/overlays.py
ClipForge subtitle overlay engine — Styles 1-9
Generates .ass subtitle files from Whisper word-level segments.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────


@dataclass
class WordToken:
    word: str
    start: float
    end: float


@dataclass
class SubLine:
    words: list[WordToken]
    start: float
    end: float
    emoji: str = ""

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)

    @property
    def display_text(self) -> str:
        t = self.text
        return f"{t} {self.emoji}" if self.emoji else t


# ─────────────────────────────────────────────
# ASS HELPERS
# ─────────────────────────────────────────────


def _ts(seconds: float) -> str:
    """Convert seconds → ASS timestamp  H:MM:SS.cs"""
    cs = int(round((seconds % 1) * 100))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_header(style_block: str) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,Strikeout,"
        "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"{style_block}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )


def _dialogue(start: float, end: float, style: str, text: str, layer: int = 0) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}\n"


# ─────────────────────────────────────────────
# SEGMENTER  (shared by all styles)
# ─────────────────────────────────────────────


def _segment_words(
    segments: list[dict],
    max_words: int = 5,
    max_chars: int = 32,
    clip_start: float = 0.0,
    clip_end: float = float("inf"),
) -> list[SubLine]:
    """
    Flatten Whisper word-level segments into SubLine objects
    clipped to the clip's time window.
    """
    tokens: list[WordToken] = []
    for seg in segments:
        for w in seg.get("words", []):
            ws = w.get("start", seg["start"])
            we = w.get("end", seg["end"])
            if we <= clip_start or ws >= clip_end:
                continue
            ws = max(ws, clip_start) - clip_start
            we = min(we, clip_end) - clip_start
            word = w.get("word", "").strip()
            if word:
                tokens.append(WordToken(word, round(ws, 3), round(we, 3)))

    lines: list[SubLine] = []
    buf: list[WordToken] = []

    for tok in tokens:
        buf.append(tok)
        line_text = " ".join(t.word for t in buf)
        if len(buf) >= max_words or len(line_text) >= max_chars:
            lines.append(SubLine(buf[:], buf[0].start, buf[-1].end))
            buf = []

    if buf:
        lines.append(SubLine(buf[:], buf[0].start, buf[-1].end))

    return lines


# ─────────────────────────────────────────────
# EMOJI INJECTION  (Style 9 / shared utility)
# ─────────────────────────────────────────────

# Lightweight keyword→emoji map (no API call needed for common words)
_EMOJI_MAP: dict[str, str] = {
    # Emotions
    "happy": "😊",
    "sad": "😢",
    "angry": "😤",
    "love": "❤️",
    "fear": "😨",
    "shocked": "😱",
    "laugh": "😂",
    "cry": "😭",
    # Money / business
    "money": "💰",
    "rich": "🤑",
    "bank": "🏦",
    "invest": "📈",
    "business": "💼",
    "profit": "💵",
    "sales": "📊",
    # Action
    "run": "🏃",
    "fight": "🥊",
    "win": "🏆",
    "fail": "❌",
    "build": "🏗️",
    "grow": "🌱",
    "learn": "📚",
    "teach": "🎓",
    # Tech
    "phone": "📱",
    "computer": "💻",
    "ai": "🤖",
    "code": "👨‍💻",
    # Life
    "family": "👨‍👩‍👧",
    "friend": "🤝",
    "god": "🙏",
    "death": "💀",
    "food": "🍽️",
    "travel": "✈️",
    "car": "🚗",
    "house": "🏠",
    # Generic intensifiers
    "secret": "🤫",
    "truth": "💯",
    "fact": "📌",
    "never": "🚫",
    "always": "✅",
    "best": "🔥",
    "worst": "💩",
    "first": "🥇",
}


def _pick_emoji(text: str, gemini_fn=None) -> str:
    """
    Returns a single emoji for a subtitle line.
    Uses the keyword map for speed; falls back to Gemini if a callable is supplied
    and no keyword matches.
    """
    lower = text.lower()
    for kw, em in _EMOJI_MAP.items():
        if kw in lower:
            return em
    if gemini_fn:
        try:
            result = gemini_fn(
                f"Reply with ONE emoji that best matches this short text. "
                f"No explanation, just the emoji:\n{text}"
            )
            result = result.strip()
            if len(result) <= 4:  # guard against verbose replies
                return result
        except Exception as e:
            log.debug("Emoji Gemini fallback failed: %s", e)
    return ""


# ─────────────────────────────────────────────
# STYLE RENDERERS
# ─────────────────────────────────────────────


# ── Style 1: TikTok Yellow ──────────────────────────────────────────────────
def _style1_tiktok_yellow(lines: list[SubLine], path: str) -> int:
    style = (
        "Style: Default,Arial Black,88,&H0000FFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,80,80,120,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2\b1}" + ln.display_text.upper()
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 2: White Bold ─────────────────────────────────────────────────────
def _style2_white_bold(lines: list[SubLine], path: str) -> int:
    style = (
        "Style: Default,Arial,82,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,80,80,120,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2\b1}" + ln.display_text
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 3: Neon Green ─────────────────────────────────────────────────────
def _style3_neon_green(lines: list[SubLine], path: str) -> int:
    style = (
        "Style: Default,Impact,90,&H0000FF00,&H000000FF,"
        "&H00000000,&HA0000000,-1,0,0,0,100,100,2,0,1,3,0,2,80,80,130,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2\b1\shad0}" + ln.display_text.upper()
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 4: Cinematic ──────────────────────────────────────────────────────
def _style4_cinematic(lines: list[SubLine], path: str) -> int:
    style = (
        "Style: Default,Georgia,72,&H00FFFFFF,&H000000FF,"
        "&H00000000,&HC0000000,0,1,0,0,100,100,3,0,1,2,0,2,80,80,160,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2\i1}" + ln.display_text
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 5: Word-by-Word Pop ───────────────────────────────────────────────
def _style5_word_pop(lines: list[SubLine], path: str) -> int:
    """
    MrBeast-style: each word punches in (scale 120→100) for its own duration,
    then the full line stays dimmed while the next word is highlighted.
    Uses ASS override tags with \\t() transform for the scale punch.
    """
    style_active = (
        "Style: Active,Arial Black,92,&H0000FFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,80,80,120,1"
    )
    style_dim = (
        "Style: Dim,Arial Black,92,&H80AAAAAA,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,120,1"
    )
    header = (
        "[Script Info]\\n"
        "ScriptType: v4.00+\\n"
        "PlayResX: 1080\\n"
        "PlayResY: 1920\\n"
        "ScaledBorderAndShadow: yes\\n\\n"
        "[V4+ Styles]\\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,Strikeout,"
        "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\\n"
        f"{style_active}\\n{style_dim}\\n\\n"
        "[Events]\\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\\n"
    ).replace("\\n", "\n")
    out = header
    for ln in lines:
        for i, wt in enumerate(ln.words):
            # Build full line with current word highlighted via colour override
            parts = []
            for j, w2 in enumerate(ln.words):
                if j == i:
                    # Active word: yellow, scale punch 120→100 over 80ms
                    parts.append(
                        r"{\c&H00FFFF&\fscx120\fscy120"
                        r"\t(0,80,\fscx100\fscy100)}" + w2.word.upper()
                    )
                else:
                    parts.append(r"{\c&HAAAAAA&\fscx100\fscy100}" + w2.word.upper())
            txt = r"{\an2}" + " ".join(parts)
            out += _dialogue(wt.start, wt.end, "Active", txt)

    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return sum(len(ln.words) for ln in lines)


# ── Style 6: Gradient Wave ──────────────────────────────────────────────────
def _style6_gradient_wave(lines: list[SubLine], path: str) -> int:
    """
    Each word in the line cycles through a yellow→white→yellow gradient
    based on its position index, giving a "wave" colour effect.
    """
    style = (
        "Style: Default,Arial Black,88,&H0000FFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,80,80,120,1"
    )
    # Colour stops: yellow → white → yellow
    COLORS = [
        "&H0000FFFF&",
        "&H0044FFFF&",
        "&H00AAFFFF&",
        "&H00FFFFFF&",
        "&H00AAFFFF&",
        "&H0044FFFF&",
    ]
    out = _ass_header(style)
    for ln in lines:
        n = len(ln.words)
        parts = []
        for i, wt in enumerate(ln.words):
            c = COLORS[i % len(COLORS)]
            parts.append(rf"{{\c{c}}}" + wt.word.upper())
        txt = r"{\an2}" + " ".join(parts)
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 7: Outlined Clean ─────────────────────────────────────────────────
def _style7_outlined_clean(lines: list[SubLine], path: str) -> int:
    """
    White text, thick black border (outline=5), zero shadow.
    Best for educational content on any background.
    """
    style = (
        "Style: Default,Arial,86,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,0,2,80,80,120,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2\b1\shad0}" + ln.display_text
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 8: Minimal Lowercase ──────────────────────────────────────────────
def _style8_minimal_lowercase(lines: list[SubLine], path: str) -> int:
    """
    Calm aesthetic: small-ish font, all lowercase, bottom-center.
    Popular in lifestyle / day-in-the-life content.
    """
    style = (
        "Style: Default,Helvetica Neue,64,&H00FFFFFF,&H000000FF,"
        "&H00000000,&H60000000,0,0,0,0,100,100,1,0,1,2,1,2,80,80,140,1"
    )
    out = _ass_header(style)
    for ln in lines:
        txt = r"{\an2}" + ln.display_text.lower()
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ── Style 9: Emoji Inject ───────────────────────────────────────────────────
def _style9_emoji_inject(lines: list[SubLine], path: str, gemini_fn=None) -> int:
    """
    TikTok Yellow base + one contextual emoji appended to each line.
    Uses keyword map first; Gemini as optional fallback.
    """
    style = (
        "Style: Default,Arial Black,88,&H0000FFFF,&H000000FF,"
        "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,80,80,120,1"
    )
    out = _ass_header(style)
    for ln in lines:
        ln.emoji = _pick_emoji(ln.text, gemini_fn)
        txt = r"{\an2\b1}" + ln.display_text.upper()
        out += _dialogue(ln.start, ln.end, "Default", txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(lines)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

STYLE_NAMES = {
    1: "TikTok Yellow",
    2: "White Bold",
    3: "Neon Green",
    4: "Cinematic",
    5: "Word-by-Word Pop",
    6: "Gradient Wave",
    7: "Outlined Clean",
    8: "Minimal Lowercase",
    9: "Emoji Inject",
}


def write_ass(
    segments: list[dict],
    path: str,
    style: int = 1,
    clip_start: float = 0.0,
    clip_end: float = float("inf"),
    max_words: int = 5,
    gemini_fn: Optional[callable] = None,
) -> int:
    """
    Generate a .ass subtitle file from Whisper word-level segments.

    Args:
        segments:   Whisper output (list of dicts with 'words', 'start', 'end')
        path:       Output .ass file path
        style:      1-9 (see STYLE_NAMES)
        clip_start: Start offset of the clip in the source video (seconds)
        clip_end:   End offset (seconds)
        max_words:  Max words per subtitle line (default 5)
        gemini_fn:  Optional callable(prompt)->str for Style 9 emoji injection

    Returns:
        Number of subtitle events written
    """
    lines = _segment_words(
        segments, max_words=max_words, clip_start=clip_start, clip_end=clip_end
    )
    if not lines:
        log.warning("write_ass: no subtitle lines generated for %s", path)
        return 0

    renderers = {
        1: lambda: _style1_tiktok_yellow(lines, path),
        2: lambda: _style2_white_bold(lines, path),
        3: lambda: _style3_neon_green(lines, path),
        4: lambda: _style4_cinematic(lines, path),
        5: lambda: _style5_word_pop(lines, path),
        6: lambda: _style6_gradient_wave(lines, path),
        7: lambda: _style7_outlined_clean(lines, path),
        8: lambda: _style8_minimal_lowercase(lines, path),
        9: lambda: _style9_emoji_inject(lines, path, gemini_fn),
    }

    renderer = renderers.get(style)
    if renderer is None:
        log.warning("write_ass: unknown style %d, falling back to Style 1", style)
        renderer = renderers[1]

    count = renderer()
    log.info(
        "ASS subtitles written: %s (%d events, style=%d %s)",
        path,
        count,
        style,
        STYLE_NAMES.get(style, "?"),
    )
    return count


# ─────────────────────────────────────────────
# STYLE PREVIEW TEST  (python -m modules.overlays)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    DUMMY_SEGMENTS = [
        {
            "start": 0.0,
            "end": 4.0,
            "words": [
                {"word": "This", "start": 0.0, "end": 0.4},
                {"word": "is", "start": 0.4, "end": 0.6},
                {"word": "how", "start": 0.6, "end": 0.9},
                {"word": "you", "start": 0.9, "end": 1.1},
                {"word": "build", "start": 1.1, "end": 1.5},
                {"word": "money", "start": 1.5, "end": 2.0},
                {"word": "and", "start": 2.0, "end": 2.2},
                {"word": "grow", "start": 2.2, "end": 2.6},
                {"word": "fast", "start": 2.6, "end": 3.0},
                {"word": "every", "start": 3.0, "end": 3.3},
                {"word": "single", "start": 3.3, "end": 3.6},
                {"word": "day", "start": 3.6, "end": 4.0},
            ],
        }
    ]

    td = tempfile.mkdtemp()
    passed = 0
    for s in range(1, 10):
        out_path = os.path.join(td, f"style_{s}.ass")
        n = write_ass(DUMMY_SEGMENTS, out_path, style=s, clip_end=4.0)
        size = os.path.getsize(out_path)
        ok = n > 0 and size > 100
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        print(
            f"  Style {s} ({STYLE_NAMES[s]:22s}): {n:3d} events  {size:5d}B  {status}"
        )

    print(f"\n{passed}/9 styles passed")
    # Test: Style 9 emoji map coverage
    hits = sum(1 for kw in _EMOJI_MAP if _pick_emoji(kw) != "")
    print(
        f"Emoji map: {hits}/{len(_EMOJI_MAP)} keywords resolve correctly {'✅' if hits == len(_EMOJI_MAP) else '❌'}"
    )
