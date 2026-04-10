# ClipForge

ClipForge is a local-first video clipping tool that downloads a source video, transcribes it, finds high-signal moments with AI, cuts short clips, and monitors the whole run from a browser dashboard.

## What It Does

- Downloads local or YouTube video sources
- Transcribes audio with Whisper
- Uses Ollama or Gemini to identify viral clip candidates
- Cuts clips with FFmpeg
- Shows a local dashboard with live logs, stage progress, and recent outputs
- Supports optional quality enhancement for finished clips

## Project Files

- `app.py` runs the local dashboard server
- `pipeline.py` runs the clip generation pipeline
- `dashboard.html` is the dashboard UI
- `fix_quality.py` handles clip enhancement flows
- `setup_check.py` checks local dependencies

## Requirements

- Python 3.10+
- FFmpeg available on `PATH`
- Ollama installed locally if you want local AI analysis
- Optional NVIDIA/CUDA setup for faster transcription and encoding

## Python Dependencies

Install the Python packages with:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Install FFmpeg and make sure `ffmpeg` is on your `PATH`.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama if you want local LLM analysis:

```bash
ollama serve
```

4. Start the dashboard:

```bash
python app.py
```

5. Open `http://127.0.0.1:4173`

## Running The Pipeline Directly

```bash
python pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --model mistral --whisper base --clips 5 --min 30 --max 90
```

To use Gemini instead of Ollama, pass `--gemini` and set `GEMINI_API_KEY` before running.

## Notes

- Generated clips, logs, downloaded source files, and temp artifacts are intentionally ignored from Git.
- Large Windows-only binaries are also ignored so the repo stays lightweight and easier to clone.
- This repository does not include a license yet.
