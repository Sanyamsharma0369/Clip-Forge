# ClipForge

> Turn long videos into short, high-signal clips with a local AI pipeline and a live dashboard.

ClipForge is a local-first Python app that downloads a source video, transcribes it, asks an AI model to find the most viral-worthy moments, cuts those moments into short clips, and lets you monitor the full run from a browser dashboard.

## Why It Stands Out

- Local-first workflow: your pipeline runs on your own machine
- End-to-end clipping flow: download, transcribe, analyze, cut
- Live dashboard: monitor every stage with logs and progress updates
- Flexible AI stack: use Ollama locally or Gemini via API
- Creator-focused output: optimized for Shorts, Reels, and highlight clips

## Features

- YouTube or local video input
- Faster-Whisper transcription
- AI clip candidate selection
- FFmpeg clip rendering
- Browser dashboard with live run monitoring
- Stage stepper and progress updates
- Terminal and log visibility during long jobs
- Optional clip enhancement workflow

## How It Works

1. Download a source video or open a local file.
2. Transcribe the full video into timed segments.
3. Send transcript chunks to Ollama or Gemini.
4. Extract the best clip candidates.
5. Cut and save short clips.
6. Review results from the local dashboard.

## Tech Stack

- Python
- FFmpeg
- yt-dlp
- faster-whisper
- Ollama
- Plain HTML, CSS, and JavaScript dashboard

## Project Structure

- `app.py`: local dashboard server
- `pipeline.py`: full clip generation pipeline
- `dashboard.html`: control room UI
- `fix_quality.py`: enhancement workflow
- `setup_check.py`: environment diagnostics

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Sanyamsharma0369/Clip-Forge.git
cd Clip-Forge
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install system tools

You should have:

- Python 3.10+
- FFmpeg on `PATH`
- Ollama installed if you want local model inference

### 4. Start Ollama

```bash
ollama serve
```

You can also pull a model such as:

```bash
ollama pull mistral
```

### 5. Start the dashboard

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:4173
```

## Usage

### Dashboard mode

Run the local control room:

```bash
python app.py
```

Paste a YouTube URL or local file, launch the run, and track each stage from the browser UI.

### CLI mode

Run the pipeline directly:

```bash
python pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --model mistral --whisper base --clips 5 --min 30 --max 90
```

### Gemini mode

Set your API key:

```bash
set GEMINI_API_KEY=your_key_here
```

Then run:

```bash
python pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --gemini
```

## Example Use Cases

- Turn podcast episodes into social clips
- Extract Shorts from interviews or tutorials
- Make highlight reels from long-form commentary
- Explore local AI creator tooling without depending on a SaaS platform

## Demo Ideas For This Repo

If you want more GitHub stars, add these next:

- a dashboard screenshot near the top of the README
- a short GIF showing one full run
- a sample input and generated clips section
- a 30 to 60 second demo video pinned on X and LinkedIn

## Setup Notes

- Generated clips, logs, downloads, temp files, and local binaries are ignored from Git
- `GEMINI_API_KEY` is loaded from the environment
- Windows-specific binaries are intentionally not committed

## Roadmap

- Better clip scoring and ranking
- Smarter vertical framing
- One-click demo assets for the README
- Easier cross-platform setup
- Hosted demo or lightweight preview mode

## Contributing

Ideas, fixes, and feature suggestions are welcome. If you want to improve the pipeline, UI, or output quality, feel free to open an issue or pull request.

## Star This Project

If ClipForge helped you or gave you ideas for your own AI creator tools, consider starring the repo:

[https://github.com/Sanyamsharma0369/Clip-Forge](https://github.com/Sanyamsharma0369/Clip-Forge)
