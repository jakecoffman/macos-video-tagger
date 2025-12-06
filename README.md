# Video Tagger

AI-powered video tagging for macOS Finder. Analyzes videos using Apple Vision and face recognition to automatically detect and tag content (objects, scenes, animals) and people.

Also has a web UI for browsing and editing tags. You can just watch the videos from there if you don't want to use Finder tags.

## Features

- **Content detection**: Uses Apple Vision to detect objects, scenes, and animals
- **Face recognition**: Detects and clusters faces across videos with support for multiple backends:
  - `dlib` (CPU, default)
  - `insightface` (ONNX)
  - `coreml` (Neural Engine - fastest on Apple Silicon)
  - `deepface` (ArcFace, Facenet512 - most accurate)
- **Finder tags**: Applies tags directly to video files for easy searching in Finder
- **Resumable processing**: Saves progress and can resume interrupted processing
- **Parallel processing**: Process multiple videos simultaneously
- **Web UI**: Browse and edit tags via a local web interface

## Requirements

- macOS (uses Apple Vision framework)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/jakecoffman/macos-video-tagger.git
cd video-tagger
uv sync
```

## Usage

### Basic usage (dry-run)

```bash
uv run python main.py ~/Videos
```

### Use CoreML backend (fastest on Apple Silicon)

```bash
uv run python main.py ~/Videos -b coreml --write-tags
```

### Use DeepFace backend (most accurate)

```bash
# Default uses ArcFace model
uv run python main.py ~/Videos -b deepface --write-tags

# Or try Facenet512 for potentially better accuracy
uv run python main.py ~/Videos -b deepface -m Facenet512 --write-tags
```

### Parallel processing

```bash
uv run python main.py ~/Videos -j 4 --write-tags
```

### Manage detected persons

```bash
# List all detected persons
uv run python main.py ~/Videos --list-persons

# Rename "Person 1" to a real name
uv run python main.py ~/Videos --rename-persons
```

### Web UI

```bash
uv run python web/server.py ~/Videos
```

Then open http://localhost:5001 in your browser.

### Applying Finder tags

Once you are satisfied with the detected tags and persons, you can apply the tags to the video files in Finder by adding the `--write-tags` option:

```bash
uv run python main.py ~/Videos --write-tags
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--write-tags, -w` | Apply Finder tags to video files (default is dry-run) |
| `--verbose, -v` | Show detailed progress |
| `--interval, -i` | Seconds between extracted frames (default: 3.0) |
| `--backend, -b` | Face recognition backend (see comparison below) |
| `--deepface-model, -m` | DeepFace model (see comparison below) |
| `--threshold, -t` | Face matching threshold (higher = more lenient, see below) |
| `--workers, -j` | Number of parallel workers |
| `--skip-processed` | Skip videos that have already been processed |
| `--fast` | Use faster but less accurate face detection (HOG vs CNN) |
| `--list-persons` | List all detected persons |
| `--rename-persons` | Interactively rename detected persons |

### Face Recognition Backends (`-b`)

| Backend | Speed | Hardware | Notes |
|---------|-------|----------|-------|
| `dlib` | Slow | CPU | Default. Reliable but CPU-intensive. Use `--fast` for HOG mode. |
| `insightface` | Medium | CPU (ONNX) | Good balance of speed and accuracy. |
| `coreml` | Fast | Neural Engine | Best for Apple Silicon. Uses InsightFace with CoreML acceleration. |
| `deepface` | Slow | CPU/GPU | Most accurate. Requires `-m` to select model. |

### DeepFace Models (`-m`, requires `-b deepface`)

| Model | Speed | Size | Best For |
|-------|-------|------|----------|
| `ArcFace` | Medium | ~500MB | Excellent accuracy across lighting/angles. |
| `Facenet512` | Medium | ~90MB | Cross-video matching. Great for identifying same person in different videos. |
| `Facenet` | Fast | ~90MB | Lighter version of Facenet512. Good speed/accuracy trade-off. |
| `VGG-Face` | Slow | ~500MB | Older model. Still solid but slower. |
| `SFace` | Fast | ~40MB | Lightweight and efficient. Good for large video libraries. |
| `GhostFaceNet` | Fast | ~20MB | Newest, most efficient. Good accuracy with minimal resources. |

### Tuning Face Matching (`-t`)

If too many persons are being created (same person split across multiple IDs), increase the threshold:

```bash
# More lenient matching (fewer persons, may merge different people)
uv run python main.py ~/Videos -b deepface -m Facenet512 -t 0.55

# Stricter matching (more persons, less likely to merge different people)
uv run python main.py ~/Videos -b deepface -m Facenet512 -t 0.35
```

Default thresholds by backend/model:
- `dlib`: 0.6
- `insightface`/`coreml`: 0.5
- `deepface` + `Facenet512`: 0.45
- `deepface` + `ArcFace`: 0.68

> **Tip:** When switching backends or models, delete `data/faces.db` first since embeddings from different models are not compatible.

## License

MIT
