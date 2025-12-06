# Video Tagger

AI-powered video tagging for macOS Finder. Analyzes videos using Apple Vision and face recognition to automatically detect and tag content (objects, scenes, animals) and people.

Also has a web UI for browsing and editing tags. You can just watch the videos from there if you don't want to use Finder tags.

## Features

- **Content detection**: Uses Apple Vision to detect objects, scenes, and animals
- **Face recognition**: Detects and clusters faces across videos with support for multiple backends:
  - `dlib` (CPU, default)
  - `insightface` (ONNX)
  - `coreml` (Neural Engine - fastest on Apple Silicon)
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
| `--backend, -b` | Face recognition backend: `dlib`, `insightface`, `coreml` |
| `--workers, -j` | Number of parallel workers |
| `--skip-processed` | Skip videos that have already been processed |
| `--fast` | Use faster but less accurate face detection (HOG vs CNN) |
| `--list-persons` | List all detected persons |
| `--rename-persons` | Interactively rename detected persons |

## License

MIT
