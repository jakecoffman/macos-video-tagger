# Video Tagger Web UI (Rust)

A fast Rust implementation of the Video Tagger web server. Uses the same database and API as the Python version, but with significantly better performance for serving videos.

## Building

```bash
cd web-rust
cargo build --release
```

## Running

```bash
# Default: http://127.0.0.1:5000
./target/release/web-rust

# Custom host/port
./target/release/web-rust --host 0.0.0.0 --port 8080
```

Or use cargo:

```bash
cargo run --release
```

## Features

- Same API routes as Python version (`/api/videos`, `/api/persons`, etc.)
- Same HTML templates (shared)
- Much faster video serving thanks to Rust's async I/O
- Automatic database path detection (uses `../data/faces.db`)

## Why Rust?

The Python Flask server is single-threaded and can be slow when:
- Loading many video thumbnails simultaneously
- Serving large video files
- Handling multiple concurrent requests

The Rust version uses Tokio for async I/O and can handle many concurrent requests efficiently.
