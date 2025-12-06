#!/usr/bin/env python3
"""
Video Tagger - AI-powered video tagging for macOS Finder.

Scans videos in a directory, analyzes them using Apple Vision and face recognition,
and applies Finder tags for easy sorting.
"""

import argparse
import os
import sys
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from urllib.parse import quote


def file_url(path: Path) -> str:
    """Convert a path to a clickable file:// URL."""
    return f"file://{quote(str(path.resolve()))}"

from config import (
    FACES_DB_PATH,
    FRAME_INTERVAL_SECONDS,
    VIDEO_EXTENSIONS,
    DATA_DIR,
    FACE_MODEL,
)
from extractors.frames import extract_frames, get_video_duration
from analyzers.vision_analyzer import analyze_pil_image, map_to_tags
from analyzers.face_analyzer import FaceAnalyzer, BACKEND_DLIB, BACKEND_INSIGHTFACE, BACKEND_COREML
from storage.database import FaceDatabase
from storage.finder_tags import add_tags, get_current_tags


def find_videos(directory: Path) -> list[Path]:
    """Find all video files in a directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.glob(f"*{ext}"))
        videos.extend(directory.glob(f"**/*{ext}"))  # Recursive
    return sorted(set(videos))


def process_video(
    video_path: Path,
    face_analyzer: FaceAnalyzer,
    db: FaceDatabase,
    frame_interval: float = FRAME_INTERVAL_SECONDS,
    verbose: bool = False
) -> tuple[list[str], list[str]]:
    """
    Process a single video and return detected tags.
    Supports resuming from where it left off if interrupted.

    Args:
        video_path: Path to the video file
        face_analyzer: Face analyzer instance
        db: Database for progress tracking
        frame_interval: Seconds between extracted frames
        verbose: Print detailed progress

    Returns:
        Tuple of (content_tags, person_tags)
    """
    video_path_str = str(video_path)

    # Check for existing progress
    progress = db.get_video_progress(video_path_str)
    start_frame = 0
    all_content_tags = set()
    all_people = set()

    if progress:
        total_frames, processed_frames, existing_content, existing_people = progress
        if processed_frames > 0:
            start_frame = processed_frames
            all_content_tags = set(existing_content)
            all_people = set(existing_people)
            if verbose:
                print(f"    Resuming from frame {start_frame}...")

    # Extract frames
    if verbose:
        duration = get_video_duration(video_path_str)
        print(f"    Duration: {duration:.1f}s, extracting frames every {frame_interval}s...")

    frames = extract_frames(video_path_str, interval_seconds=frame_interval)

    if not frames:
        print(f"    Warning: No frames extracted from {video_path.name}")
        return [], []

    # Initialize progress tracking if new
    if not progress:
        db.start_video_progress(video_path_str, len(frames))

    if verbose:
        if start_frame > 0:
            print(f"    Processing frames {start_frame + 1}-{len(frames)} of {len(frames)}...")
        else:
            print(f"    Extracted {len(frames)} frames, analyzing...")

    for i, frame in enumerate(frames):
        # Skip already-processed frames
        if i < start_frame:
            continue

        # Vision analysis (objects, scenes, animals)
        vision_results = analyze_pil_image(frame)
        content_tags = map_to_tags(vision_results)
        all_content_tags.update(content_tags)

        # Face recognition
        people = face_analyzer.analyze_pil_image(frame, video_path_str)
        all_people.update(people)

        # Save progress after each frame (allows resume on Ctrl+C)
        db.update_video_progress(
            video_path_str,
            i + 1,
            list(all_content_tags),
            list(all_people)
        )

        if verbose and (i + 1) % 5 == 0:
            print(f"    Processed {i + 1}/{len(frames)} frames...")

    # Mark as complete (removes from progress table)
    db.complete_video_progress(video_path_str)

    return list(all_content_tags), list(all_people)


# Global variables for worker processes (initialized once per worker)
_worker_db = None
_worker_face_analyzer = None
_worker_config = None


def _init_worker(db_path: str, face_model: str, face_threshold: float, face_backend: str):
    """Initialize resources for a worker process."""
    global _worker_db, _worker_face_analyzer, _worker_config
    _worker_db = FaceDatabase(db_path)
    _worker_face_analyzer = FaceAnalyzer(
        _worker_db,
        threshold=face_threshold,
        model=face_model,
        backend=face_backend
    )
    _worker_config = {"db_path": db_path, "face_model": face_model, "face_backend": face_backend}


def _process_video_worker(args: tuple) -> dict:
    """
    Worker function for parallel video processing.

    Args:
        args: Tuple of (video_path, frame_interval, verbose, worker_id, total_workers)

    Returns:
        Dict with video_path, content_tags, person_tags, success, error
    """
    video_path, frame_interval, verbose = args
    video_path = Path(video_path)

    result = {
        "video_path": str(video_path),
        "content_tags": [],
        "person_tags": [],
        "success": False,
        "error": None
    }

    try:
        prefix = f"[{os.getpid()}]" if verbose else ""
        if verbose:
            print(f"{prefix} Processing: {file_url(video_path)}")

        content_tags, person_tags = process_video(
            video_path,
            _worker_face_analyzer,
            _worker_db,
            frame_interval=frame_interval,
            verbose=verbose
        )

        result["content_tags"] = content_tags
        result["person_tags"] = person_tags
        result["success"] = True

        if verbose:
            print(f"{prefix} Completed: {video_path.name}")
            print(f"{prefix}   Content: {content_tags}")
            print(f"{prefix}   People: {person_tags}")

    except Exception as e:
        result["error"] = str(e)
        print(f"Error processing {video_path.name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tag videos with AI-detected content and faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/Videos                    # Process with dlib (CPU, default)
  %(prog)s ~/Videos -b insightface     # Use InsightFace + ONNX
  %(prog)s ~/Videos -b coreml          # Use InsightFace + CoreML (Neural Engine)
  %(prog)s ~/Videos -j 4               # Process with 4 parallel workers
  %(prog)s ~/Videos --write-tags       # Apply Finder tags to files
  %(prog)s . --rename-persons          # Rename detected persons
        """
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing videos to process"
    )
    parser.add_argument(
        "--write-tags", "-w",
        action="store_true",
        help="Actually write Finder tags to video files (default is dry-run)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=FRAME_INTERVAL_SECONDS,
        help=f"Seconds between extracted frames (default: {FRAME_INTERVAL_SECONDS})"
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip videos that have already been processed"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster but less accurate face detection (dlib HOG instead of CNN)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["dlib", "insightface", "coreml"],
        default="dlib",
        help="Face recognition backend: dlib (CPU, default), insightface (ONNX), coreml (Neural Engine)"
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        help=f"Number of parallel workers (default: 1, max recommended: {cpu_count()})"
    )
    parser.add_argument(
        "--rename-persons",
        action="store_true",
        help="Interactive mode to rename detected persons"
    )
    parser.add_argument(
        "--list-persons",
        action="store_true",
        help="List all detected persons and exit"
    )

    args = parser.parse_args()

    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: Not a directory: {args.directory}")
        sys.exit(1)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database and face analyzer
    # Use CNN model by default (better for babies/kids), HOG if --fast is specified
    face_model = "hog" if args.fast else FACE_MODEL

    # Map backend argument to constant
    backend_map = {
        "dlib": BACKEND_DLIB,
        "insightface": BACKEND_INSIGHTFACE,
        "coreml": BACKEND_COREML,
    }
    face_backend = backend_map[args.backend]

    db = FaceDatabase(FACES_DB_PATH)
    face_analyzer = FaceAnalyzer(db, model=face_model, backend=face_backend)

    if args.verbose:
        backend_desc = {
            "dlib": "dlib/face_recognition (CPU)",
            "insightface": "InsightFace + ONNX (CPU)",
            "coreml": "InsightFace + CoreML (Neural Engine)",
        }
        model_desc = 'faster HOG' if face_model == 'hog' else 'accurate CNN'
        print(f"Backend: {backend_desc[args.backend]}")
        if args.backend == "dlib":
            print(f"Model: {model_desc}")
        print()

    # Handle special modes
    if args.list_persons:
        persons = db.get_all_persons()
        if not persons:
            print("No persons detected yet.")
        else:
            print("Detected persons:")
            for person_id, name in persons:
                print(f"  {person_id}: {name}")
        db.close()
        return

    if args.rename_persons:
        persons = db.get_all_persons()
        if not persons:
            print("No persons detected yet. Process some videos first.")
        else:
            print("Rename detected persons (press Enter to skip):\n")
            for person_id, name in persons:
                new_name = input(f"  Rename '{name}' to: ").strip()
                if new_name:
                    db.rename_person(person_id, new_name)
                    print(f"    ✓ Renamed to '{new_name}'")
        db.close()
        return

    # Find videos
    videos = find_videos(args.directory)

    if not videos:
        print(f"No video files found in {args.directory}")
        print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        db.close()
        return

    # Check for incomplete/interrupted videos
    incomplete = db.get_incomplete_videos()
    if incomplete:
        print(f"Found {len(incomplete)} interrupted video(s) to resume:")
        for path, total, done in incomplete:
            print(f"  - {Path(path).name}: {done}/{total} frames")
        print()

    print(f"Found {len(videos)} video(s) in {args.directory}\n")

    # Filter out already-processed videos if requested
    videos_to_process = []
    skipped = 0
    for video in videos:
        if args.skip_processed and db.is_video_processed(str(video)):
            if args.verbose:
                print(f"Skipping (already processed): {video.name}")
            skipped += 1
        else:
            videos_to_process.append(video)

    if not videos_to_process:
        print("All videos already processed!")
        db.close()
        return

    # Determine number of workers
    num_workers = min(args.workers, len(videos_to_process), cpu_count())

    if num_workers > 1:
        print(f"Processing {len(videos_to_process)} video(s) with {num_workers} parallel workers...\n")

        # Close main DB connection before forking (workers will create their own)
        db.close()

        # Prepare worker arguments
        worker_args = [
            (str(video), args.interval, args.verbose)
            for video in videos_to_process
        ]

        # Process in parallel
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(str(FACES_DB_PATH), face_model, 0.6, face_backend)
        ) as pool:
            results = pool.map(_process_video_worker, worker_args)

        # Reopen DB to save final tags and show summary
        db = FaceDatabase(FACES_DB_PATH)

        processed = 0
        for result in results:
            if result["success"]:
                video_path = result["video_path"]
                content_tags = result["content_tags"]
                person_tags = result["person_tags"]
                all_tags = content_tags + person_tags

                if all_tags:
                    # Save to database (may already be saved by worker, but ensure it's there)
                    db.save_video_tags(video_path, content_tags, person_tags)

                    if args.write_tags:
                        add_tags(video_path, all_tags)

                processed += 1
            else:
                print(f"Failed: {result['video_path']}: {result['error']}")

    else:
        # Single-threaded processing (original behavior)
        processed = 0

        for video in videos_to_process:
            print(f"Processing: {file_url(video)}")

            try:
                content_tags, person_tags = process_video(
                    video,
                    face_analyzer,
                    db,
                    frame_interval=args.interval,
                    verbose=args.verbose
                )

                all_tags = content_tags + person_tags

                if all_tags:
                    print(f"  Content tags: {content_tags}")
                    print(f"  People: {person_tags}")

                    # Always save to database
                    db.save_video_tags(str(video), content_tags, person_tags)

                    if args.write_tags:
                        add_tags(str(video), all_tags)
                        print(f"  ✓ Applied {len(all_tags)} tags to file")
                    else:
                        print(f"  ✓ Saved to database (use --write-tags to apply Finder tags)")
                else:
                    print(f"  No tags detected")

                processed += 1
                print()

            except Exception as e:
                print(f"  Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                print()

    # Summary
    print("-" * 40)
    print(f"Done! Processed {processed} video(s)")
    if skipped:
        print(f"Skipped {skipped} already-processed video(s)")

    # Show all detected persons
    persons = db.get_all_persons()
    if persons:
        print(f"\nDetected {len(persons)} person(s):")
        for _, name in persons:
            print(f"  - {name}")
        print(f"\nRun with --rename-persons to give them real names.")

    db.close()


if __name__ == "__main__":
    main()
