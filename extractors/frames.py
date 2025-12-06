# extractors/frames.py
"""Extract frames from video files."""

import av
from PIL import Image


def extract_frames(video_path: str, interval_seconds: float = 2.0) -> list[Image.Image]:
    """
    Extract frames from video at specified interval.

    Args:
        video_path: Path to the video file
        interval_seconds: Extract one frame every N seconds

    Returns:
        List of PIL Image objects
    """
    frames = []

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Calculate frame interval based on FPS
        fps = float(stream.average_rate or stream.base_rate or 30)
        frame_interval = max(1, int(fps * interval_seconds))

        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_interval == 0:
                img = frame.to_image()
                frames.append(img)

        container.close()

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")

    return frames


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        container = av.open(video_path)
        duration = float(container.duration) / 1_000_000  # Convert from microseconds
        container.close()
        return duration
    except Exception:
        return 0.0
