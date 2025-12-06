# config.py
"""Configuration settings for video tagger."""

from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
FACES_DB_PATH = DATA_DIR / "faces.db"

# Frame extraction
FRAME_INTERVAL_SECONDS = 3.0  # Extract 1 frame every N seconds

# Face recognition
FACE_MATCH_THRESHOLD = 0.6  # Lower = stricter matching (0.6 is typical)
FACE_MODEL = "cnn"  # "cnn" = more accurate (better for babies), "hog" = faster

# Vision analysis
VISION_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
MAX_TAGS_PER_CATEGORY = 5  # Limit tags to avoid clutter

# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}
