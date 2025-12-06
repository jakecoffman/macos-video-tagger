# storage/database.py
"""SQLite database for storing face embeddings and video tags."""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime


class FaceDatabase:
    """Database for storing face embeddings and person information."""

    def __init__(self, db_path: str | Path):
        """
        Initialize the face database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source_video TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS video_tags (
                id INTEGER PRIMARY KEY,
                video_path TEXT NOT NULL UNIQUE,
                content_tags TEXT,
                person_tags TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_person
            ON face_embeddings(person_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_path
            ON video_tags(video_path)
        """)
        # Track progress for resumable processing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS video_progress (
                video_path TEXT PRIMARY KEY,
                total_frames INTEGER NOT NULL,
                processed_frames INTEGER NOT NULL DEFAULT 0,
                content_tags TEXT DEFAULT '',
                person_tags TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_person(self, name: str) -> int:
        """
        Add a new person to the database.

        Args:
            name: Person's name/label

        Returns:
            The new person's ID
        """
        cursor = self.conn.execute(
            "INSERT INTO persons (name) VALUES (?)", (name,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_embedding(self, person_id: int, embedding: np.ndarray, source_video: str):
        """
        Add a face embedding for a person.

        Args:
            person_id: ID of the person
            embedding: Face embedding vector (size varies by backend)
            source_video: Path to the video where face was found
        """
        # Store dtype info along with the embedding so we can reconstruct it correctly
        # Format: first byte is dtype indicator (0=float32, 1=float64), rest is embedding
        dtype_byte = b'\x00' if embedding.dtype == np.float32 else b'\x01'
        data = dtype_byte + embedding.astype(embedding.dtype).tobytes()

        self.conn.execute(
            "INSERT INTO face_embeddings (person_id, embedding, source_video) VALUES (?, ?, ?)",
            (person_id, data, source_video)
        )
        self.conn.commit()

    def get_all_embeddings(self) -> list[tuple[int, str, np.ndarray]]:
        """
        Get all face embeddings with person info.

        Returns:
            List of (person_id, person_name, embedding) tuples
        """
        rows = self.conn.execute("""
            SELECT p.id, p.name, f.embedding
            FROM face_embeddings f
            JOIN persons p ON f.person_id = p.id
        """).fetchall()

        result = []
        for row in rows:
            data = row[2]
            # Check if this is new format (with dtype byte) or old format
            # New format: first byte indicates dtype, old format: raw float64 bytes
            # Old format embeddings would be 128*8=1024 bytes (dlib)
            # New format would be 1 + 512*4=2049 bytes (insightface float32) or 1 + 128*8=1025 (dlib float64)
            if len(data) == 1024:
                # Old dlib format (128 float64, no dtype byte)
                embedding = np.frombuffer(data, dtype=np.float64)
            elif data[0:1] == b'\x00':
                # New format, float32
                embedding = np.frombuffer(data[1:], dtype=np.float32)
            elif data[0:1] == b'\x01':
                # New format, float64
                embedding = np.frombuffer(data[1:], dtype=np.float64)
            else:
                # Assume old format float64
                embedding = np.frombuffer(data, dtype=np.float64)

            result.append((row[0], row[1], embedding))

        return result

    def get_all_persons(self) -> list[tuple[int, str]]:
        """Get all persons (id, name) from database."""
        return self.conn.execute(
            "SELECT id, name FROM persons ORDER BY id"
        ).fetchall()

    def rename_person(self, person_id: int, new_name: str):
        """
        Rename a person.

        Args:
            person_id: ID of the person to rename
            new_name: New name for the person
        """
        self.conn.execute(
            "UPDATE persons SET name = ? WHERE id = ?",
            (new_name, person_id)
        )
        self.conn.commit()

    def is_video_processed(self, video_path: str) -> bool:
        """Check if a video has already been processed."""
        result = self.conn.execute(
            "SELECT 1 FROM video_tags WHERE video_path = ?",
            (video_path,)
        ).fetchone()
        return result is not None

    def save_video_tags(self, video_path: str, content_tags: list[str], person_tags: list[str]):
        """
        Save the tags for a processed video.

        Args:
            video_path: Path to the video
            content_tags: List of content-related tags
            person_tags: List of person-related tags
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO video_tags (video_path, content_tags, person_tags, processed_at)
            VALUES (?, ?, ?, ?)
        """, (video_path, ",".join(content_tags), ",".join(person_tags), datetime.now()))
        self.conn.commit()

    def get_video_tags(self, video_path: str) -> tuple[list[str], list[str]] | None:
        """
        Get saved tags for a video.

        Returns:
            Tuple of (content_tags, person_tags) or None if not found
        """
        row = self.conn.execute(
            "SELECT content_tags, person_tags FROM video_tags WHERE video_path = ?",
            (video_path,)
        ).fetchone()

        if row:
            content = row[0].split(",") if row[0] else []
            persons = row[1].split(",") if row[1] else []
            return (content, persons)
        return None

    def close(self):
        """Close the database connection."""
        self.conn.close()

    # --- Progress tracking for resumable processing ---

    def get_video_progress(self, video_path: str) -> tuple[int, int, list[str], list[str]] | None:
        """
        Get processing progress for a video.

        Returns:
            Tuple of (total_frames, processed_frames, content_tags, person_tags) or None
        """
        row = self.conn.execute(
            "SELECT total_frames, processed_frames, content_tags, person_tags FROM video_progress WHERE video_path = ?",
            (video_path,)
        ).fetchone()

        if row:
            content = row[2].split(",") if row[2] else []
            persons = row[3].split(",") if row[3] else []
            return (row[0], row[1], content, persons)
        return None

    def start_video_progress(self, video_path: str, total_frames: int):
        """Initialize progress tracking for a video."""
        self.conn.execute("""
            INSERT OR REPLACE INTO video_progress (video_path, total_frames, processed_frames, content_tags, person_tags, updated_at)
            VALUES (?, ?, 0, '', '', ?)
        """, (video_path, total_frames, datetime.now()))
        self.conn.commit()

    def update_video_progress(
        self,
        video_path: str,
        processed_frames: int,
        content_tags: list[str],
        person_tags: list[str]
    ):
        """Update progress for a video after processing frames."""
        self.conn.execute("""
            UPDATE video_progress
            SET processed_frames = ?, content_tags = ?, person_tags = ?, updated_at = ?
            WHERE video_path = ?
        """, (processed_frames, ",".join(content_tags), ",".join(person_tags), datetime.now(), video_path))
        self.conn.commit()

    def complete_video_progress(self, video_path: str):
        """Mark a video as fully processed and clean up progress tracking."""
        self.conn.execute("DELETE FROM video_progress WHERE video_path = ?", (video_path,))
        self.conn.commit()

    def get_incomplete_videos(self) -> list[tuple[str, int, int]]:
        """Get list of videos that were interrupted mid-processing.

        Returns:
            List of (video_path, total_frames, processed_frames)
        """
        return self.conn.execute(
            "SELECT video_path, total_frames, processed_frames FROM video_progress ORDER BY updated_at DESC"
        ).fetchall()
