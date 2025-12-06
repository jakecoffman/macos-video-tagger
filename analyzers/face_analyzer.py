# analyzers/face_analyzer.py
"""Face detection and recognition with multiple backend support."""

import numpy as np
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod

from storage.database import FaceDatabase


# Available backends
BACKEND_DLIB = "dlib"           # Original face_recognition library (CPU)
BACKEND_INSIGHTFACE = "insightface"  # InsightFace with ONNX (can use CoreML)
BACKEND_COREML = "coreml"       # Apple Vision + CoreML (Neural Engine)


class FaceBackend(ABC):
    """Abstract base class for face detection/recognition backends."""

    @abstractmethod
    def detect_and_encode(self, image_array: np.ndarray) -> list[np.ndarray]:
        """Detect faces and return their embeddings."""
        pass

    @abstractmethod
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute distance between two embeddings."""
        pass


class DlibBackend(FaceBackend):
    """Original face_recognition/dlib backend (CPU-based)."""

    def __init__(self, model: str = "cnn"):
        import face_recognition
        self.face_recognition = face_recognition
        self.model = model  # "cnn" or "hog"

    def detect_and_encode(self, image_array: np.ndarray) -> list[np.ndarray]:
        face_locations = self.face_recognition.face_locations(
            image_array,
            model=self.model,
            number_of_times_to_upsample=1
        )
        if not face_locations:
            return []

        encodings = self.face_recognition.face_encodings(
            image_array,
            face_locations,
            num_jitters=1
        )
        return encodings

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(np.linalg.norm(embedding1 - embedding2))


class InsightFaceBackend(FaceBackend):
    """InsightFace backend with ONNX Runtime (can use CoreML on M1)."""

    def __init__(self, use_coreml: bool = True):
        from insightface.app import FaceAnalysis

        # Configure providers - try CoreML first on macOS
        providers = []
        if use_coreml:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')

        # Initialize InsightFace with buffalo_l model (good accuracy)
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._use_coreml = use_coreml

        # Detection thresholds to reduce false positives
        self.det_score_threshold = 0.6  # Confidence (default is ~0.5)
        self.min_face_size = 40         # Minimum face width/height in pixels

    def detect_and_encode(self, image_array: np.ndarray) -> list[np.ndarray]:
        # InsightFace expects BGR, but we have RGB
        image_bgr = image_array[:, :, ::-1]

        faces = self.app.get(image_bgr)

        embeddings = []
        for face in faces:
            # Filter by detection confidence
            if hasattr(face, 'det_score') and face.det_score < self.det_score_threshold:
                continue

            # Filter by face size (ignore tiny faces - likely false positives)
            if hasattr(face, 'bbox'):
                bbox = face.bbox  # [x1, y1, x2, y2]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < self.min_face_size or height < self.min_face_size:
                    continue

            if face.embedding is not None:
                embeddings.append(face.embedding)

        return embeddings

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # InsightFace uses cosine similarity, convert to distance
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        # Convert similarity (1 = same, -1 = opposite) to distance (0 = same)
        return float(1 - similarity)


class CoreMLBackend(FaceBackend):
    """
    Apple Vision for detection + CoreML for embeddings.
    Uses the Neural Engine on M1/M2 Macs.
    """

    def __init__(self):
        import Vision
        from Quartz import CIImage
        from Foundation import NSData
        import coremltools as ct

        self.Vision = Vision
        self.CIImage = CIImage
        self.NSData = NSData

        # We'll use InsightFace's model converted to CoreML for embeddings
        # For now, fall back to a simpler approach using Vision's face detection
        # and InsightFace for embeddings (but with CoreML provider)
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Detection thresholds to reduce false positives
        self.det_score_threshold = 0.6  # Confidence (default is ~0.5)
        self.min_face_size = 40         # Minimum face width/height in pixels

    def detect_and_encode(self, image_array: np.ndarray) -> list[np.ndarray]:
        # Use InsightFace with CoreML provider
        image_bgr = image_array[:, :, ::-1]
        faces = self.app.get(image_bgr)

        embeddings = []
        for face in faces:
            # Filter by detection confidence
            if hasattr(face, 'det_score') and face.det_score < self.det_score_threshold:
                continue

            # Filter by face size (ignore tiny faces - likely false positives)
            if hasattr(face, 'bbox'):
                bbox = face.bbox  # [x1, y1, x2, y2]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < self.min_face_size or height < self.min_face_size:
                    continue

            if face.embedding is not None:
                embeddings.append(face.embedding)

        return embeddings

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(1 - similarity)


def create_backend(backend_name: str, model: str = "cnn") -> FaceBackend:
    """Factory function to create the appropriate backend."""
    if backend_name == BACKEND_DLIB:
        return DlibBackend(model=model)
    elif backend_name == BACKEND_INSIGHTFACE:
        return InsightFaceBackend(use_coreml=False)
    elif backend_name == BACKEND_COREML:
        # InsightFace with CoreML execution provider
        return InsightFaceBackend(use_coreml=True)
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use one of: {BACKEND_DLIB}, {BACKEND_INSIGHTFACE}, {BACKEND_COREML}")


class FaceAnalyzer:
    """Analyzes faces in images and matches them against known persons."""

    def __init__(
        self,
        db: FaceDatabase,
        threshold: float = 0.6,
        model: str = "cnn",
        backend: str = BACKEND_DLIB
    ):
        """
        Initialize the face analyzer.

        Args:
            db: Face database for storing/retrieving embeddings
            threshold: Distance threshold for face matching (lower = stricter)
            model: For dlib backend: "cnn" (more accurate) or "hog" (faster)
            backend: Which backend to use - "dlib", "insightface", or "coreml"
        """
        self.db = db
        self.threshold = threshold
        self.backend_name = backend

        # Adjust threshold for different backends (they have different scales)
        if backend in (BACKEND_INSIGHTFACE, BACKEND_COREML):
            # InsightFace uses cosine distance (0-2 scale, 0 = identical)
            # Typical threshold is 0.4-0.6 for cosine distance
            self.threshold = min(threshold, 0.5)  # Adjust if needed

        self._backend = create_backend(backend, model)
        self._known_faces_cache = None

    def _refresh_cache(self):
        """Refresh the known faces cache from database."""
        self._known_faces_cache = self.db.get_all_embeddings()

    def analyze_image_file(self, image_path: str, source_video: str) -> list[str]:
        """
        Detect and identify faces in an image file.

        Args:
            image_path: Path to the image file
            source_video: Path to the source video (for tracking)

        Returns:
            List of person names found in the image
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            return self._analyze_image_array(image_array, source_video)
        except Exception as e:
            print(f"Face analysis error for {image_path}: {e}")
            return []

    def analyze_pil_image(self, pil_image: Image.Image, source_video: str) -> list[str]:
        """
        Detect and identify faces in a PIL Image.

        Args:
            pil_image: PIL Image object
            source_video: Path to the source video (for tracking)

        Returns:
            List of person names found in the image
        """
        try:
            # Convert PIL to numpy array (RGB)
            image_array = np.array(pil_image.convert("RGB"))
            return self._analyze_image_array(image_array, source_video)
        except Exception as e:
            print(f"Face analysis error: {e}")
            return []

    def _analyze_image_array(self, image_array: np.ndarray, source_video: str) -> list[str]:
        """Internal method to analyze a numpy image array."""
        # Detect faces and get embeddings using the configured backend
        face_encodings = self._backend.detect_and_encode(image_array)

        if not face_encodings:
            return []

        # Refresh cache before matching
        self._refresh_cache()

        identified_people = []
        for encoding in face_encodings:
            person_name = self._match_or_create(encoding, source_video)
            if person_name not in identified_people:
                identified_people.append(person_name)

        return identified_people

    def _match_or_create(self, encoding: np.ndarray, source_video: str) -> str:
        """
        Match a face encoding to a known person or create a new person.

        Args:
            encoding: Face embedding vector
            source_video: Source video path for tracking

        Returns:
            Person name (existing or newly created)
        """
        known_faces = self._known_faces_cache or []

        if not known_faces:
            # First face ever - create Person 1
            person_id = self.db.add_person("Person 1")
            self.db.add_embedding(person_id, encoding, source_video)
            self._refresh_cache()
            return "Person 1"

        # Compare against all known faces
        distances = []
        for _, _, known_encoding in known_faces:
            dist = self._backend.compute_distance(known_encoding, encoding)
            distances.append(dist)

        distances = np.array(distances)
        best_match_idx = int(np.argmin(distances))
        best_distance = distances[best_match_idx]

        if best_distance < self.threshold:
            # Match found
            person_id, person_name, _ = known_faces[best_match_idx]
            # Optionally store this embedding too (improves future matching)
            # Only store if it's different enough from existing ones
            store_threshold = 0.3 if self.backend_name == BACKEND_DLIB else 0.15
            if best_distance > store_threshold:
                self.db.add_embedding(person_id, encoding, source_video)
            return person_name
        else:
            # New person
            existing_person_ids = set(f[0] for f in known_faces)
            new_person_num = len(existing_person_ids) + 1
            new_name = f"Person {new_person_num}"
            person_id = self.db.add_person(new_name)
            self.db.add_embedding(person_id, encoding, source_video)
            self._refresh_cache()
            return new_name

    def get_all_persons(self) -> list[tuple[int, str]]:
        """Get all known persons from database."""
        return self.db.get_all_persons()
