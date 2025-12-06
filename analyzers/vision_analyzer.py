# analyzers/vision_analyzer.py
"""Apple Vision framework analysis for scene/object detection."""

import Vision
from Quartz import CIImage
from Foundation import NSURL
import tempfile
from pathlib import Path
from PIL import Image


def analyze_image_file(image_path: str) -> dict:
    """
    Use Apple Vision to analyze image content.

    Args:
        image_path: Path to the image file

    Returns:
        Dict with 'objects', 'animals', 'text', 'faces_detected'
    """
    results = {
        "objects": [],
        "animals": [],
        "text": [],
        "faces_detected": 0
    }

    try:
        # Load image
        url = NSURL.fileURLWithPath_(image_path)
        ci_image = CIImage.imageWithContentsOfURL_(url)

        if ci_image is None:
            print(f"Failed to load image: {image_path}")
            return results

        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )

        # Object/Scene Classification
        classify_request = Vision.VNClassifyImageRequest.alloc().init()
        success, error = handler.performRequests_error_([classify_request], None)

        if success and classify_request.results():
            for observation in classify_request.results():
                confidence = observation.confidence()
                if confidence > 0.8:  # Require high confidence
                    results["objects"].append({
                        "label": observation.identifier(),
                        "confidence": float(confidence)
                    })

        # Sort by confidence and limit
        results["objects"].sort(key=lambda x: x["confidence"], reverse=True)

        # Animal Detection
        animal_request = Vision.VNRecognizeAnimalsRequest.alloc().init()
        success, error = handler.performRequests_error_([animal_request], None)

        if success and animal_request.results():
            for observation in animal_request.results():
                if observation.labels():
                    for label in observation.labels():
                        results["animals"].append(label.identifier())

        # Face Detection (count only - recognition is handled by face_recognition lib)
        face_request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        success, error = handler.performRequests_error_([face_request], None)

        if success and face_request.results():
            results["faces_detected"] = len(face_request.results())

        # Text Detection (OCR)
        text_request = Vision.VNRecognizeTextRequest.alloc().init()
        success, error = handler.performRequests_error_([text_request], None)

        if success and text_request.results():
            for observation in text_request.results():
                if observation.confidence() > 0.5:
                    text = observation.topCandidates_(1)
                    if text and len(text) > 0:
                        results["text"].append(text[0].string())

    except Exception as e:
        print(f"Vision analysis error for {image_path}: {e}")

    return results


def analyze_pil_image(pil_image: Image.Image) -> dict:
    """
    Analyze a PIL Image using Vision framework.
    Saves to temp file first since Vision needs a file path.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name, "JPEG", quality=85)
        results = analyze_image_file(tmp.name)
        Path(tmp.name).unlink()  # Clean up temp file
    return results


# Mapping of Vision labels to human-friendly Finder tags
TAG_MAPPING = {
    # People/Age
    "baby": "Baby",
    "infant": "Baby",
    "child": "Kids",
    "teenager": "Kids",
    "person": None,  # Too generic, skip
    "people": None,

    # Animals
    "dog": "Dog",
    "cat": "Cat",
    "bird": "Bird",
    "pet": "Pet",

    # Locations/Scenes
    "beach": "Beach",
    "ocean": "Beach",
    "mountain": "Nature",
    "forest": "Nature",
    "park": "Outdoors",
    "outdoor": "Outdoors",
    "indoor": "Indoors",
    "home": "Home",
    "kitchen": "Home",
    "living_room": "Home",
    "bedroom": "Home",
    "bathroom": "Home",

    # Events
    "birthday_cake": "Birthday",
    "birthday": "Birthday",
    "cake": "Birthday",
    "christmas_tree": "Christmas",
    "christmas": "Christmas",
    "wedding": "Wedding",
    "party": "Party",
    "celebration": "Party",

    # Activities
    "swimming": "Swimming",
    "playing": "Playing",
    "eating": "Food",
    "food": "Food",
    "cooking": "Cooking",
    "sports": "Sports",
    "travel": "Travel",
    "vacation": "Vacation",
}


def map_to_tags(vision_results: dict, max_tags: int = 20) -> list[str]:
    """
    Convert Vision results to Finder-friendly tags.

    Args:
        vision_results: Output from analyze_image_file
        max_tags: Maximum number of tags to return

    Returns:
        List of tag strings
    """
    tags = set()

    # Pass through all object labels (cleaned up for readability)
    for obj in vision_results.get("objects", [])[:20]:  # Top 20 by confidence
        label = obj["label"]
        # Convert underscores to spaces and title case
        clean_label = label.replace("_", " ").title()
        # Skip overly generic labels
        if clean_label.lower() not in ("person", "people", "object", "thing"):
            tags.add(clean_label)

    # Add animal tags
    for animal in vision_results.get("animals", []):
        tags.add(animal.capitalize())

    return list(tags)[:max_tags]
