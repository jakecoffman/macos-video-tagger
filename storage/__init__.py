# storage/__init__.py
"""Storage utilities for database and Finder tags."""

from .database import FaceDatabase
from .finder_tags import get_current_tags, set_tags, add_tags, remove_tags, clear_all_tags

__all__ = [
    "FaceDatabase",
    "get_current_tags",
    "set_tags",
    "add_tags",
    "remove_tags",
    "clear_all_tags"
]
