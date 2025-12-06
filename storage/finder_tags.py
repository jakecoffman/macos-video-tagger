# storage/finder_tags.py
"""Apply Finder tags to files on macOS using xattr."""

import subprocess
import plistlib
from pathlib import Path


def get_current_tags(file_path: str) -> list[str]:
    """
    Get existing Finder tags for a file.

    Args:
        file_path: Path to the file

    Returns:
        List of current tag names
    """
    try:
        result = subprocess.run(
            ["xattr", "-px", "com.apple.metadata:_kMDItemUserTags", file_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Convert hex output to bytes
            hex_str = result.stdout.replace(" ", "").replace("\n", "")
            plist_bytes = bytes.fromhex(hex_str)
            plist_data = plistlib.loads(plist_bytes)
            # Tags may have color suffix like "tag\n6" - strip it
            return [tag.split("\n")[0] for tag in plist_data]
    except Exception as e:
        pass
    return []


def set_tags(file_path: str, tags: list[str]):
    """
    Set Finder tags for a file (replaces existing tags).

    Args:
        file_path: Path to the file
        tags: List of tag names to set
    """
    if not tags:
        # Remove tags if empty list
        try:
            subprocess.run(
                ["xattr", "-d", "com.apple.metadata:_kMDItemUserTags", file_path],
                capture_output=True
            )
        except:
            pass
        return

    # Convert tags to plist format (binary plist)
    plist_bytes = plistlib.dumps(tags, fmt=plistlib.FMT_BINARY)

    # Convert to hex string for xattr -wx
    hex_str = plist_bytes.hex()

    # Apply using xattr with hex format
    result = subprocess.run(
        ["xattr", "-wx", "com.apple.metadata:_kMDItemUserTags", hex_str, file_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error setting tags on {file_path}: {result.stderr}")


def add_tags(file_path: str, new_tags: list[str]):
    """
    Add tags to a file (preserving existing tags).

    Args:
        file_path: Path to the file
        new_tags: List of tag names to add
    """
    current = get_current_tags(file_path)
    combined = list(set(current + new_tags))
    set_tags(file_path, combined)


def remove_tags(file_path: str, tags_to_remove: list[str]):
    """
    Remove specific tags from a file.

    Args:
        file_path: Path to the file
        tags_to_remove: List of tag names to remove
    """
    current = get_current_tags(file_path)
    remaining = [t for t in current if t not in tags_to_remove]
    set_tags(file_path, remaining)


def clear_all_tags(file_path: str):
    """Remove all Finder tags from a file."""
    set_tags(file_path, [])
