"""Shared image encoding utilities for LLM clients."""

import base64
from pathlib import Path

# Mapping from file extension to MIME type for supported image formats.
SUPPORTED_IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ImageEncodingError(Exception):
    """Raised when an image file cannot be encoded to base64."""


def encode_image_to_base64(image_path: str | Path) -> tuple[str, str]:
    """
    Encode an image file to a base64 string with MIME type detection.

    Reads the file at *image_path*, validates that the format is supported,
    and returns the base64-encoded contents together with the corresponding
    MIME type string.

    Args:
        image_path (str | Path): Filesystem path to the image file.

    Returns:
        tuple[str, str]: A ``(base64_data, media_type)`` pair.

    Raises:
        ImageEncodingError: If the file does not exist or has an unsupported
            extension.
    """
    path = Path(image_path)

    if not path.exists():
        raise ImageEncodingError(f"Image file not found: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ImageEncodingError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_MIME_TYPES))}"
        )

    media_type = SUPPORTED_IMAGE_MIME_TYPES[suffix]

    with path.open("rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")

    return base64_data, media_type
