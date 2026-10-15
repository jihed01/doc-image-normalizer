import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_skewed_image() -> tuple[np.ndarray, float]:
    """Generate a synthetic image with a rectangle rotated by a known angle."""
    expected_angle = 5.0

    # Create a white 400x400 image.
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    # Draw a filled black rectangle representing a text block.
    cv2.rectangle(
        image,
        (100, 150),
        (300, 250),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )

    # Calculate the rotation matrix around the image center.
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        expected_angle,
        1.0,
    )

    # Rotate the image and fill the background with white.
    skewed_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderValue=(255, 255, 255),
    )

    return skewed_image, expected_angle
