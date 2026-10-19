import cv2
import numpy as np
import pytest

from doc_normalizer.core.denoise import denoise_and_binarize
from doc_normalizer.core.deskew import deskew_image

# ==================================================================================
#  DESKEWING
# ==================================================================================

# Test that the algorithm detects the skew angle accurately and corrects the image.


def test_deskew_image_accuracy(
    synthetic_skewed_image: tuple[np.ndarray, float],
) -> None:
    """Verify skew angle detection and image correction."""
    image, expected_angle = synthetic_skewed_image

    # Run the deskewing algorithm.
    corrected_image, detected_angle = deskew_image(image)

    # Verify that the detected angle is within 0.5 degrees of the expected angle.
    assert abs(detected_angle - expected_angle) < 0.5

    # Verify that the corrected image preserves the original dimensions.
    assert corrected_image.shape == image.shape

    # Verify that the corrected image differs from the input image.
    assert not np.array_equal(corrected_image, image)


# Test that the deskewing algorithm reduces the image's geometric skew.


def test_deskew_image_reduces_skew(
    synthetic_skewed_image: tuple[np.ndarray, float],
) -> None:
    """Verify that deskewing reduces the image's geometric skew."""
    image, _ = synthetic_skewed_image

    corrected_image, _ = deskew_image(image)

    # Measure the longest rectangle edge to determine its orientation.
    def get_rectangle_angle(input_image: np.ndarray) -> float:
        """Estimate the dominant rectangle orientation in degrees."""
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        # Extract foreground pixel coordinates in (x, y) order.
        points = np.column_stack(np.where(binary > 0)[::-1])
        rectangle = cv2.minAreaRect(points.astype(np.float32))
        corners = cv2.boxPoints(rectangle)

        # Find the longest edge of the rotated rectangle.
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        longest_edge = edges[np.argmax(lengths)]

        angle = float(np.degrees(np.arctan2(longest_edge[1], longest_edge[0])))

        # Normalize the orientation to the range [-90, 90].
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        return angle

    original_angle = get_rectangle_angle(image)
    corrected_angle = get_rectangle_angle(corrected_image)

    # Verify that the corrected image has less skew than the original image.
    assert abs(corrected_angle) < abs(original_angle)


# Test that an image without skew is returned without unnecessary rotation.


def test_deskew_image_without_skew() -> None:
    """Verify that an image without skew remains unchanged."""
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 150), (300, 250), (0, 0, 0), -1)

    corrected_image, detected_angle = deskew_image(image)

    # Verify that no rotation is detected.
    assert detected_angle == 0.0

    # Verify that the image remains unchanged.
    assert np.array_equal(corrected_image, image)


# Test that a blank image is handled without errors.


def test_deskew_image_blank() -> None:
    """Verify that a blank image is returned unchanged."""
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    corrected_image, detected_angle = deskew_image(image)

    # Verify that no skew is detected.
    assert detected_angle == 0.0

    # Verify that the output matches the input.
    assert np.array_equal(corrected_image, image)


# ==================================================================================
# DENOISE AND BINARIZE
# ==================================================================================

# Test denoise and binarize output shape and size


def test_denoise_and_binarize_output_shape_and_type() -> None:
    """Test output dimensions and binary pixel values."""

    noisy_image = np.random.randint(
        0,
        256,
        (200, 200),
        dtype=np.uint8,
    )

    processed_img = denoise_and_binarize(noisy_image)

    assert processed_img.shape == noisy_image.shape

    unique_values = np.unique(processed_img)
    assert set(unique_values).issubset({0, 255})


# Test denoise and binarize grayscale and BGR


def test_denoise_and_binarize_grayscale_and_bgr() -> None:
    """Test that grayscale and BGR inputs preserve image dimensions."""

    gray_image = np.full((200, 200), 255, dtype=np.uint8)
    cv2.putText(
        gray_image,
        "TEST",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        3,
    )
    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    gray_result = denoise_and_binarize(gray_image)
    bgr_result = denoise_and_binarize(bgr_image)

    assert gray_result.shape == gray_image.shape
    assert bgr_result.shape == gray_image.shape
    assert set(np.unique(gray_result)).issubset({0, 255})
    assert set(np.unique(bgr_result)).issubset({0, 255})


# Test denoise and binarize blank image


def test_denoise_and_binarize_blank_image() -> None:
    """Test that a blank white image remains white after processing."""

    image = np.full((200, 200), 255, dtype=np.uint8)

    result = denoise_and_binarize(image)

    assert result.shape == image.shape
    assert np.all(result == 255)


# Test denoise and binarize rejects invalid dimensioons


def test_denoise_and_binarize_rejects_invalid_dimensions() -> None:
    """Test that unsupported input dimensions raise a ValueError."""

    image = np.zeros((20, 20, 3, 1), dtype=np.uint8)

    with pytest.raises(ValueError):
        denoise_and_binarize(image)


#   test black borders


def test_denoise_and_binarize_removes_black_borders() -> None:
    """Test that black acquisition borders are removed."""

    image = np.full((200, 200), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (199, 199), 0, thickness=8)
    cv2.putText(
        image,
        "TEST",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        3,
    )

    result = denoise_and_binarize(image)

    assert result.shape == image.shape
    assert np.all(result[0, :] == 255)
    assert np.all(result[-1, :] == 255)
    assert np.all(result[:, 0] == 255)
    assert np.all(result[:, -1] == 255)


def test_denoise_and_binarize_reduces_noise() -> None:
    """Verify that processing reduces noise while preserving text."""
    clean_image = np.full((200, 200), 255, dtype=np.uint8)
    cv2.putText(
        clean_image,
        "TEST",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        3,
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 35, clean_image.shape)
    noisy_image = np.clip(
        clean_image.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    noisy_result = denoise_and_binarize(noisy_image)

    clean_background = clean_image == 255
    clean_text = clean_image < 128

    background_error_before = np.mean(noisy_image[clean_background] < 128)
    background_error_after = np.mean(noisy_result[clean_background] == 0)

    text_retention = np.mean(noisy_result[clean_text] == 0)

    assert background_error_after < background_error_before
    assert text_retention > 0.8
