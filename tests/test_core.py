import cv2
import numpy as np

from doc_normalizer.core.deskew import deskew_image


# ==================================================================================
# Test that the algorithm detects the skew angle accurately and corrects the image.
# ==================================================================================
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


# ==================================================================================
# Test that the deskewing algorithm reduces the image's geometric skew.
# ==================================================================================
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


# ==================================================================================
# Test that an image without skew is returned without unnecessary rotation.
# ==================================================================================
def test_deskew_image_without_skew() -> None:
    """Verify that an image without skew remains unchanged."""
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 150), (300, 250), (0, 0, 0), -1)

    corrected_image, detected_angle = deskew_image(image)

    # Verify that no rotation is detected.
    assert detected_angle == 0.0

    # Verify that the image remains unchanged.
    assert np.array_equal(corrected_image, image)


# ==================================================================================
# Test that a blank image is handled without errors.
# ==================================================================================
def test_deskew_image_blank() -> None:
    """Verify that a blank image is returned unchanged."""
    image = np.full((400, 400, 3), 255, dtype=np.uint8)

    corrected_image, detected_angle = deskew_image(image)

    # Verify that no skew is detected.
    assert detected_angle == 0.0

    # Verify that the output matches the input.
    assert np.array_equal(corrected_image, image)
