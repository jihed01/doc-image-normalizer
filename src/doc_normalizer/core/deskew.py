import cv2
import numpy as np


# Estimate the document skew angle using a reduced-resolution image.
def estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate the document skew angle in degrees."""
    # Convert to grayscale if the input image is in color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Reduce the image size to accelerate angle estimation.
    height, width = gray.shape[:2]
    max_dimension = 1000

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        gray = cv2.resize(
            gray,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    # Apply Gaussian blur and inverse Otsu thresholding.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    # Extract the coordinates of foreground pixels.
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return 0.0

    # Compute the minimum-area bounding rectangle.
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize the OpenCV angle to the expected range.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return float(-angle)


# Correct image skew while preserving the original image dimensions.
def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct image skew and return the corrected image and estimated skew angle."""
    angle = estimate_skew_angle(image)

    # Avoid unnecessary rotation for negligible skew.
    if abs(angle) < 0.1:
        return image.copy(), 0.0

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Build the rotation matrix to compensate for the detected skew.
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Apply the affine transformation while preserving the original dimensions.
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected, angle
