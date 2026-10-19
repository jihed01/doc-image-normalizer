import cv2
import numpy as np


def denoise_and_binarize(image: np.ndarray) -> np.ndarray:
    """Reduce image noise, remove dark borders, and apply adaptive thresholding."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Input image must be grayscale or BGR.")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binarized = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        30,
    )

    height, width = binarized.shape
    border_width = max(1, min(height, width) // 25)

    top_border = gray[:border_width, :]
    bottom_border = gray[height - border_width :, :]
    left_border = gray[:, :border_width]
    right_border = gray[:, width - border_width :]

    if np.mean(top_border < 50) >= 0.5:
        binarized[:border_width, :] = 255

    if np.mean(bottom_border < 50) >= 0.5:
        binarized[height - border_width :, :] = 255

    if np.mean(left_border < 50) >= 0.5:
        binarized[:, :border_width] = 255

    if np.mean(right_border < 50) >= 0.5:
        binarized[:, width - border_width :] = 255

    return binarized
