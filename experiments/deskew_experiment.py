import time

import cv2
import numpy as np

from doc_normalizer.core.deskew import deskew_image

# Create a synthetic A4-sized image with a black rectangle.
image = np.full((3508, 2480, 3), 255, dtype=np.uint8)
cv2.rectangle(image, (500, 1200), (1980, 2200), (0, 0, 0), -1)

# Warm up the algorithm before measuring its latency.
for _ in range(3):
    deskew_image(image)

# Measure individual processing times.
durations_ms = []

for _ in range(30):
    start = time.perf_counter()
    deskew_image(image)
    durations_ms.append((time.perf_counter() - start) * 1000)

# Report median, p95, and maximum latency.
print(f"Median latency: {np.median(durations_ms):.2f} ms")
print(f"p95 latency: {np.percentile(durations_ms, 95):.2f} ms")
print(f"Maximum latency: {max(durations_ms):.2f} ms")
