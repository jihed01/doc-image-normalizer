import base64
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from doc_normalizer.api.schemas import NormalizationResponse
from doc_normalizer.core.denoise import denoise_and_binarize
from doc_normalizer.core.deskew import deskew_image

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
}

app = FastAPI(
    title="Doc Image Normalizer API",
    version="0.1.0",
    description=("Document preprocessing and normalization microservice for OCR."),
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}


@app.post(
    "/v1/normalize",
    response_model=NormalizationResponse,
    status_code=status.HTTP_200_OK,
)
async def normalize_document(
    file: Annotated[UploadFile, File()],
) -> NormalizationResponse:
    """Normalize an uploaded document image and return its metadata."""
    start_time = time.perf_counter()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Format non supporte. Formats acceptes : PNG, JPG, JPEG et TIFF.",
        )

    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="La taille du fichier depasse la limite de 10 Mio.",
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsupported image file.",
        )

    corrected_image, angle = deskew_image(image)
    final_image = denoise_and_binarize(corrected_image)

    success, buffer = cv2.imencode(".png", final_image)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode the normalized image.",
        )

    image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    return NormalizationResponse(
        detected_angle=round(float(angle), 2),
        processing_time_ms=round(elapsed_time_ms, 2),
        image_base64=image_base64,
    )
