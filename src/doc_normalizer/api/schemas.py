from pydantic import BaseModel, Field


class NormalizationResponse(BaseModel):
    """Response schema for document image normalization."""

    detected_angle: float = Field(
        ...,
        description="Detected and corrected skew angle in degrees.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds.",
    )
    image_base64: str = Field(
        ...,
        description="Deskewed and binarized image encoded in Base64.",
    )
