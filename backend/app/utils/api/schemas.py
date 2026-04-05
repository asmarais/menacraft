"""
API Request/Response Schemas

Pydantic models for API validation and serialization.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class AnalysisOptions(BaseModel):
    """Options for video analysis request."""

    frames_per_second: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Number of frames to sample per second"
    )
    aggregation_method: Literal[
        "majority",
        "average",
        "weighted_average",
        "max_confidence",
        "threshold"
    ] = Field(
        default="weighted_average",
        description="Method to aggregate frame predictions"
    )
    enable_gradcam: bool = Field(
        default=True,
        description="Generate GradCAM visualization video"
    )
    gradcam_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="GradCAM overlay transparency (0=transparent, 1=opaque)"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Batch size for inference"
    )


class FrameResult(BaseModel):
    """Result for a single frame."""

    prediction: str = Field(description="Predicted class (FAKE or REAL)")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction"
    )
    probabilities: dict[str, float] = Field(
        description="Probability for each class"
    )


class AggregationMetadata(BaseModel):
    """Metadata about score aggregation."""

    method: str = Field(description="Aggregation method used")
    frame_count: int = Field(description="Number of frames analyzed")
    fake_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of frames predicted as FAKE"
    )
    real_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of frames predicted as REAL"
    )
    average_fake_confidence: Optional[float] = Field(
        default=None,
        description="Average confidence for FAKE predictions"
    )
    average_real_confidence: Optional[float] = Field(
        default=None,
        description="Average confidence for REAL predictions"
    )


class AnalysisResponse(BaseModel):
    """Response for video analysis."""

    success: bool = Field(description="Whether analysis completed successfully")
    prediction: str = Field(description="Overall prediction (FAKE or REAL)")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    authenticity_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Authenticity score (100 = definitely real, 0 = definitely fake)"
    )
    num_frames_analyzed: int = Field(description="Number of frames analyzed")
    aggregation: AggregationMetadata = Field(
        description="Aggregation details"
    )
    frame_results: Optional[list[FrameResult]] = Field(
        default=None,
        description="Per-frame results (if requested)"
    )
    gradcam_video_url: Optional[str] = Field(
        default=None,
        description="URL to download GradCAM visualization video"
    )
    processing_time_seconds: float = Field(
        description="Total processing time in seconds"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    device: str = Field(description="Inference device (cpu/cuda)")
    version: str = Field(description="API version")
