"""
API Configuration

Settings for the FastAPI server using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class APISettings(BaseSettings):
    """API server configuration settings."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # File handling
    upload_dir: Path = Field(
        default=Path("temp/uploads"),
        description="Directory for uploaded files"
    )
    output_dir: Path = Field(
        default=Path("temp/outputs"),
        description="Directory for output files (GradCAM videos)"
    )
    max_upload_size_mb: int = Field(
        default=500,
        description="Maximum upload file size in MB"
    )
    cleanup_after_response: bool = Field(
        default=True,
        description="Clean up uploaded files after processing"
    )

    # Inference settings
    inference_config_path: Optional[Path] = Field(
        default=None,
        description="Path to inference config file (JSON/YAML)"
    )
    default_fps: int = Field(
        default=1,
        description="Default frames per second for video analysis"
    )
    default_aggregation_method: str = Field(
        default="weighted_average",
        description="Default score aggregation method"
    )
    default_batch_size: int = Field(
        default=8,
        description="Default batch size for inference"
    )

    # GradCAM settings
    gradcam_alpha: float = Field(
        default=0.4,
        description="GradCAM overlay transparency"
    )

    # CORS settings
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    model_config = {
        "env_prefix": "VERIDISQUO_API_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> APISettings:
    """Get API settings singleton."""
    return APISettings()
