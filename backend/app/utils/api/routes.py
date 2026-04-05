"""
API Routes

FastAPI route definitions for the VeridisQuo API.
"""

import uuid
import aiofiles
from pathlib import Path
from logging import getLogger

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import (
    AnalysisOptions,
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
)

logger = getLogger("api.routes")

router = APIRouter()


def get_service(request: Request):
    """Get inference service from app state."""
    return request.app.state.inference_service


def get_settings(request: Request):
    """Get API settings from app state."""
    return request.app.state.settings


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """Check API health and model status."""
    service = get_service(request)
    return HealthResponse(
        status="healthy" if service.is_ready else "initializing",
        model_loaded=service.is_ready,
        device=service.device,
        version="1.0.0"
    )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Unsupported file format"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Analysis"],
    summary="Analyze video for deepfake detection"
)
async def analyze_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to analyze"),
    frames_per_second: int = Form(default=1, ge=1, le=30),
    aggregation_method: str = Form(default="weighted_average"),
    enable_gradcam: bool = Form(default=True),
    gradcam_alpha: float = Form(default=0.4, ge=0.0, le=1.0),
    batch_size: int = Form(default=8, ge=1, le=32),
    include_frame_results: bool = Form(default=False),
) -> AnalysisResponse:
    """
    Analyze a video file for deepfake detection.

    Upload a video and receive an authenticity score along with detection details.
    Optionally generates a GradCAM visualization video showing areas of concern.

    **Supported formats**: MP4, AVI, MOV, MKV, WEBM

    **Parameters**:
    - **video**: Video file to analyze
    - **frames_per_second**: Frames to sample per second (1-30)
    - **aggregation_method**: How to combine frame predictions
      - `majority`: Most common prediction wins
      - `average`: Average confidence scores
      - `weighted_average`: Confidence-weighted average (recommended)
      - `max_confidence`: Use highest confidence frame
      - `threshold`: Count frames above threshold
    - **enable_gradcam**: Generate visualization video showing detection areas
    - **gradcam_alpha**: Heatmap overlay transparency
    - **batch_size**: Processing batch size (affects memory usage)
    - **include_frame_results**: Include per-frame predictions in response

    **Returns**: Analysis results with authenticity score (0-100)
    """
    settings = get_settings(request)
    service = get_service(request)

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(video.filename).suffix.lower() if video.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file size
    video.file.seek(0, 2)  # Seek to end
    file_size = video.file.tell()
    video.file.seek(0)  # Reset to beginning

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / (1024*1024):.1f}MB. Maximum: {settings.max_upload_size_mb}MB"
        )

    # Create unique upload path
    upload_id = uuid.uuid4().hex[:12]
    upload_path = settings.upload_dir / f"{upload_id}{file_ext}"

    try:
        # Save uploaded file
        async with aiofiles.open(upload_path, "wb") as f:
            content = await video.read()
            await f.write(content)

        logger.info(f"Saved upload: {upload_path} ({file_size / (1024*1024):.1f}MB)")

        # Build analysis options
        options = AnalysisOptions(
            frames_per_second=frames_per_second,
            aggregation_method=aggregation_method,
            enable_gradcam=enable_gradcam,
            gradcam_alpha=gradcam_alpha,
            batch_size=batch_size,
        )

        # Run analysis
        response, gradcam_path = service.analyze_video(
            video_path=upload_path,
            options=options,
            include_frame_results=include_frame_results
        )

        # Set gradcam URL if available
        if gradcam_path and gradcam_path.exists():
            response.gradcam_video_url = f"/api/v1/outputs/{gradcam_path.name}"

        # Schedule cleanup
        if settings.cleanup_after_response:
            background_tasks.add_task(service.cleanup_file, upload_path)

        return response

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        # Cleanup on error
        if upload_path.exists():
            service.cleanup_file(upload_path)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/outputs/{filename}",
    tags=["Outputs"],
    summary="Download output file"
)
async def download_output(
    request: Request,
    filename: str,
) -> FileResponse:
    """
    Download a generated output file (e.g., GradCAM visualization video).

    Note: Files are NOT automatically cleaned up to support video streaming.
    Use DELETE /outputs/{filename} to manually clean up files.
    """
    settings = get_settings(request)

    file_path = settings.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Validate path is within output directory (prevent path traversal)
    try:
        file_path.resolve().relative_to(settings.output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Don't cleanup here - browsers make multiple Range requests for video streaming
    # Use DELETE endpoint to manually cleanup files

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )


@router.delete(
    "/outputs/{filename}",
    tags=["Outputs"],
    summary="Delete output file"
)
async def delete_output(
    request: Request,
    filename: str,
) -> JSONResponse:
    """Delete a generated output file."""
    settings = get_settings(request)
    service = get_service(request)

    file_path = settings.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Validate path
    try:
        file_path.resolve().relative_to(settings.output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    service.cleanup_file(file_path)

    return JSONResponse(
        status_code=200,
        content={"message": f"File {filename} deleted successfully"}
    )
