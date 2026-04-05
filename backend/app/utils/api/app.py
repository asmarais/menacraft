"""
FastAPI Application Factory

Creates and configures the VeridisQuo API application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import APISettings, get_settings
from api.routes import router
from api.services import InferenceService


def setup_logging(debug: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    logger = logging.getLogger("api.app")

    # Startup
    logger.info("Starting VeridisQuo API...")

    settings: APISettings = app.state.settings
    settings.ensure_directories()

    # Initialize inference service
    service = InferenceService(settings)
    service.initialize()
    app.state.inference_service = service

    logger.info(f"API ready - Device: {service.device}")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")

    yield

    # Shutdown
    logger.info("Shutting down VeridisQuo API...")


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters:
        settings: Optional API settings. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    setup_logging(settings.debug)

    app = FastAPI(
        title="VeridisQuo API",
        description=(
            "REST API for deepfake video detection.\n\n"
            "VeridisQuo analyzes videos using a neural network combining "
            "spatial features (EfficientNet-B4) with frequency-domain analysis "
            "(FFT and DCT) to detect manipulated content.\n\n"
            "**Features:**\n"
            "- Video authenticity scoring (0-100)\n"
            "- GradCAM visualization of suspicious areas\n"
            "- Multiple aggregation methods for frame-level predictions\n"
            "- Configurable analysis parameters"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    return app


# Default app instance for uvicorn
app = create_app()
