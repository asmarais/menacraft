"""
VeridisQuo API Package

REST API for deepfake detection using the VeridisQuo inference engine.

Usage:
    # Start the server
    ./scripts/run_api.sh

    # Or directly with uvicorn
    cd src && uv run uvicorn api.app:app --reload

    # API will be available at:
    # - http://localhost:8000/docs (Swagger UI)
    # - http://localhost:8000/redoc (ReDoc)
    # - http://localhost:8000/api/v1/health (Health check)
    # - POST http://localhost:8000/api/v1/analyze (Video analysis)
"""

from api.app import create_app
from api.config import APISettings, get_settings

__all__ = ["create_app", "APISettings", "get_settings"]
