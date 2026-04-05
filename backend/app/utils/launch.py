"""
VeridisQuo API Launcher

Launch the FastAPI server for deepfake detection.

Usage:
    uv run python src/launch_api.py
    uv run python src/launch_api.py --port 8080
    uv run python src/launch_api.py --reload --debug
"""

import argparse
import uvicorn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch VeridisQuo API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    return parser.parse_args()


def main():
    """Launch the API server."""
    args = parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                   VeridisQuo API                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Host:        {args.host:<42} ║
    ║  Port:        {args.port:<42} ║
    ║  Workers:     {args.workers:<42} ║
    ║  Reload:      {str(args.reload):<42} ║
    ║  Debug:       {str(args.debug):<42} ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Docs:        http://{args.host}:{args.port}/docs{' ':<21} ║
    ║  Health:      http://{args.host}:{args.port}/api/v1/health{' ':<12} ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
