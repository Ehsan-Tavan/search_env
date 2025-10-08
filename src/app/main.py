import argparse
import uvicorn
from fastapi import FastAPI

from src.configurations import Config
from .routers import search, root


def create_app(config):
    app = FastAPI(title="Search API", version="1.0.0")

    # Attach config to app state
    app.state.config = config

    # Include routers
    app.include_router(root.router)
    app.include_router(search.router)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Environment")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config YAML")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=5250)

