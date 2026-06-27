"""Resource HTTP Endpoint App — FastAPI-based HTTP server for Matrix resources.

Routes:
    GET /health                              — health check
    GET /resources                           — list all schemes and their hosts
    GET /resources/{scheme}                  — list hosts for scheme
    GET /resources/{scheme}/{host}           — list resources for host
    GET /resources/{scheme}/{host}/{path:path} — serve resource

This app does NOT register a Channel — its consumer is the browser, not the model.
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.configs import ConfigType, YamlConfigStore


class ResourceServerConfig(ConfigType):
    """Resource HTTP server configuration."""

    host: str = "127.0.0.1"
    port: int = 20880

    @classmethod
    def conf_name(cls) -> str:
        return "resource_server"


def create_app(matrix: Matrix) -> FastAPI:
    app = FastAPI(title="MOSS Resource Server", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    registry = matrix.resources()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/resources")
    async def list_schemes():
        result = {}
        for scheme in registry.schemes():
            result[scheme] = registry.hosts(scheme)
        return result

    @app.get("/resources/{scheme}")
    async def list_hosts(scheme: str):
        hosts = registry.hosts(scheme)
        if not hosts:
            return JSONResponse(
                {"error": f"No hosts for scheme: {scheme}"}, status_code=404
            )
        return {"scheme": scheme, "hosts": hosts}

    @app.get("/resources/{scheme}/{host}")
    async def list_resources(scheme: str, host: str):
        infos = await registry.list_infos(scheme, host=host)
        return [
            {
                "host": info.host,
                "path": info.path,
                "description": info.description,
            }
            for info in infos
        ]

    @app.get("/resources/{scheme}/{host}/{path:path}")
    async def get_resource(scheme: str, host: str, path: str):
        locator = f"{scheme}://{host}/{path}"
        item = await registry.get(locator)
        if item is None:
            return JSONResponse(
                {"error": f"Resource not found: {locator}"}, status_code=404
            )

        data = await item.get()

        if isinstance(data, (str, Path)):
            from pathlib import Path as _Path
            file_path = _Path(data) if isinstance(data, str) else data
            if not file_path.exists():
                return JSONResponse(
                    {"error": f"File not found on disk: {file_path}"}, status_code=404
                )
            return FileResponse(file_path)

        if isinstance(data, bytes):
            content_type = getattr(item.info, "content_type", None) or "application/octet-stream"
            return Response(content=data, media_type=content_type)

        try:
            from PIL.Image import Image as PILImage
        except ImportError:
            PILImage = None

        if PILImage is not None and isinstance(data, PILImage):
            import io
            buf = io.BytesIO()
            fmt = getattr(data, "format", None) or "PNG"
            data.save(buf, format=fmt)
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type=f"image/{fmt.lower()}")

        return {"locator": locator, "type": type(data).__name__}

    return app


async def main(matrix: Matrix):
    store = YamlConfigStore(matrix.cell_workspace.configs())
    conf = store.get_or_create(ResourceServerConfig(host="127.0.0.1", port=20880))

    app = create_app(matrix)

    config = uvicorn.Config(app, host=conf.host, port=conf.port, log_level="info")
    server = uvicorn.Server(config)

    matrix.logger.info("Resource HTTP server starting at http://%s:%d", conf.host, conf.port)
    matrix.logger.info("Resources endpoint: http://%s:%d/resources", conf.host, conf.port)

    try:
        await server.serve()
    finally:
        matrix.logger.info("Resource HTTP server stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
