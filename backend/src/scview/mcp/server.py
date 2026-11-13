"""scView's MCP server: an AI agent's tools, served at ``/mcp`` on the backend's own app.

Streamable HTTP (MCP SDK ``mcp`` 2.x, ``MCPServer``), mounted as one route on the
FastAPI app, so it shares the dataset cache, edit locks and memory budget with the
UI. ``McpEndpoint`` guards it before the SDK sees a request: 404 when
``MCP_ENABLED`` is off (default: on in private mode, off in public), the same
``ACCESS_TOKEN`` the ``/api`` routes take, and a Host / Origin check against DNS
rebinding (a web page in the user's browser must not reach a local scView's tools).
The SDK's own DNS-rebinding check is left off: this one is the one rule, the same
for every protocol version the SDK speaks.

The SDK's session manager runs inside the app's lifespan (``lifespan()``); a new one
each time, as one can run only once (tests start the app many times).
"""

from __future__ import annotations

import contextlib
import hmac
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ConfigDict
from starlette.types import Receive, Scope, Send

from scview.mcp import api, params, spatial, tools  # noqa: F401 - spatial registers its tools

INSTRUCTIONS = """\
scView: single-cell RNA-seq analysis running locally; these tools call the same routes as its browser UI.
- Start with list_datasets, then get_dataset (columns, embeddings, steps done) and assess_dataset.
- Before run_pipeline on a large dataset call preflight (memory and time) and memory_status.
- Long work is a job (run_pipeline, annotate_celltypist, milo, import_public_dataset, joint_embedding): job_id
  at once (spatial runs wait 20 s first); job_status(job_id, wait_seconds=50) until complete; cancel_job stops it.
- Conditions with donors: be donor-aware. Use differential_expression(method="pseudobulk") between
  conditions within a cell type, check a gene with per_donor_expression, test cell-type shares with
  composition_test (donors as replicates), not cell-level p values alone. Read each result's warnings.
- Name clusters or build a column (e.g. sample = donor AND day) with set_labels and a cell filter.
- enrichment takes a differential_expression result_id (ORA of up/down genes, or GSEA), a gene list,
  or a marker group. plot returns a PNG (embedding, violin, dotplot).
- Spatial (get_dataset lists the sections): plot(embedding="spatial") draws sections side by side, or one
  (section=); spatial_autocorr (Moran's I), spatial_domains, nhood_enrichment, cooccurrence, spatial_niches,
  spot_composition. Pool datasets (one file per slide) with joint_embedding_plan, then joint_embedding.
- Results and saved changes also show in the open browser UI (http://127.0.0.1:3000).
- The server runs in a container and cannot read host paths. To add a local file, upload it:
  curl -F "file=@/path/to/data.h5ad" http://127.0.0.1:8080/api/v1/datasets/upload
  (add -H "Authorization: Bearer <token>" when ACCESS_TOKEN is set), or use the UI's Data panel.
- Tables come back as the top rows with the total; ask for more with limit (max 200).
"""

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "backend"}  # backend: nginx in compose


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("scview")
    except PackageNotFoundError:
        return ""


def build() -> MCPServer:
    server = MCPServer(name="scview", title="scView", instructions=INSTRUCTIONS, version=_version())
    for fn, hints in tools.TOOLS:
        server.add_tool(fn, annotations=hints, structured_output=False)
    _strict(server)
    return server


def _strict(server: MCPServer) -> None:
    """Every tool refuses arguments it does not have (``additionalProperties: false``):
    the SDK's argument models ignore unknown names, which would make a misspelt
    argument a call with its default. ``params`` of the run tools gets the
    pipeline's own parameter schema (checked by the tool, with a hint)."""
    for t in server._tool_manager.list_tools():
        base = t.fn_metadata.arg_model
        strict = type(base.__name__, (base,), {
            "model_config": ConfigDict(**{**base.model_config, "extra": "forbid"}),
            "__module__": base.__module__})
        t.fn_metadata.arg_model = strict
        schema = strict.model_json_schema(by_alias=True)
        if t.name == "run_pipeline":
            pp = params.PipelineParamsInput.model_json_schema()
            schema["properties"]["params"] = {
                "anyOf": [{"type": "object", "properties": pp["properties"],
                           "additionalProperties": False}, {"type": "null"}],
                "default": None,
                "description": schema["properties"]["params"].get("description", ""),
            }
        t.parameters = schema


SERVER = build()


class _Holder:
    manager: StreamableHTTPSessionManager | None = None


def _origin_ok(origin: str, allowed: list[str]) -> bool:
    if origin.rstrip("/") in allowed:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and (parts.hostname or "") in _LOCAL_HOSTS


def _hostname(netloc: str) -> str:
    try:
        return urlsplit("//" + netloc).hostname or ""  # "[::1]:8080" -> "::1"
    except ValueError:
        return ""


def _host_ok(host: str, allowed: list[str]) -> bool:
    name = _hostname(host)
    return name in _LOCAL_HOSTS or name in {urlsplit(o).hostname for o in allowed}


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class McpEndpoint:
    """The ``/mcp`` route: the guards, then the SDK's Streamable HTTP handler."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        settings = api.settings()
        if not settings.mcp_enabled:
            await _reply(send, 404, "Not Found")
            return
        headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        allowed = settings.mcp_allowed_origins
        origin = headers.get("origin")
        if origin and not _origin_ok(origin, allowed):
            await _reply(send, 403, f"Origin '{origin}' is not allowed; add it to MCP_ALLOWED_ORIGINS.")
            return
        if not _host_ok(headers.get("host", ""), allowed):
            await _reply(send, 421, "Host not allowed; add the address to MCP_ALLOWED_ORIGINS.")
            return
        if settings.ACCESS_TOKEN:
            auth = headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-access-token", "")
            if not hmac.compare_digest(token.strip(), settings.ACCESS_TOKEN):
                await _reply(send, 401, "Missing or invalid access token.")
                return
        manager = _Holder.manager
        if manager is None:
            await _reply(send, 503, "The MCP server is starting; try again in a moment.")
            return
        await manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Run a fresh session manager for the app's lifetime."""
    manager = StreamableHTTPSessionManager(
        app=SERVER._lowlevel_server,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    async with manager.run():
        _Holder.manager = manager
        try:
            yield
        finally:
            _Holder.manager = None


def routes() -> list[Any]:
    from starlette.routing import Route

    endpoint = McpEndpoint()
    return [Route("/mcp", endpoint=endpoint, methods=["GET", "POST", "DELETE"]),
            Route("/mcp/", endpoint=endpoint, methods=["GET", "POST", "DELETE"])]
