"""Long runs as jobs: a tool starts one and returns its id at once; ``job_status``
waits for it (up to 50 s a call) and ``cancel_job`` stops it.

A job reads a streaming route (SSE) in a task, keeping the latest progress line
and the final event. Jobs live in this process's memory (one uvicorn worker):
after a restart their ids are gone, and the run they drove has stopped with the
backend as a browser's run does (a pipeline run saves nothing unless finished).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from scview.mcp import api

KEEP = 50  # finished jobs kept for job_status
MAX_WAIT = 50.0


@dataclass
class Job:
    id: str
    kind: str
    dataset_id: str | None
    path: str
    body: dict[str, Any]
    progress_line: Callable[[str, Any], str | None]
    finish: Callable[[Job], Awaitable[dict[str, Any]] | dict[str, Any]]
    cancel_route: Callable[[Job], Awaitable[None]] | None = None
    status: str = "running"  # running | complete | stopped | error | cancelled
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    progress: str = "starting"
    events: dict[str, Any] = field(default_factory=dict)  # the last data of each event type
    error: str | None = None
    result: dict[str, Any] | None = None
    disconnect: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    cancel_requested: bool = False

    def seconds(self) -> float:
        return round((self.ended or time.monotonic()) - self.started, 1)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"job_id": self.id, "kind": self.kind, "status": self.status,
                               "dataset_id": self.dataset_id, "seconds": self.seconds()}
        if self.status == "running":
            out["progress"] = self.progress
        if self.error:
            out["error"] = self.error
        if self.result is not None:
            out["result"] = self.result
        return out


_jobs: OrderedDict[str, Job] = OrderedDict()


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def running(kind: str | None = None) -> list[Job]:
    return [j for j in _jobs.values() if j.status == "running" and (kind is None or j.kind == kind)]


async def _drive(job: Job) -> None:
    async def on_event(event: str, data: Any) -> None:
        job.events[event] = data
        if event == "error":
            job.error = str(data.get("error") if isinstance(data, dict) else data)
        line = job.progress_line(event, data)
        if line:
            job.progress = line

    try:
        await api.stream(job.path, job.body, on_event, job.disconnect)
        if job.error is None and job.cancel_requested:
            job.status = "cancelled"
        elif job.error is None and "cancelled" in job.events:
            job.status = "cancelled"
        elif job.error is not None:
            job.status = "cancelled" if job.cancel_requested else "error"
        else:
            out = job.finish(job)
            job.result = await out if asyncio.iscoroutine(out) else out
            if job.dataset_id is None and (job.result or {}).get("dataset_id"):
                job.dataset_id = job.result["dataset_id"]  # an import: the dataset it made
            status = (job.result or {}).get("status")
            # A pipeline run that stopped itself (nothing saved) is not "complete".
            job.status = status if status in ("cancelled", "stopped") else "complete"
    except api.RouteError as e:
        job.status, job.error = "error", e.detail
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:  # noqa: BLE001 — a job's failure is reported, never raised
        job.status, job.error = "error", str(e) or type(e).__name__
    finally:
        job.ended = time.monotonic()
        job.done.set()


def start(job: Job) -> Job:
    _jobs[job.id] = job
    job.task = asyncio.get_running_loop().create_task(_drive(job))
    finished = [k for k, j in _jobs.items() if j.status != "running"]
    for k in finished[: max(0, len(_jobs) - KEEP)]:
        _jobs.pop(k, None)
    return job


def new(kind: str, dataset_id: str | None, path: str, body: dict[str, Any], **kw: Any) -> Job:
    return start(Job(id=uuid.uuid4().hex[:12], kind=kind, dataset_id=dataset_id, path=path,
                     body=body, **kw))


async def wait(job: Job, seconds: float) -> None:
    seconds = max(0.0, min(MAX_WAIT, float(seconds)))
    if seconds and not job.done.is_set():
        try:
            await asyncio.wait_for(job.done.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


async def cancel(job: Job) -> None:
    """Stop a running job: its route's cancel when it has one, else leave the stream
    (the route stops its work when its client goes, as for a closed tab)."""
    if job.status != "running":
        return
    job.cancel_requested = True
    if job.cancel_route is not None:
        try:
            await job.cancel_route(job)
            return
        except api.RouteError:
            pass  # nothing to cancel there (not started yet, or finished): leave the stream
    job.disconnect.set()
