"""Watch a pipeline run while it works, and let it stop between steps.

A run near the container's memory limit gave no sign until the kernel killed
the backend (v0.25.0 only says afterwards that it restarted). ``RunWatchdog``
samples memory every ``SAMPLE_SECONDS`` in a thread of its own while a run works
and, for a streamed run, sends ``{"type": "watchdog", ...}`` events into the
run's stream:

- ``kind: "memory"`` when use first crosses 80% and again at 90% of the limit
  (each once, and again only after use has dropped ``REARM_MARGIN`` below it);
- ``kind: "slow_step"`` every ``SLOW_STEP_SECONDS`` one step keeps running.

"Use" is the container's memory minus its inactive file cache (what ``docker
stats`` shows): the kernel drops that cache before it kills anything, and
reading or writing a 2 GB h5ad leaves about that much of it. The run's peak and
each step's seconds and peak go into the run result and the step's provenance
entry (``resources``), so the preflight estimates can be checked against what
runs really used. Samples are 2 s apart, so a spike shorter than that is missed.

``RunControl`` is what the pipeline's step loop consults between steps
(``core/pipeline.py``): a cancel flag (``POST .../assessment/cancel``, which
also terminates a Leiden child process, ``core/leiden.py``) and the pre-step
hook ``before_step``. ``core/processing.py`` registers each run's control for
the cancel endpoint and starts and stops its watchdog.

The sampler only reads two or three small cgroup files per sample, so it holds
the GIL for microseconds; it stops when the run's worker thread ends, on errors
too.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from scview.core import memory_budget

logger = logging.getLogger(__name__)

SAMPLE_SECONDS = 2.0
MEMORY_THRESHOLDS = (0.8, 0.9)
# A threshold fires again only after use has fallen this far (of the limit) below it,
# so a reading that wobbles around 80% does not repeat the warning every sample.
REARM_MARGIN = 0.05
SLOW_STEP_SECONDS = 60.0
GB = 1024**3


class Memory(NamedTuple):
    used: int | None  # bytes in use (without reclaimable file cache)
    limit: int | None  # bytes the backend may use
    limited: bool  # a container limit, not the machine's memory


MemoryReader = Callable[[], Memory]
Emit = Callable[[dict[str, Any]], None]


def read_memory(cgroup: Path = memory_budget.CGROUP) -> Memory:
    """The backend's memory use and limit (``memory_budget.read_budget``, which
    already leaves out the reclaimable file cache)."""
    budget = memory_budget.read_budget(cgroup)
    return Memory(budget.used_bytes, budget.limit_bytes, budget.limited)


def _gb(n: int | None) -> float | None:
    return None if n is None else round(n / GB, 2)


class RunWatchdog:
    """Samples memory while a run works and times its steps (see the module docstring).

    ``read`` and ``clock`` are injectable for tests; ``sample()`` takes one
    sample and can be called directly instead of starting the thread.
    """

    def __init__(
        self,
        emit: Emit | None = None,
        *,
        read: MemoryReader | None = None,
        clock: Callable[[], float] = time.monotonic,
        interval: float | None = None,
        thresholds: tuple[float, ...] = MEMORY_THRESHOLDS,
        slow_every: float | None = None,
    ) -> None:
        self.emit = emit  # set by core/processing.py for a streamed run
        # Module defaults looked up now (not at import), so tests can change them.
        self._read = read if read is not None else read_memory
        self._clock = clock
        self.interval = interval if interval is not None else SAMPLE_SECONDS
        self._thresholds = tuple(sorted(thresholds))
        self._slow_every = slow_every if slow_every is not None else SLOW_STEP_SECONDS
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._armed = dict.fromkeys(self._thresholds, True)
        self.limit_bytes: int | None = None
        self.limited = False
        self.start_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.n_samples = 0
        self._step: str | None = None
        self._step_t0 = 0.0
        self._step_peak: int | None = None
        self._next_slow = self._slow_every
        self.steps: dict[str, dict[str, Any]] = {}  # step -> elapsed_s, peak_gb
        self.events: list[dict[str, Any]] = []  # every warning sent, for the result

    # -- sampling ---------------------------------------------------------

    def sample(self) -> None:
        """Take one sample: update the peaks, send any warning it calls for."""
        try:
            mem = self._read()
        except Exception as e:  # noqa: BLE001 — a failed reading must never stop a run
            logger.debug("Memory reading failed: %s", e)
            mem = Memory(None, None, False)
        now = self._clock()
        out: list[dict[str, Any]] = []
        with self._lock:
            self.n_samples += 1
            if mem.limit:
                self.limit_bytes, self.limited = mem.limit, mem.limited
            if mem.used is not None:
                if self.start_bytes is None:
                    self.start_bytes = mem.used
                self.peak_bytes = max(self.peak_bytes or 0, mem.used)
                if self._step is not None:
                    self._step_peak = max(self._step_peak or 0, mem.used)
                if mem.limit:
                    out.extend(self._memory_events(mem))
            if self._step is not None:
                elapsed = now - self._step_t0
                if elapsed >= self._next_slow:
                    out.append({"type": "watchdog", "kind": "slow_step", "step": self._step,
                                "elapsed_s": int(round(elapsed))})
                    while self._next_slow <= elapsed:
                        self._next_slow += self._slow_every
            self.events.extend(out)
        for event in out:
            self._send(event)

    def _memory_events(self, mem: Memory) -> list[dict[str, Any]]:
        share = mem.used / mem.limit
        fired = None
        for t in self._thresholds:
            if share >= t:
                if self._armed[t]:
                    self._armed[t] = False
                    fired = t
            elif share < t - REARM_MARGIN:
                self._armed[t] = True
        if fired is None:
            return []
        # One event even when a jump crossed both thresholds: the higher one.
        return [{"type": "watchdog", "kind": "memory", "threshold_pct": int(round(fired * 100)),
                 "used_gb": _gb(mem.used), "limit_gb": _gb(mem.limit),
                 "pct": int(round(share * 100)), "limited": mem.limited, "step": self._step}]

    def _send(self, event: dict[str, Any]) -> None:
        if event["kind"] == "memory":
            logger.warning("Run memory at %s%% of %s GB (step %s)",
                           event["pct"], event["limit_gb"], event["step"])
        else:
            logger.info("Step %s still running after %s s", event["step"], event["elapsed_s"])
        if self.emit is None:
            return
        try:
            self.emit(event)
        except Exception as e:  # noqa: BLE001 — the stream may be gone
            logger.debug("Could not send a watchdog event: %s", e)

    # -- steps --------------------------------------------------------------

    def step_started(self, step: str) -> None:
        with self._lock:
            self._step = step
            self._step_t0 = self._clock()
            self._step_peak = None
            self._next_slow = self._slow_every
        self.sample()

    def step_finished(self, step: str) -> None:
        self.sample()
        with self._lock:
            if self._step != step:
                return
            self.steps[step] = {"elapsed_s": round(self._clock() - self._step_t0, 2),
                                "peak_gb": _gb(self._step_peak)}
            self._step = None

    # -- thread -------------------------------------------------------------

    def start(self) -> None:
        """Sample now, then every ``interval`` seconds until ``stop()``."""
        if self._thread is not None:
            return
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, name="run-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.interval + 1)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def summary(self) -> dict[str, Any]:
        """What the run used, for its result: peak and starting use, the limit, each
        step's seconds and peak, and the warnings sent."""
        with self._lock:
            return {
                "peak_gb": _gb(self.peak_bytes),
                "start_gb": _gb(self.start_bytes),
                "limit_gb": _gb(self.limit_bytes),
                "limited": self.limited,
                "sample_seconds": self.interval,
                "n_samples": self.n_samples,
                "steps": {k: dict(v) for k, v in self.steps.items()},
                "warnings": [dict(e) for e in self.events],
            }


# ---------------------------------------------------------------------------
# Run control: cancel + the pre-step hook
# ---------------------------------------------------------------------------

# (step, adata, params) -> a message that stops the run before ``step``, or None.
BeforeStep = Callable[[str, Any, Any], "str | None"]


def default_before_step(step: str, adata: Any, params: Any) -> str | None:
    """The check every pipeline run makes before each step: stop before a step the
    preflight estimate says cannot fit in the memory left.

    A step that outgrows the container limit gets the backend killed, which loses
    the whole run; the watchdog's 2 s sampler cannot warn in time (Scrublet on Kang
    grows 2.3 GB inside 2 s). So the run stops before such a step instead, keeping
    the steps already done. It stops only when even the low end of the estimate's
    error band (``preflight.ERROR_BAND``) is more than the memory free, and only
    under a container limit: without one the "limit" is the machine's memory,
    which the backend shares with everything else, so the check would be a guess.

    Contract for a check plugged in here (or passed as ``RunControl(before_step=)``):
    called in the run's worker thread with the step about to run, the AnnData as
    it is now (earlier steps applied) and the run's ``PipelineParams``. Return a
    plain sentence to stop the run before ``step`` (status "stopped", that
    sentence as ``stop_reason``; nothing is saved, as for a cancel), or None to
    let it run. It must be quick (well under a second: read shapes and ``nnz``,
    never densify), must not change ``adata``.
    An exception is logged and the step runs.
    """
    from scview.core import preflight

    avail = preflight.available_memory()
    if not avail.limited or avail.available_bytes is None or avail.limit_bytes is None:
        return None
    need = preflight.step_peak_for(adata, step, params)
    if need * (1 - preflight.ERROR_BAND) <= avail.available_bytes:
        return None
    label = step.replace("_", " ")
    return (f"Stopped before {label}: it needs about {need / GB:.1f} GB more at its peak, and scView "
            f"has {avail.available_bytes / GB:.1f} GB free of its {avail.limit_bytes / GB:.1f} GB limit, "
            f"so the system would have stopped the backend. Nothing was saved: the dataset is as it "
            f"was before the run. Raise "
            f"SCVIEW_MEMORY_GB in .env (then restart scView), or leave this step out.")


_DEFAULT = object()
_CURRENT: contextvars.ContextVar[RunControl | None] = contextvars.ContextVar(
    "scview_run_control", default=None)


class RunCancelled(Exception):
    """Raised inside a step that noticed the run was cancelled (between substeps)."""


def current() -> RunControl | None:
    """The control of the run whose step is executing in this thread, if any."""
    return _CURRENT.get()


class RunControl:
    """What a pipeline run checks between steps: cancel and ``before_step``."""

    def __init__(self, watchdog: RunWatchdog | None = None,
                 before_step: BeforeStep | None | object = _DEFAULT,
                 plan: tuple[list[str], Any] | None = None) -> None:
        self.watchdog = watchdog if watchdog is not None else RunWatchdog()
        # (steps, PipelineParams) the run will make: before it starts, core/processing.py
        # closes other open datasets whose memory its estimated peak needs.
        self.plan = plan
        # Looked up now, so a check plugged into default_before_step applies.
        self.before_step = default_before_step if before_step is _DEFAULT else before_step
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._children: set[Any] = set()
        self.current_step: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        """Ask the run to stop at the next step boundary; end a child process now."""
        self._cancel.set()
        with self._lock:
            children = list(self._children)
        for proc in children:
            with contextlib.suppress(OSError):
                proc.terminate()

    def halt_before(self, step: str, adata: Any, params: Any) -> tuple[str, str] | None:
        """(status, reason) when the run must not start ``step``, else None."""
        if self.cancelled:
            return "cancelled", cancelled_reason("before", step)
        if self.before_step is None:
            return None
        try:
            message = self.before_step(step, adata, params)
        except Exception:  # noqa: BLE001 — a broken check must not stop a run
            logger.warning("The pre-step check failed before %s; running it", step, exc_info=True)
            return None
        return ("stopped", str(message)) if message else None

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Around one step: it is timed and watched, and its Leiden child cancellable."""
        token = _CURRENT.set(self)
        self.current_step = name
        self.watchdog.step_started(name)
        try:
            yield
        finally:
            self.watchdog.step_finished(name)
            self.current_step = None
            try:
                _CURRENT.reset(token)
            except ValueError:  # a generator resumed from another thread's context
                _CURRENT.set(None)

    @contextlib.contextmanager
    def child(self, proc: Any) -> Iterator[None]:
        """Register a child process for ``cancel`` to terminate while it runs."""
        with self._lock:
            self._children.add(proc)
        if self.cancelled:  # the cancel came just before the child started
            with contextlib.suppress(OSError):
                proc.terminate()
        try:
            yield
        finally:
            with self._lock:
                self._children.discard(proc)


def cancelled_reason(when: str, step: str) -> str:
    """The stop reason of a cancelled run (``when``: "before" | "during")."""
    return (f"Cancelled {when} {step.replace('_', ' ')}: nothing was saved, so the dataset is "
            "as it was before the run.")
