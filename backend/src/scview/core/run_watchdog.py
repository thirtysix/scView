"""Watch a pipeline run while it works, and let it stop between steps.

A run near the container's memory limit gave no sign until the kernel killed
the backend (v0.25.0 only says afterwards that it restarted). ``RunWatchdog``
samples memory every ``SAMPLE_SECONDS`` in a thread of its own while a run works
and, for a streamed run, sends ``{"type": "watchdog", ...}`` events into the
run's stream:

- ``kind: "memory"`` when use first crosses 80% and again at 90% of the limit
  (each once, and again only after use has dropped ``REARM_MARGIN`` below it);
- ``kind: "slow_step"`` while one step keeps running: when the run has the step's
  preflight time range (``expected``, set by ``core/processing.py``, or ``expect`` from
  a single estimate), first at ``SLOW_ESTIMATE_FACTOR`` times its estimate (never before
  ``SLOW_STEP_SECONDS``), with the range (``expected_s``, ``expected_text``) and the
  estimate (``estimate_s``), so the page can say whether the step is still inside its
  estimate or past it; without one, first at ``SLOW_STEP_SECONDS``; then every
  ``SLOW_STEP_SECONDS``;
- ``kind: "waiting"`` (sent by the pre-step check, not the sampler) while the run waits
  before a step for other work to finish (``room_for_step``), with the sentence to show.

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
# A step with an estimate gets its first slow note at this many times it (QA round 4: a note
# at 61 s on an ATAC run estimated at 139 s said nothing was wrong).
SLOW_ESTIMATE_FACTOR = 1.5
GB = 1024**3
# A run whose next step fits only once other work has finished (another run's or a read's
# reserved growth, ``DatasetManager.outstanding``) waits for it before the step instead of
# stopping (``room_for_step``): it checks again every ``QUEUE_POLL_SECONDS``, sends a
# ``kind: "waiting"`` event at once and then every ``QUEUE_NOTE_SECONDS``, and stops as
# before once it has waited ``QUEUE_WAIT_SECONDS`` (30 min: about the longest step of a
# large in-memory run; a whole 1.3M-cell run takes 1 h 45 min, so a queue behind one stops
# rather than holding its dataset for that long). 0 turns the wait off.
QUEUE_WAIT_SECONDS = 30 * 60.0
QUEUE_POLL_SECONDS = 2.0
QUEUE_NOTE_SECONDS = 60.0


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


def _point(rng: tuple[float, float]) -> float:
    """The point estimate of a (low, high) range: the preflight's ranges are half to twice
    it (``preflight._range``), so their geometric mean."""
    lo, hi = float(rng[0]), float(rng[1])
    return (lo * hi) ** 0.5 if lo > 0 else hi


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
        # step -> (low, high) seconds the preflight estimated for it (core/processing.py)
        self.expected: dict[str, tuple[float, float]] = {}
        self._points: dict[str, float] = {}  # step -> its single estimate (``expect``)
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
                    out.append(self._slow_event(elapsed))
                    while self._next_slow <= elapsed:
                        self._next_slow += self._slow_every
            self.events.extend(out)
        for event in out:
            self._send(event)

    def _slow_event(self, elapsed: float) -> dict[str, Any]:
        event = {"type": "watchdog", "kind": "slow_step", "step": self._step,
                 "elapsed_s": int(round(elapsed))}
        rng = self.expected.get(self._step)
        if rng is not None:
            from scview.core.preflight import time_text

            event["expected_s"] = [float(rng[0]), float(rng[1])]
            event["expected_text"] = time_text(rng)
            event["estimate_s"] = int(round(self._estimate(self._step)))
        return event

    def _first_slow(self, step: str) -> float:
        """When ``step``'s first slow note is due: ``SLOW_ESTIMATE_FACTOR`` times its
        estimate (at least the fixed interval), or the fixed interval without one."""
        if step not in self.expected:
            return self._slow_every
        return max(self._slow_every, SLOW_ESTIMATE_FACTOR * self._estimate(step))

    def _estimate(self, step: str) -> float:
        return self._points.get(step) or _point(self.expected[step])

    def expect(self, step: str, seconds: float | None) -> None:
        """A run with one estimate for ``step`` (seconds): its range is the estimate's error
        band (``preflight.ERROR_BAND``). None or 0: no estimate."""
        from scview.core.preflight import ERROR_BAND

        if seconds and seconds > 0:
            self.expected[step] = (float(seconds) * (1 - ERROR_BAND), float(seconds) * (1 + ERROR_BAND))
            self._points[step] = float(seconds)

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

    def note(self, event: dict[str, Any]) -> None:
        """Send an event that is not a warning (``kind: "waiting"``): logged and streamed,
        not kept for the run's result."""
        self._send(event)

    def _send(self, event: dict[str, Any]) -> None:
        if event["kind"] == "memory":
            logger.warning("Run memory at %s%% of %s GB (step %s)",
                           event["pct"], event["limit_gb"], event["step"])
        elif event["kind"] == "waiting":
            logger.info("Run waits before %s: %s", event["step"], event["message"])
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
            self._next_slow = self._first_slow(step)
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


LOG_MAX = 20.0  # log1p of normalised counts stays under this (the assessor's log check)
_LOG_ROWS = 200


def log_only_stop(step: str, adata: Any) -> str | None:
    """Stop before normalisation when X already holds log-normalised values and there
    are no counts to start from: not whole numbers, none negative, max <= ``LOG_MAX``.

    The normalisation step keeps X as the counts when there is nothing else, so a
    published h5ad that ships only log values (infercnvpy's maynard2020_3k) was logged
    twice, which flattens every fold change (its tumour cells' copy-number score fell
    from 5.5x to 1.8x the reference). Counts an ambient RNA correction made are not whole
    numbers either; they keep their originals in ``counts_uncorrected`` and pass."""
    import numpy as np
    import scipy.sparse as sp

    if step != "normalization" or "counts_uncorrected" in adata.layers:
        return None
    m = adata.X
    m = m[: min(_LOG_ROWS, m.shape[0])]
    vals = np.asarray(m.data if sp.issparse(m) else (m.toarray() if hasattr(m, "toarray") else m))
    vals = vals.ravel().astype(np.float64)
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if vals.size == 0 or vals.min() < 0 or np.all(np.mod(vals, 1) == 0) or vals.max() > LOG_MAX:
        return None
    return (f"Stopped before normalization: the expression values are already log-normalised "
            f"(not whole numbers, largest {vals.max():.1f}) and the dataset has no raw counts, so "
            f"normalising and logging them again would flatten every difference between cells. "
            f"Leave out Reset to counts, Normalization and Log transform, and run the other steps. "
            f"Nothing was saved: the dataset is as it was before the run.")


# What a stopped run says it kept, and how to make a step fit (``memory_stop``).
NOTHING_SAVED = "Nothing was saved: the dataset is as it was before the run."
RAISE_LIMIT = "Raise SCVIEW_MEMORY_GB in .env (then restart scView), or leave this step out."


def others_stop(lead: str, need: int, avail: Any, others: int, waited: float = 0.0,
                saved: str = NOTHING_SAVED) -> str:
    """The stop of a step that fits by itself but not beside what other work still running
    is expected to grow by (``DatasetManager.outstanding``): that fits again once it ends,
    so the sentence says to run it again, not to raise the limit. ``waited``: seconds the
    run waited for that work first (``room_for_step``)."""
    after = f" It waited {_minutes(waited)} for that work to finish first." if waited >= 1 else ""
    return (f"{lead} about {need / GB:.1f} GB more at its peak, and of the "
            f"{avail.available_bytes / GB:.1f} GB scView has free ({avail.limit_bytes / GB:.1f} GB limit) other "
            f"work still running will take about {others / GB:.1f} GB, so together they would get the backend "
            f"stopped.{after} {saved} Run it again once that work has finished.")


def _minutes(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m} min" if m >= 1 else f"{int(round(seconds))} s"


class Room(NamedTuple):
    """``room_for_step``'s answer: ``verdict`` "fits" | "others" (fits only once other
    work has finished, and waiting for it did not help) | "never" (not even with nothing
    else running) | "cancelled" (while waiting), with the last reading."""

    verdict: str
    avail: Any
    others: int
    waited: float = 0.0


def room_for_step(control: RunControl | None, step: str, need: int) -> Room:
    """Whether ``need`` more bytes (the step's estimated peak) fit before ``step``: the low
    end of the estimate's error band (``preflight.ERROR_BAND``) against what is free less
    what other runs and reads are still expected to grow by (``RunControl.others_growth``).

    When it fits only once that other work has finished, a run wired for it
    (``RunControl.waiting_on``, set by ``core/processing.py``) waits instead of stopping,
    in its worker thread: it is marked waiting with the manager (``on_wait``), so a run
    that started earlier no longer counts its growth and goes on (a later run waits for an
    earlier one, never the reverse: two runs cannot wait on each other), checks again every
    ``QUEUE_POLL_SECONDS``, sends a ``kind: "waiting"`` event, and gives up after
    ``QUEUE_WAIT_SECONDS``. Cancel ends the wait at once. Without a container limit it
    always fits (the check would be a guess)."""
    from scview.core import preflight

    low = need * (1 - preflight.ERROR_BAND)
    started: float | None = None
    next_note = 0.0
    try:
        while True:
            avail = preflight.available_memory()
            if not avail.limited or avail.available_bytes is None or avail.limit_bytes is None:
                return Room("fits", avail, 0)
            others = control.others_growth(avail.used_bytes) if control is not None else 0
            waited = 0.0 if started is None else time.monotonic() - started
            if low <= avail.available_bytes - others:
                return Room("fits", avail, others, waited)
            if low > avail.available_bytes:
                return Room("never", avail, others, waited)
            if control is None or control.waiting_on is None or waited >= QUEUE_WAIT_SECONDS:
                return Room("others", avail, others, waited)
            if started is None:
                started = time.monotonic()
                control.set_waiting(True)
            if waited >= next_note:
                control.note_waiting(step, need, avail, others, waited)
                next_note += QUEUE_NOTE_SECONDS
            if control.wait_cancelled(QUEUE_POLL_SECONDS):
                return Room("cancelled", avail, others, waited)
    finally:
        if started is not None and control is not None:
            control.set_waiting(False)


def default_before_step(step: str, adata: Any, params: Any) -> str | None:
    """The check every pipeline run makes before each step: stop before a step the
    preflight estimate says cannot fit in the memory left, which is what is free less
    what other runs and reads are still expected to grow by (``RunControl.others_growth``):
    two runs that each pass alone can overrun together. A step that would fit once that
    other work has finished waits for it first (``room_for_step``).

    A step that outgrows the container limit gets the backend killed, which loses
    the whole run; the watchdog's 2 s sampler cannot warn in time (Scrublet on Kang
    grows 2.3 GB inside 2 s). So the run stops before such a step instead, keeping
    the steps already done. It also stops before normalising values that are already
    log-normalised (``log_only_stop``). The memory stop happens only when even the low end of the estimate's
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

    reason = log_only_stop(step, adata)
    if reason:
        return reason
    avail = preflight.available_memory()
    if not avail.limited or avail.available_bytes is None or avail.limit_bytes is None:
        return None
    return memory_stop(step, preflight.step_peak_for(adata, step, params))


def memory_stop(step: str, need: int, *, label: str | None = None, saved: str = NOTHING_SAVED,
                fix: str = RAISE_LIMIT) -> str | None:
    """The memory half of ``default_before_step`` for a step needing ``need`` more bytes
    at its peak: None when it fits (after waiting for other work, ``room_for_step``), else
    the stop sentence. ``label`` names the step in it (default: its name), ``saved`` says
    what the stop kept and ``fix`` how to make it fit: a joint embedding
    (core/integration.py) keeps no dataset and has no step to leave out."""
    from scview.core import preflight

    avail = preflight.available_memory()
    if not avail.limited or avail.available_bytes is None or avail.limit_bytes is None:
        return None
    room = room_for_step(current(), step, need)  # waits while other work holds the room
    if room.verdict in ("fits", "cancelled"):  # cancelled: ``halt_before`` says so
        return None
    label = label or step.replace("_", " ")
    avail = room.avail
    if room.verdict == "others":
        return others_stop(f"Stopped before {label}: it needs", need, avail, room.others, room.waited,
                           saved)
    return (f"Stopped before {label}: it needs about {need / GB:.1f} GB more at its peak, and scView "
            f"has {avail.available_bytes / GB:.1f} GB free of its {avail.limit_bytes / GB:.1f} GB limit, "
            f"so the system would have stopped the backend. {saved} {fix}")


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
                 plan: tuple[list[str], Any] | None = None, allow_large: bool = True) -> None:
        self.watchdog = watchdog if watchdog is not None else RunWatchdog()
        # False: a run that would put a dataset in large-dataset layout is refused before it
        # starts (core/processing.py), naming what that turns off (the MCP's opt-in).
        self.allow_large = allow_large
        # (steps, PipelineParams) the run will make: before it starts, core/processing.py
        # closes other open datasets whose memory its estimated peak needs.
        self.plan = plan
        # Work without a plan that estimates its own growth (the re-filter re-processes the
        # original upload, not the dataset: ``mito_refilter.run_step_needs``): per step, the
        # bytes it needs beyond what it holds; core/processing.py reserves it like a run's.
        self.step_needs: Callable[[Any], dict[str, int]] | None = None  # called with the adaptor
        # In memory or in large-dataset mode (core/large_run.py ModeChoice): chosen by
        # core/processing.py before the run starts; None: not chosen (in memory).
        self.mode: Any = None
        # Bytes other runs and reads are still expected to grow by, from the bytes used
        # now (``DatasetManager.outstanding`` without this run): set by core/processing.py.
        self.other_growth: Callable[[int], int] | None = None
        # Set by core/processing.py: what a pre-step wait waits for, in words, from the
        # bytes used now (``DatasetManager.waits_for``); None: the run never waits, it
        # stops (``room_for_step``). ``on_wait`` tells the manager it waits or goes on.
        self.waiting_on: Callable[[int], list[str]] | None = None
        self.on_wait: Callable[[bool], None] | None = None
        # Looked up now, so a check plugged into default_before_step applies.
        self.before_step = default_before_step if before_step is _DEFAULT else before_step
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._children: set[Any] = set()
        self.current_step: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def others_growth(self, used: int | None) -> int:
        """What other runs and reads are still expected to grow by; 0 when nothing is known."""
        if self.other_growth is None or used is None:
            return 0
        try:
            return max(0, int(self.other_growth(used)))
        except Exception:  # noqa: BLE001 — the check goes on without them
            logger.warning("Could not read the other runs' reservations", exc_info=True)
            return 0

    def set_waiting(self, waiting: bool) -> None:
        if self.on_wait is None:
            return
        try:
            self.on_wait(waiting)
        except Exception:  # noqa: BLE001 — the wait goes on without it
            logger.warning("Could not mark the run as waiting", exc_info=True)

    def wait_cancelled(self, seconds: float) -> bool:
        """Sleep ``seconds``, or less when the run is cancelled meanwhile; whether it was."""
        return self._cancel.wait(seconds)

    def note_waiting(self, step: str, need: int, avail: Any, others: int, waited: float) -> None:
        """The ``kind: "waiting"`` event: what the run waits for before ``step``, and why."""
        try:
            holders = list(self.waiting_on(avail.used_bytes)) if self.waiting_on is not None else []
        except Exception:  # noqa: BLE001 — the sentence goes on without names
            logger.warning("Could not name the work this run waits for", exc_info=True)
            holders = []
        names = [h[:1].lower() + h[1:] for h in holders]
        who = "; ".join(names) if names else "other work still running"
        label = step.replace("_", " ")
        message = (f"Waiting before {label}: it needs about {need / GB:.1f} GB more at its peak, and of the "
                   f"{avail.available_bytes / GB:.1f} GB free, {who} will still take about {others / GB:.1f} GB. "
                   f"The run goes on once that has finished (it waits up to "
                   f"{_minutes(QUEUE_WAIT_SECONDS)}); Cancel stops it without saving.")
        self.watchdog.note({"type": "watchdog", "kind": "waiting", "step": step, "waiting_for": holders,
                            "waited_s": int(round(waited)), "max_wait_s": int(QUEUE_WAIT_SECONDS),
                            "need_gb": _gb(need), "others_gb": _gb(others),
                            "available_gb": _gb(avail.available_bytes), "message": message})

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
        token = _CURRENT.set(self)  # default_before_step asks it for the other runs' growth
        try:
            message = self.before_step(step, adata, params)
        except Exception:  # noqa: BLE001 — a broken check must not stop a run
            logger.warning("The pre-step check failed before %s; running it", step, exc_info=True)
            message = None
        finally:
            _CURRENT.reset(token)
        if self.cancelled:  # while the check waited for other work (``room_for_step``)
            return "cancelled", cancelled_reason("before", step)
        return ("stopped", str(message)) if message else None

    @contextlib.contextmanager
    def step(self, name: str, timed: bool = True) -> Iterator[None]:
        """Around one step: it is timed and watched, and its Leiden child cancellable.
        ``timed`` False: only the latter (a part of a step the caller times as a whole)."""
        token = _CURRENT.set(self)
        self.current_step = name
        if timed:
            self.watchdog.step_started(name)
        try:
            yield
        finally:
            if timed:
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
