"""The R converter and memory: a conversion that ran out of it is said so (not "could not
be reached"), with what to do locally and on a hosted site."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scview.core import conversion, conversion_preflight, rds_export

GiB = 1024 ** 3
URL = "http://conv:8001"


# ---------------------------------------------------------------------------
# The out-of-memory sentence
# ---------------------------------------------------------------------------

def test_local_advice_names_the_env_setting_and_a_size():
    text = conversion.converter_memory_advice(3 * GiB, 5, hosted=False)
    assert "SCVIEW_MEMORY_GB" in text and "at least 5 GB" in text and "restart scView" in text


def test_hosted_advice_asks_the_site_admin_and_never_names_env_files():
    text = conversion.converter_memory_advice(3 * GiB, 5, hosted=True)
    assert "site admin" in text and "at least 5 GB" in text
    assert ".env" not in text and "SCVIEW_MEMORY_GB" not in text


def test_no_limit_set_advises_freeing_the_machine():
    text = conversion.converter_memory_advice(None, None, hosted=False)
    assert "Docker or Podman" in text and "SCVIEW_MEMORY_GB" not in text


def _once(text: str, *parts: str) -> None:
    for part in parts:
        assert text.count(part) == 1, (part, text)


def test_message_with_the_estimate_above_the_limit_gives_the_size_to_raise_to():
    text = conversion.out_of_memory_message("converting SCA.rds", 3 * GiB, int(3.6 * GiB), hosted=False)
    assert text.startswith("The converter ran out of its 3.0 GB converting SCA.rds. "
                           "This object needs about 3.6 GB. Raise SCVIEW_MEMORY_GB")
    assert "at least 6 GB" in text  # 3.6 GB / 0.7 rounded up, as the preflight says
    assert "DietSeurat" in text
    assert "too low" not in text
    _once(text, "3.0 GB", "3.6 GB")


def test_message_with_the_estimate_below_the_limit_says_it_was_too_low():
    """The hub (v0.82.0): "ran out of memory (it has 3.0 GB; this object needs about 2.0 GB).
    Ask the site admin ... (it has 3.0 GB)": the estimate contradicted the fact, and the
    limit came twice. The fact leads; the estimate is said to be too low."""
    for hosted in (False, True):
        text = conversion.out_of_memory_message("converting SCA.rds", 3 * GiB, 2 * GiB, hosted=hosted)
        assert text.startswith("The converter ran out of its 3.0 GB converting SCA.rds. "
                               "The estimate (about 2.0 GB) was too low for this object: "
                               "it needs more than 3.0 GB. ")
        assert "needs about" not in text
        assert "at least" not in text  # ceil(2.0 / 0.7) = 3 GB: no size above the limit to vouch for
        _once(text, "ran out", "2.0 GB", "DietSeurat")
        assert text.count("3.0 GB") == 2  # the fact, and "more than" it
        assert ("site admin" in text) is hosted
    hub = conversion.out_of_memory_message("converting SCA.rds", 3 * GiB, 2 * GiB, hosted=True)
    assert "Ask the site admin to give the R converter more memory. " in hub
    assert "(it has" not in hub


def test_a_tight_estimate_below_the_limit_still_gives_a_size_above_it():
    text = conversion.out_of_memory_message("converting x.rds", 3 * GiB, int(2.5 * GiB), hosted=True)
    assert "The estimate (about 2.5 GB) was too low for this object: it needs more than 3.0 GB." in text
    assert "Ask the site admin to give the R converter more memory (this needs at least 4 GB)." in text


def test_message_without_a_limit_gives_the_estimate():
    text = conversion.out_of_memory_message("converting x.rds", None, int(3.6 * GiB), hosted=False)
    assert text.startswith("The converter ran out of memory converting x.rds. This object needs about 3.6 GB. ")
    assert "Docker or Podman more memory to at least 6 GB" in text


def test_the_dietseurat_tip_only_when_it_can_help():
    args = ("converting x.rds", 3 * GiB, 4 * GiB)
    counts_only = conversion.out_of_memory_message(*args, hosted=False, droppable_layers=False)
    assert "DietSeurat" not in counts_only and "smaller object" not in counts_only
    layered = conversion.out_of_memory_message(*args, hosted=False, droppable_layers=True)
    assert layered.endswith(conversion.SMALLER_OBJECT_TIP)
    unknown = conversion.out_of_memory_message(*args, hosted=False)
    assert unknown.endswith(conversion.SMALLER_OBJECT_IF)
    assert "Or, if it has data or scale.data layers, save a smaller object" in unknown


@pytest.mark.parametrize("found,tip", [(False, None), (True, "SMALLER_OBJECT_TIP"), ("x", "SMALLER_OBJECT_IF")])
async def test_the_out_of_memory_answer_reads_the_layers_from_the_progress_file(small_rds, tmp_path, found, tip):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"stage": "writing", "source_class": "Seurat", "droppable_layers": found}))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error", "error_kind": "out_of_memory",
                                         "memory_limit_bytes": 3 * GiB, "error": "x"})

    with pytest.raises(conversion.ConversionError) as exc:
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL, progress_path=str(progress),
                                            transport=httpx.MockTransport(handler))
    text = str(exc.value)
    if tip is None:
        assert "DietSeurat" not in text
    else:
        assert text.endswith(getattr(conversion, tip))


def test_message_without_a_size_when_the_estimate_said_it_fits():
    text = conversion.out_of_memory_message("converting x.rds", 8 * GiB, 2 * GiB, hosted=False)
    assert conversion.target_gb(2 * GiB) == 3
    assert "at least" not in text


# ---------------------------------------------------------------------------
# trigger_conversion: the converter's answers and a dropped connection
# ---------------------------------------------------------------------------

@pytest.fixture
def small_rds(tmp_path: Path) -> Path:
    p = tmp_path / "SCA.rds"
    p.write_bytes(gzip.compress(b"X\n" + bytes(3000)))
    return p


async def test_out_of_memory_answer_becomes_the_memory_sentence(small_rds):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={
            "status": "error", "error_kind": "out_of_memory", "memory_limit_bytes": 3 * GiB,
            "error": "The converter ran out of memory converting SCA.rds (it has 3.0 GB)."})

    with pytest.raises(conversion.ConversionError) as exc:
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL,
                                            transport=httpx.MockTransport(handler))
    text = str(exc.value)
    # the estimate of a 3 kB object is far under 3 GB: the fact leads, the estimate was too low
    assert text.startswith("The converter ran out of its 3.0 GB converting SCA.rds. The estimate (about ")
    assert "SCVIEW_MEMORY_GB" in text
    assert "could not be reached" not in text


async def test_out_of_memory_on_a_hosted_site_asks_the_admin(small_rds, monkeypatch):
    monkeypatch.setattr(conversion, "hosted", lambda: True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error", "error_kind": "out_of_memory",
                                         "memory_limit_bytes": 3 * GiB, "error": "x"})

    with pytest.raises(conversion.ConversionError) as exc:
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL,
                                            transport=httpx.MockTransport(handler))
    assert "site admin" in str(exc.value) and ".env" not in str(exc.value)


async def test_a_conversion_reports_its_measured_peak_beside_the_estimate(small_rds):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "message": "Converted",
                                         "report": {"source_class": "Seurat"}, "peak_bytes": 2 * GiB})

    data = await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL,
                                               transport=httpx.MockTransport(handler))
    mem = data["report"]["memory"]
    assert data["report"]["source_class"] == "Seurat"
    assert mem["peak_bytes"] == 2 * GiB and mem["serialized_bytes"] == 3002 and mem["serialized_exact"]
    assert mem["estimate_bytes"] == conversion_preflight.estimate_peak(3002)


async def test_an_older_converter_without_a_peak_reports_no_memory(small_rds):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "message": "Converted", "report": {"kept": []}})

    data = await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL,
                                               transport=httpx.MockTransport(handler))
    assert "memory" not in data["report"]


def _dropping(health: dict | None):
    """POST /convert drops mid-request; GET /health answers ``health`` (None: down)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.",
                                            request=request)
        if health is None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=health)
    return httpx.MockTransport(handler)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000%z")


async def test_a_converter_that_restarted_mid_conversion_most_likely_ran_out_of_memory(small_rds):
    later = _stamp(datetime.now(timezone.utc) + timedelta(seconds=5))
    transport = _dropping({"status": "ok", "started_at": later, "memory_limit_bytes": 3 * GiB})
    with pytest.raises(conversion.ConversionError) as exc:
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL, transport=transport)
    text = str(exc.value)
    assert text.startswith("The converter stopped and restarted while converting SCA.rds: "
                           "it most likely ran out of its 3.0 GB. If so, the estimate (about ")
    assert "was too low for this object: it needs more than 3.0 GB." in text
    assert text.count("3.0 GB") == 2


def test_the_restart_message_with_the_estimate_above_the_limit():
    text = conversion.out_of_memory_message("converting x.rds", 3 * GiB, int(3.6 * GiB), hosted=True,
                                            restarted=True)
    assert text.startswith("The converter stopped and restarted while converting x.rds: it most likely "
                           "ran out of its 3.0 GB. This object needs about 3.6 GB. Ask the site admin to "
                           "give the R converter more memory (this needs at least 6 GB).")


async def test_a_drop_without_a_restart_is_not_called_memory(small_rds):
    earlier = _stamp(datetime.now(timezone.utc) - timedelta(hours=2))
    transport = _dropping({"status": "ok", "started_at": earlier, "memory_limit_bytes": 3 * GiB})
    with pytest.raises(conversion.ConversionError, match="connection to the R converter dropped"):
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL, transport=transport)


async def test_a_converter_that_stays_down_could_not_be_reached(small_rds, monkeypatch):
    monkeypatch.setattr(conversion, "RESTART_WAIT_S", 0.05)
    monkeypatch.setattr(conversion, "RESTART_POLL_S", 0.01)
    with pytest.raises(conversion.ConversionError, match="could not be reached"):
        await conversion.trigger_conversion(str(small_rds), "/d/x.h5ad", URL, transport=_dropping(None))


async def test_export_out_of_memory_is_said_as_memory():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error", "error_kind": "out_of_memory",
                                         "memory_limit_bytes": 2 * GiB, "error": "x"})

    with pytest.raises(conversion.ConversionError) as exc:
        await rds_export.trigger_export("/d/s.h5ad", "/d/o.rds", "seurat", URL,
                                        transport=httpx.MockTransport(handler))
    text = str(exc.value)
    assert text.startswith("The converter ran out of its 2.0 GB writing the .rds file. Raise SCVIEW_MEMORY_GB")
    assert "DietSeurat" not in text
