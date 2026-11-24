"""The R converter and memory: a conversion that ran out of it is said so (not "could not
be reached"), with what to do locally and on a hosted site."""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scview.core import conversion, rds_export

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


def test_message_with_the_estimate_gives_the_size_to_raise_to():
    text = conversion.out_of_memory_message("converting SCA.rds", 3 * GiB, int(3.6 * GiB), hosted=False)
    assert text.startswith("The converter ran out of memory converting SCA.rds (it has 3.0 GB; "
                           "this object needs about 3.6 GB).")
    assert "at least 6 GB" in text  # 3.6 GB / 0.7 rounded up, as the preflight says
    assert "DietSeurat" in text


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
    assert text.startswith("The converter ran out of memory converting SCA.rds (it has 3.0 GB; this object needs about ")
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
    assert str(exc.value).startswith("The converter stopped and restarted while converting SCA.rds: "
                                     "it most likely ran out of memory (it has 3.0 GB")


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
    assert text.startswith("The converter ran out of memory writing the .rds file (it has 2.0 GB).")
    assert "DietSeurat" not in text
