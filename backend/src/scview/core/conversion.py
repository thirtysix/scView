"""Conversion orchestrator – talks to the R-based converter sidecar service."""

from __future__ import annotations

import httpx


async def trigger_conversion(
    rds_path: str,
    output_path: str,
    converter_url: str,
) -> dict:
    """POST to the converter service to start RDS -> h5ad conversion.

    Parameters
    ----------
    rds_path : str
        Absolute path to the uploaded .rds / .rdata file.
    output_path : str
        Absolute path where the resulting .h5ad should be written.
    converter_url : str
        Base URL of the converter sidecar (e.g. http://converter:8001).

    Returns
    -------
    dict – response body from the converter service.
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"{converter_url}/convert",
            json={
                "input_path": rds_path,
                "output_path": output_path,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def check_conversion_status(
    dataset_id: str,
    converter_url: str,
) -> dict:
    """Poll the converter service for the status of an ongoing conversion.

    Parameters
    ----------
    dataset_id : str
    converter_url : str

    Returns
    -------
    dict – e.g. {"status": "running"} or {"status": "done", "output_path": "..."}
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{converter_url}/status/{dataset_id}")
        resp.raise_for_status()
        return resp.json()
