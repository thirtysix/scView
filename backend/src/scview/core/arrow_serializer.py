"""Arrow IPC serializer – numpy/pandas → Arrow IPC binary for frontend."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pyarrow as pa


def embedding_to_arrow_ipc(
    coords: np.ndarray,
    color_values: np.ndarray | None = None,
    color_name: str = "color",
) -> bytes:
    """Serialize embedding coordinates (and optional color column) to Arrow IPC.

    Parameters
    ----------
    coords : ndarray, shape (n_cells, 2) or (n_cells, 3)
        Float32 embedding coordinates.
    color_values : ndarray, optional
        Per-cell values for coloring (float32 for continuous, int32 for category index).
    color_name : str
        Column name for the color values.

    Returns
    -------
    bytes – Arrow IPC stream format.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n_cells, n_dims = coords.shape

    arrays = [
        pa.array(coords[:, 0], type=pa.float32()),
        pa.array(coords[:, 1], type=pa.float32()),
    ]
    names = ["x", "y"]

    if n_dims == 3:
        # Only include z for genuine 3D embeddings, not PCA with 50+ dims
        arrays.append(pa.array(coords[:, 2], type=pa.float32()))
        names.append("z")

    if color_values is not None:
        color_values = np.asarray(color_values)
        if np.issubdtype(color_values.dtype, np.floating):
            arrays.append(pa.array(color_values.astype(np.float32), type=pa.float32()))
        else:
            arrays.append(pa.array(color_values.astype(np.int32), type=pa.int32()))
        names.append(color_name)

    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    return _batch_to_ipc_bytes(batch)


def expression_to_arrow_ipc(
    values: np.ndarray,
    gene_names: list[str],
) -> bytes:
    """Serialize expression matrix to Arrow IPC.

    Parameters
    ----------
    values : ndarray, shape (n_cells, n_genes)
        Expression values (float32).
    gene_names : list[str]
        Column names for each gene.

    Returns
    -------
    bytes – Arrow IPC stream format.
    """
    values = np.asarray(values, dtype=np.float32)
    arrays = []
    names = []
    for i, name in enumerate(gene_names):
        arrays.append(pa.array(values[:, i], type=pa.float32()))
        names.append(name)

    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    return _batch_to_ipc_bytes(batch)


def dataframe_to_arrow_ipc(df: pd.DataFrame) -> bytes:
    """Serialize a pandas DataFrame to Arrow IPC."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    return _table_to_ipc_bytes(table)


def series_to_arrow_ipc(series: pd.Series, name: str = "values") -> bytes:
    """Serialize a pandas Series to Arrow IPC."""
    if hasattr(series, "cat"):
        # Convert categorical to string for Arrow
        arr = pa.array(series.astype(str))
    else:
        arr = pa.array(series)
    batch = pa.RecordBatch.from_arrays([arr], names=[name])
    return _batch_to_ipc_bytes(batch)


def _batch_to_ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


def _table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()
