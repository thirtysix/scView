"""Saved per-cell scores (``core/saved_scores.py``): list, remove, and the save the scoring
endpoints call (``persist``).

A score is computed first, outside the edit lock; ``persist`` then saves it under the lock
the way hand labels are saved (obs and uns only: the matrices are copied, not rewritten; a
large dataset's obs is written into its h5ad in place). If the cells changed in between
(a run filtered them) or a run is changing the dataset now, the score is drawn but not
saved, and the response says why.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from scview.core import saved_scores
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

BUSY_NOTE = "Not saved: the dataset is being processed. Score it again when the run ends."
CHANGED_NOTE = "Not saved: the dataset's cells changed while it was scored. Score it again."


async def _edit_and_save(dm: DatasetManager, dataset_id: str, edit: Callable[[Any], Any],
                         expect_cells: str | None = None) -> Any:
    """Run ``edit`` on the dataset and save it, one edit at a time (``api/v1/labels.py``).
    ``expect_cells``: the cells fingerprint the edit was computed for; a different one
    now raises SavedScoreError 409 before anything changes."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        if expect_cells is not None and adaptor.cells_fingerprint() != expect_cells:
            raise saved_scores.SavedScoreError(CHANGED_NOTE, status=409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            result = edit(work)
            save_edit(adaptor, work, output_path, before, source_unchanged=True)
            return result

        try:
            result = await held.run(_run)
        except saved_scores.SavedScoreError:
            raise  # raised before anything changed
        except Exception:
            dm.evict(dataset_id)  # may hold an edit that was never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


async def persist(dm: DatasetManager, dataset_id: str, cells: str, **score: Any) -> tuple[str | None, str | None]:
    """Save one score (``saved_scores.save`` keywords) computed for the cells ``cells``
    (``adaptor.cells_fingerprint()`` before scoring): (column, None), or (None, why not).
    Never raises: the score on the plot does not depend on the save."""
    if dm.is_busy(dataset_id):
        return None, BUSY_NOTE
    # The same score drawn again (a program or motif clicked twice): nothing to write.
    open_now = dm.get_dataset(dataset_id)
    if open_now is not None and open_now.cells_fingerprint() == cells:
        try:
            same = saved_scores.already_saved(open_now.adata, **score)
        except Exception as e:  # noqa: BLE001 - only a shortcut: save as usual
            logger.warning("Could not compare a score with its saved column: %s", e)
            same = None
        if same is not None:
            return same, None
    try:
        column = await _edit_and_save(dm, dataset_id, lambda a: saved_scores.save(a, **score), expect_cells=cells)
    except saved_scores.SavedScoreError as e:
        return None, str(e)
    except HTTPException as e:
        return None, f"Not saved: {e.detail}"
    except Exception as e:  # noqa: BLE001 - a failed save leaves the score on the plot
        logger.error("Saving a score for %s failed: %s", dataset_id, e, exc_info=True)
        return None, f"Not saved: {e!s}"
    return column, None


@router.get("/datasets/{dataset_id}/scores")
async def list_saved_scores(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The saved per-cell scores: column -> what it holds (kind, name, method, genes...)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"scores": saved_scores.record(adaptor.adata)}


@router.delete("/datasets/{dataset_id}/scores/{column}")
async def remove_saved_score(dataset_id: str, column: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Remove a saved score: its obs column and its record entry. A column that is not a
    saved score (the file's own, a hand label) is a 404 and stays."""
    if dm.is_busy(dataset_id):
        raise HTTPException(status_code=409, detail="The dataset is being processed; remove it when the run ends.")
    try:
        entry = await _edit_and_save(dm, dataset_id, lambda a: saved_scores.remove(a, column))
    except saved_scores.SavedScoreError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Removing saved score %s of %s failed: %s", column, dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not remove the score: {e!s}")
    return {"removed": column, "kind": entry.get("kind"), "name": entry.get("name")}
