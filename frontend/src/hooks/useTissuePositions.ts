import { useEffect, useMemo, useState } from "react";
import type { DatasetInfo, SpatialLibrary } from "@/api/types";
import { fetchSpatialCoords } from "@/api/spatial";
import { decodeArrowBuffer } from "@/lib/arrow";
import { activeLibrary, maskToLibrary, tissueSpatial } from "@/lib/spatial";
import { useTissueViewStore } from "@/stores/tissueViewStore";

/**
 * The tissue plot's positions for the section shown. One library: the
 * positions as they are. Several: the library column is fetched once (its
 * codes per cell) and the other sections' cells are masked out (NaN), so the
 * plot shows one section at a time; null until the codes are in.
 * Outside the tissue plot: `positions` unchanged, `library` null.
 */
export function useTissuePositions(
  dataset: DatasetInfo | null,
  embedding: string,
  positions: Float32Array | null,
): { positions: Float32Array | null; library: SpatialLibrary | null } {
  const spatial = tissueSpatial(embedding, dataset);
  const wanted = useTissueViewStore((s) => s.library);
  const library = spatial ? activeLibrary(spatial, wanted) : null;
  const several = (spatial?.libraries.length ?? 0) > 1;
  const datasetId = dataset?.id ?? null;
  const column = spatial?.library_column ?? null;

  const key = `${datasetId}:${column}`;
  const [codes, setCodes] = useState<{ key: string; codes: ArrayLike<number> } | null>(null);
  useEffect(() => {
    if (!several || !datasetId || !column) return;
    let cancelled = false;
    fetchSpatialCoords(datasetId, column)
      .then((buf) => {
        const v = decodeArrowBuffer(buf)[column];
        if (!cancelled && v) setCodes({ key, codes: v });
      })
      .catch((err) => console.error("Could not read the tissue sections:", err));
    return () => {
      cancelled = true;
    };
  }, [several, datasetId, column, key]);

  const values = dataset?.obs_columns.find((c) => c.name === column)?.values;
  const shown = useMemo(() => {
    if (!several || !positions) return positions;
    if (!codes || codes.key !== key || !library) return null;
    const code = values?.indexOf(library.id) ?? -1;
    return maskToLibrary(positions, codes.codes, code);
  }, [several, positions, codes, key, library, values]);

  return { positions: spatial ? shown : positions, library };
}
