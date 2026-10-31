import { useEffect, useMemo, useRef, useState } from "react";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { Layer } from "@deck.gl/core";
import { fetchOutlines, fetchTranscripts, type OutlineKind, type TranscriptView } from "@/api/transcripts";
import type { CellOutlines, WorldRect } from "@/lib/cellOutlines";
import { errorMessage } from "@/lib/errorMessage";
import { MAX_RIMMED_DOTS } from "@/lib/transcripts";
import { useTissueViewStore } from "@/stores/tissueViewStore";

/** The chosen outlines of a dataset, kept while the choice stands (17 MB on a Xenium run). */
export function useCellOutlines(datasetId: string | null | undefined, kind: OutlineKind | null) {
  const [state, setState] = useState<{ key: string; outlines: CellOutlines | null; error: string | null } | null>(null);
  const key = datasetId && kind ? `${datasetId}:${kind}` : null;
  useEffect(() => {
    if (!datasetId || !kind || !key) return;
    const ctl = new AbortController();
    setState({ key, outlines: null, error: null });
    fetchOutlines(datasetId, kind, ctl.signal)
      .then((outlines) => setState({ key, outlines, error: null }))
      .catch((e) => {
        if (e?.name !== "AbortError") setState({ key, outlines: null, error: errorMessage(e) });
      });
    return () => ctl.abort();
  }, [datasetId, kind, key]);
  const current = state?.key === key ? state : null;
  return {
    outlines: current?.outlines ?? null,
    loading: !!key && !current?.outlines && !current?.error,
    error: current?.error ?? null,
  };
}

/**
 * Coloured dots of the chosen genes over the tissue, for the view the camera settled on: fetched
 * again per view (a request in flight is dropped), the last answer drawn until the next arrives.
 */
export function useTranscriptDots(datasetId: string | null | undefined, enabled: boolean, rect: WorldRect | null) {
  const genes = useTissueViewStore((s) => s.transcriptGenes);
  const dotSize = useTissueViewStore((s) => s.transcriptDotSize);
  const setShown = useTissueViewStore((s) => s.setTranscriptShown);
  const [view, setView] = useState<{ key: string; data: TranscriptView } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const genesKey = genes.join(",");
  const on = enabled && !!datasetId && genes.length > 0;
  const box = rect ? { x0: rect.x0, x1: rect.x1, y0: -rect.y1, y1: -rect.y0 } : null;
  const boxKey = box ? `${box.x0},${box.x1},${box.y0},${box.y1}` : "";
  const boxRef = useRef(box);
  boxRef.current = box;
  const key = `${datasetId}:${genesKey}`;

  useEffect(() => {
    if (!on || !datasetId || !boxRef.current) {
      setView(null);
      setShown(null);
      setError(null);
      return;
    }
    const ctl = new AbortController();
    fetchTranscripts(datasetId, genesKey.split(","), boxRef.current, ctl.signal)
      .then((data) => {
        setView({ key, data });
        setShown({ shown: data.n, total: data.total, perGene: data.perGene });
        setError(null);
      })
      .catch((e) => {
        if (e?.name !== "AbortError") setError(errorMessage(e));
      });
    return () => ctl.abort();
  }, [on, datasetId, genesKey, boxKey, key, setShown]);

  const current = view?.key === key ? view.data : null;
  const layers = useMemo<Layer[]>(() => {
    if (!on || !current || current.n === 0) return [];
    return [
      new ScatterplotLayer({
        id: "transcript-dots",
        data: {
          length: current.n,
          attributes: {
            getPosition: { value: current.positions, size: 2 },
            getFillColor: { value: current.colors, size: 4 },
          },
        },
        getRadius: dotSize,
        radiusUnits: "pixels" as const,
        radiusMinPixels: 0.5,
        stroked: current.n <= MAX_RIMMED_DOTS,
        getLineColor: [0, 0, 0, 140],
        lineWidthUnits: "pixels" as const,
        getLineWidth: 0.5,
        pickable: false,
        updateTriggers: { getRadius: [dotSize] },
      }),
    ];
  }, [on, current, dotSize]);
  return { layers, error };
}
