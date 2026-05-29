import { useQuery } from "@tanstack/react-query";
import { useState, useEffect, useRef, useCallback } from "react";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";

import ArrowWorker from "@/workers/arrowDecoder.worker?worker";

interface EmbeddingData {
  positions: Float32Array | null;
  colors: Float32Array | Int32Array | null;
  numCells: number;
  colorColumnName: string | null;
  dimensions: 2 | 3;
}

interface WorkerSuccessMessage {
  type: "success";
  result: Record<string, { data: ArrayBuffer; type: string; length: number }>;
  numRows: number;
}

interface WorkerErrorMessage {
  type: "error";
  error: string;
}

type WorkerMessage = WorkerSuccessMessage | WorkerErrorMessage;

/**
 * Hook that fetches embedding coordinates via Arrow IPC and decodes them in a
 * Web Worker, returning typed arrays ready for deck.gl rendering.
 *
 * Re-fetches automatically when the active embedding or colorBy column changes.
 */
export function useEmbedding() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const embedding = useSettingsStore((s) => s.embedding);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);

  const workerRef = useRef<Worker | null>(null);

  const [embeddingData, setEmbeddingData] = useState<EmbeddingData>({
    positions: null,
    colors: null,
    numCells: 0,
    colorColumnName: null,
    dimensions: 2,
  });
  const [decodeError, setDecodeError] = useState<string | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);

  // Create the worker once and reuse it
  useEffect(() => {
    workerRef.current = new ArrowWorker();
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  // Fetch the binary Arrow IPC buffer
  const {
    data: arrowBuffer,
    isLoading: isFetching,
    error: fetchError,
  } = useQuery({
    queryKey: ["embedding", datasetId, embedding, colorBy],
    queryFn: () => fetchEmbeddingBinary(datasetId!, embedding, colorBy || undefined),
    enabled: !!datasetId && !!embedding && embedding !== "" && !pipelineRunning,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 60 * 1000, // 1 minute — large binary buffers shouldn't linger
  });

  // Decode callback — sends the buffer to the worker
  const decode = useCallback(
    (buffer: ArrayBuffer) => {
      const worker = workerRef.current;
      if (!worker) return;

      setIsDecoding(true);
      setDecodeError(null);

      const handler = (e: MessageEvent<WorkerMessage>) => {
        worker.removeEventListener("message", handler);
        setIsDecoding(false);

        if (e.data.type === "error") {
          setDecodeError(e.data.error);
          return;
        }

        const { result, numRows } = e.data;

        // Build interleaved positions Float32Array
        // 2D: [x0, y0, x1, y1, ...]  3D: [x0, y0, z0, x1, y1, z1, ...]
        const xCol = result["x"] ?? result[Object.keys(result)[0]!];
        const yCol = result["y"] ?? result[Object.keys(result)[1]!];
        const zCol = result["z"];
        const is3D = zCol != null;

        let positions: Float32Array | null = null;
        if (xCol && yCol) {
          const xData = new Float32Array(xCol.data);
          const yData = new Float32Array(yCol.data);
          if (is3D) {
            const zData = new Float32Array(zCol.data);
            positions = new Float32Array(numRows * 3);
            for (let i = 0; i < numRows; i++) {
              positions[i * 3] = xData[i]!;
              positions[i * 3 + 1] = yData[i]!;
              positions[i * 3 + 2] = zData[i]!;
            }
          } else {
            positions = new Float32Array(numRows * 2);
            for (let i = 0; i < numRows; i++) {
              positions[i * 2] = xData[i]!;
              positions[i * 2 + 1] = yData[i]!;
            }
          }
        }

        // Color column is the third column (if present), typically named after
        // the colorBy parameter
        let colors: Float32Array | Int32Array | null = null;
        let colorColName: string | null = null;
        const colorKeys = Object.keys(result).filter(
          (k) => k !== "x" && k !== "y" && k !== "z",
        );
        if (colorKeys.length > 0) {
          const key = colorKeys[0]!;
          colorColName = key;
          const colResult = result[key]!;
          if (colResult.type === "int32") {
            colors = new Int32Array(colResult.data);
          } else {
            colors = new Float32Array(colResult.data);
          }
        }

        setEmbeddingData({
          positions,
          colors,
          numCells: numRows,
          colorColumnName: colorColName,
          dimensions: is3D ? 3 : 2,
        });
      };

      worker.addEventListener("message", handler);

      // Transfer the buffer to the worker (zero-copy)
      worker.postMessage({ buffer }, [buffer]);
    },
    [],
  );

  // Trigger decode when a new buffer arrives
  useEffect(() => {
    if (arrowBuffer) {
      // We need to copy the buffer because React Query may cache the original
      const copy = arrowBuffer.slice(0);
      decode(copy);
    }
  }, [arrowBuffer, decode]);

  return {
    positions: embeddingData.positions,
    colors: embeddingData.colors,
    numCells: embeddingData.numCells,
    colorColumnName: embeddingData.colorColumnName,
    dimensions: embeddingData.dimensions,
    isLoading: isFetching || isDecoding,
    error: fetchError ? String(fetchError) : decodeError,
  };
}
