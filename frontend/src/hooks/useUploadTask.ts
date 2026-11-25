import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort, type UploadOptions } from "@/api/upload";
import { useUploadMeter } from "./useUploadMeter";
import type { UploadProgress } from "@/lib/uploadProgress";

export interface UploadTask {
  /** What is going up ("sample.loom", "2 files"), null when nothing is. */
  name: string | null;
  /** The name of the last upload when it was cancelled (cleared by the next one). */
  cancelled: string | null;
  progress: UploadProgress | null;
  rate: number | null;
  left: number | null;
  /** Send one upload (or one set sent in turn) with progress and a signal; one at a time. */
  run: <T>(name: string, send: (opts: UploadOptions) => Promise<T>) => Promise<T>;
  /** Stop the upload going now (it rejects with an AbortError). */
  cancel: () => void;
}

/**
 * One upload at a time from a card: its meter (percent, speed, time left), its Cancel, and the
 * lock (`name` is set while it runs; a second `run` meanwhile is refused). `abortOnUnmount`: a card
 * that goes away stops its upload too.
 */
export function useUploadTask({ abortOnUnmount = false }: { abortOnUnmount?: boolean } = {}): UploadTask {
  const meter = useUploadMeter();
  const { start, stop, onProgress } = meter;
  const abort = useRef<AbortController | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState<string | null>(null);

  const run = useCallback(
    async <T,>(label: string, send: (opts: UploadOptions) => Promise<T>): Promise<T> => {
      if (abort.current) throw new Error("One upload at a time: wait for this one to end.");
      const ctl = new AbortController();
      abort.current = ctl;
      start();
      setName(label);
      setCancelled(null);
      try {
        return await send({ onProgress, signal: ctl.signal });
      } catch (e) {
        if (isAbort(e)) setCancelled(label);
        throw e;
      } finally {
        abort.current = null;
        stop();
        setName(null);
      }
    },
    [start, stop, onProgress],
  );

  const cancel = useCallback(() => abort.current?.abort(), []);

  useEffect(() => {
    if (!abortOnUnmount) return;
    return () => abort.current?.abort();
  }, [abortOnUnmount]);

  return { name, cancelled, progress: meter.progress, rate: meter.rate, left: meter.left, run, cancel };
}
