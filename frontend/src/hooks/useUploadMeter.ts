import { useCallback, useEffect, useRef, useState } from "react";
import { addSample, rateOf, secondsLeft, type Sample, type UploadProgress } from "@/lib/uploadProgress";

/** How often the card redraws while bytes go out (progress events come many times a second). */
const DRAW_MS = 250;
/** A reading taken this often with no progress event, so a stalled link shows a falling speed. */
const TICK_MS = 1000;

export interface UploadMeter {
  progress: UploadProgress | null;
  /** Bytes per second, null for the first seconds. */
  rate: number | null;
  /** Seconds left at that speed, null while unknown. */
  left: number | null;
  /** Start a new upload (clears the last one's readings). */
  start: () => void;
  /** Pass to the upload call's `onProgress`. */
  onProgress: (p: UploadProgress) => void;
  /** Stop reading (the upload ended). */
  stop: () => void;
}

/** Progress, speed and time left of one upload at a time, redrawn at most four times a second. */
export function useUploadMeter(now: () => number = Date.now): UploadMeter {
  const samples = useRef<Sample[]>([]);
  const latest = useRef<UploadProgress | null>(null);
  const lastDraw = useRef(0);
  const [active, setActive] = useState(false);
  const [view, setView] = useState<{ progress: UploadProgress | null; rate: number | null; left: number | null }>({
    progress: null,
    rate: null,
    left: null,
  });

  const draw = useCallback(() => {
    const p = latest.current;
    const rate = rateOf(samples.current);
    lastDraw.current = now();
    setView({ progress: p, rate, left: p ? secondsLeft(p.loaded, p.total, rate) : null });
  }, [now]);

  const onProgress = useCallback(
    (p: UploadProgress) => {
      latest.current = p;
      samples.current = addSample(samples.current, { t: now(), loaded: p.loaded });
      if (p.loaded >= p.total || now() - lastDraw.current >= DRAW_MS) draw();
    },
    [draw, now],
  );

  const start = useCallback(() => {
    samples.current = [];
    latest.current = null;
    lastDraw.current = 0;
    setView({ progress: null, rate: null, left: null });
    setActive(true);
  }, []);

  const stop = useCallback(() => setActive(false), []);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      const p = latest.current;
      if (!p || p.loaded >= p.total) return;
      samples.current = addSample(samples.current, { t: now(), loaded: p.loaded });
      draw();
    }, TICK_MS);
    return () => clearInterval(id);
  }, [active, draw, now]);

  return { ...view, start, onProgress, stop };
}
