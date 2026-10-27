import { useEffect, useRef } from "react";
import type { ObsColumnInfo } from "@/api/types";

/** A sample column asked for up front (the co-pilot's open_compare_tab), applied once:
 *  after the first condition is set, since a new condition resets the sample picker.
 *  Call it AFTER that reset effect so it runs later in the same commit. A column the
 *  picker does not offer is ignored (the backend's detected default stays). */
export function usePrefillSample(
  sample: string | null | undefined,
  conditionKey: string,
  sampleOptions: ObsColumnInfo[],
  setSampleKey: (v: string) => void,
): void {
  const pending = useRef(sample ?? "");
  useEffect(() => {
    const s = pending.current;
    if (!s || !conditionKey) return;
    pending.current = "";
    if (sampleOptions.some((c) => c.name === s)) setSampleKey(s);
  }, [conditionKey, sampleOptions, setSampleKey]);
}

/** `preferred` when the picker offers it, else `fallback`. */
export function pickOffered(preferred: string | null | undefined, offered: string[], fallback: string): string {
  return preferred && offered.includes(preferred) ? preferred : fallback;
}
