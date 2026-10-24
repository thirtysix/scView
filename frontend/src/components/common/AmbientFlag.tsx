import { useEffect, useMemo, useState } from "react";

import { getAmbient, type AmbientMarker, type AmbientResponse } from "@/api/ambient";
import { flagIndex, flagTitle } from "@/lib/ambient";

/** A small "ambient" mark beside a marker gene whose signal outside its cluster looks
 *  like ambient RNA; the tooltip says why. */
export function AmbientFlag({ flag, group, approximate }: { flag?: AmbientMarker; group: string; approximate: boolean }) {
  if (!flag) return null;
  return (
    <span
      title={flagTitle(flag, group, approximate)}
      data-testid="ambient-marker-flag"
      className="rounded border border-sky-100 bg-sky-50 px-1 text-[9px] font-medium leading-4 text-sky-700"
    >
      ambient{approximate ? "?" : ""}
    </span>
  );
}

/** The column's ambient-looking top markers (backend core/ambient.py), for the marker
 *  tables: group -> gene -> flag. Optional: no flags when they can't be worked out (a
 *  large dataset, no counts). Fetched once markers are there (`ready`). */
export function useAmbientFlags(datasetId: string | null | undefined, column: string | undefined, ready: unknown) {
  const [ambient, setAmbient] = useState<AmbientResponse | null>(null);
  useEffect(() => {
    setAmbient(null);
    if (!datasetId || !ready) return;
    let cancelled = false;
    getAmbient(datasetId, column || undefined)
      .then((d) => {
        if (!cancelled) setAmbient(d ?? null);
      })
      .catch(() => {
        /* no flags */
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, column, ready]);
  const flags = useMemo(() => flagIndex(ambient), [ambient]);
  return { flags, approximate: ambient?.markers?.approximate ?? false };
}
