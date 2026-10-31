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
 *  large dataset, no counts). Fetched once markers are there (`ready`); not fetched at all on a
 *  large dataset (`large` set: the backend refuses with a 422, which was a console error on every
 *  Unified View open). */
export function useAmbientFlags(
  datasetId: string | null | undefined,
  column: string | undefined,
  ready: unknown,
  large?: unknown,
) {
  const [ambient, setAmbient] = useState<AmbientResponse | null>(null);
  useEffect(() => {
    setAmbient(null);
    if (!datasetId || !ready || large) return;
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
  }, [datasetId, column, ready, large]);
  const flags = useMemo(() => flagIndex(ambient), [ambient]);
  return { flags, approximate: ambient?.markers?.approximate ?? false };
}
