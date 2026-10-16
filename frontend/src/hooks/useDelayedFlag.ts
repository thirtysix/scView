import { useEffect, useState } from "react";

/**
 * True once `active` has stayed true for `delayMs`; false again as soon as it
 * drops. Tells a quick fetch (stored results) from a slow one (computed on
 * demand) without the server saying which it is.
 */
export function useDelayedFlag(active: boolean, delayMs: number): boolean {
  const [late, setLate] = useState(false);
  useEffect(() => {
    if (!active) {
      setLate(false);
      return;
    }
    const t = setTimeout(() => setLate(true), delayMs);
    return () => clearTimeout(t);
  }, [active, delayMs]);
  return active && late;
}
