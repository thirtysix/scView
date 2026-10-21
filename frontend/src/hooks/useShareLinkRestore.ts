import { useCallback, useEffect, useRef, useState } from "react";
import { clearShareHash, hasShareHash, openShareHash, type ShareOutcome } from "@/lib/shareUrl";

/**
 * Opens a share link (lib/shareUrl.ts): on first load with a `#view=` fragment,
 * and whenever one is pasted into the address bar of an open tab (hashchange).
 * Once the view is restored the fragment is taken off the address, so a reload
 * does not apply it again; after a server error it stays, so a reload retries.
 * The reload restore (useSessionRestore) stands aside while a link is open.
 */
export function useShareLinkRestore(): { outcome: ShareOutcome | null; busy: boolean; dismiss: () => void } {
  const [outcome, setOutcome] = useState<ShareOutcome | null>(null);
  const [busy, setBusy] = useState(false);
  // The fragment being opened, so StrictMode's second effect run and a repeated
  // hashchange don't open it twice; the latest one waiting behind it.
  const current = useRef<string | null>(null);
  const waiting = useRef<string | null>(null);

  useEffect(() => {
    const open = async (hash: string) => {
      if (!hasShareHash(hash) || hash === current.current) return;
      if (current.current != null) {
        waiting.current = hash;
        return;
      }
      current.current = hash;
      setBusy(true);
      try {
        let next: string | null = hash;
        while (next != null) {
          const result = await openShareHash(next);
          if (result) setOutcome(result);
          // A newer link pasted meanwhile replaced the fragment: leave that one.
          if (!result?.retry && window.location.hash === next) clearShareHash();
          next = waiting.current;
          waiting.current = null;
          current.current = next;
        }
      } finally {
        current.current = null;
        setBusy(false);
      }
    };
    void open(window.location.hash);
    const onHashChange = () => void open(window.location.hash);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const dismiss = useCallback(() => setOutcome(null), []);
  return { outcome, busy, dismiss };
}
