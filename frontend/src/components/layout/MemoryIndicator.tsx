import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { getSystemMemory, type SystemMemory } from "@/api/system";

const POLL_MS = 15_000;
const GB = 1024 ** 3;
const SEEN_KEY = "scview.uncleanRestartSeen";

const gb = (bytes: number) => (bytes / GB >= 10 ? (bytes / GB).toFixed(0) : (bytes / GB).toFixed(1));

/** Share of the budget in use, or null when either side is unknown. */
export function memoryShare(m: SystemMemory | null): number | null {
  if (!m || !m.limit_bytes || m.used_bytes == null) return null;
  return Math.min(1, m.used_bytes / m.limit_bytes);
}

export function memoryTitle(m: SystemMemory): string {
  if (m.limited && m.limit_bytes)
    return (
      `scView may use ${gb(m.limit_bytes)} GB (SCVIEW_MEMORY_GB in the .env file in the scView ` +
      "folder; restart scView after changing it)."
    );
  if (m.limit_bytes)
    return (
      `No memory limit is set, so scView may use all ${gb(m.limit_bytes)} GB this machine gives it. ` +
      "Set SCVIEW_MEMORY_GB in the .env file in the scView folder (the start scripts suggest a value)."
    );
  return "scView could not read its memory limit.";
}

/** What to tell the user after the backend was killed and restarted, or null. */
export function restartNotice(m: SystemMemory | null): string | null {
  const r = m?.unclean_restart;
  if (!r) return null;
  const at = new Date(r.restarted_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const lost = "Anything not yet saved, such as a run in progress, was lost.";
  if (m!.limited && m!.limit_bytes)
    return (
      `scView stopped at ${at} and restarted, most likely because it ran out of memory: it may use ` +
      `${gb(m!.limit_bytes)} GB. ${lost} To allow more, raise SCVIEW_MEMORY_GB in the .env file in ` +
      "the scView folder and start scView again."
    );
  return `scView stopped at ${at} and restarted, possibly because the computer ran out of memory. ${lost}`;
}

function readSeen(): string | null {
  try {
    return sessionStorage.getItem(SEEN_KEY);
  } catch {
    return null;
  }
}

/** The backend's memory use against its budget, at the bottom of the sidebar, and a notice
 *  at the top of the page after the backend was killed and restarted. */
export function MemoryIndicator({ collapsed }: { collapsed: boolean }) {
  const [mem, setMem] = useState<SystemMemory | null>(null);
  const [seen, setSeen] = useState<string | null>(readSeen);

  useEffect(() => {
    let live = true;
    const load = () =>
      getSystemMemory()
        .then((m) => live && setMem(m))
        .catch(() => {
          /* an older backend, or it is restarting: keep the last reading */
        });
    load();
    const timer = setInterval(load, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  const notice = restartNotice(mem);
  const restartedAt = mem?.unclean_restart?.restarted_at ?? null;
  const dismiss = () => {
    try {
      sessionStorage.setItem(SEEN_KEY, restartedAt!);
    } catch {
      /* storage blocked: dismissed for this page only */
    }
    setSeen(restartedAt);
  };
  const banner = notice && restartedAt !== seen && (
    <div
      role="alert"
      className="fixed left-1/2 top-3 z-50 flex w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 gap-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 shadow-lg"
    >
      <p className="flex-1">{notice}</p>
      <button onClick={dismiss} aria-label="Dismiss" className="h-fit text-amber-700 hover:text-amber-900">
        <X className="h-4 w-4" />
      </button>
    </div>
  );

  const share = memoryShare(mem);
  if (!mem || share == null) return banner || null;
  const colour = share >= 0.9 ? "bg-red-500" : share >= 0.75 ? "bg-amber-400" : "bg-emerald-400";
  const label = `${gb(mem.used_bytes!)} / ${gb(mem.limit_bytes!)} GB`;

  return (
    <>
      {banner}
      <div className="border-t border-white/10 px-4 py-2" title={memoryTitle(mem)} aria-label={`Memory ${label}`}>
        {!collapsed && (
          <div className="mb-1 flex items-center justify-between text-[11px] text-sidebar-foreground/60">
            <span>Memory{mem.limited ? "" : " (no limit)"}</span>
            <span>{label}</span>
          </div>
        )}
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div className={`h-full ${colour}`} style={{ width: `${Math.max(2, share * 100)}%` }} />
        </div>
      </div>
    </>
  );
}
