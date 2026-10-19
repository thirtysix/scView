import { useEffect, useState } from "react";
import { getSystemMemory, type SystemMemory } from "@/api/system";

const POLL_MS = 15_000;
const GB = 1024 ** 3;

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

/** The backend's memory use against its budget, at the bottom of the sidebar. */
export function MemoryIndicator({ collapsed }: { collapsed: boolean }) {
  const [mem, setMem] = useState<SystemMemory | null>(null);

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

  const share = memoryShare(mem);
  if (!mem || share == null) return null;
  const colour = share >= 0.9 ? "bg-red-500" : share >= 0.75 ? "bg-amber-400" : "bg-emerald-400";
  const label = `${gb(mem.used_bytes!)} / ${gb(mem.limit_bytes!)} GB`;

  return (
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
  );
}
