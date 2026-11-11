import { useEffect, useState } from "react";
import { ArrowUpCircle } from "lucide-react";
import { getSystemVersion, type SystemVersion } from "@/api/system";

// The backend asks GitHub at most once a day; a page left open still hears of a release.
const REFRESH_MS = 6 * 3600_000;

export const UPDATE_HOW =
  "To update: download the new ZIP and start scView from it (copy your .env across), or run " +
  "git pull. Your datasets are kept. See Updating in the README.";

/** scView's version at the bottom of the sidebar, with a link to a newer release when one is out. */
export function VersionInfo({ collapsed }: { collapsed: boolean }) {
  const [v, setV] = useState<SystemVersion | null>(null);

  useEffect(() => {
    let live = true;
    const load = async () => {
      try {
        const r = await getSystemVersion();
        if (live) setV(r);
      } catch {
        /* an older backend, or it is restarting: keep the last answer */
      }
    };
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  if (!v || v.version === "unknown") return null;
  const update = v.update_available && v.latest && v.release_url;
  const link = update ? (
    <a
      href={v.release_url!}
      target="_blank"
      rel="noopener noreferrer"
      title={`scView ${v.latest} is out (this is ${v.version}). ${UPDATE_HOW}`}
      aria-label={`scView ${v.latest} is available`}
      className="flex items-center gap-1 text-emerald-400 hover:text-emerald-300"
    >
      <ArrowUpCircle className="h-3.5 w-3.5 shrink-0" />
      {!collapsed && <span>{v.latest} available</span>}
    </a>
  ) : null;

  if (collapsed) return link ? <div className="flex justify-center py-1.5 text-[11px]">{link}</div> : null;
  return (
    <div
      className="flex items-center justify-between px-4 pb-2 text-[11px] text-sidebar-foreground/50"
      data-testid="version-info"
    >
      <span title={v.check === "off" ? "Update check off (UPDATE_CHECK=false)" : undefined}>scView {v.version}</span>
      {link}
    </div>
  );
}
