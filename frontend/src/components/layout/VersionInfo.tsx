import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowUpCircle, X } from "lucide-react";
import { getSystemVersion, type SystemVersion } from "@/api/system";
import { useDialogFocus } from "@/hooks/useDialogFocus";

// The backend asks GitHub at most once a day; a page left open still hears of a release.
const REFRESH_MS = 6 * 3600_000;

/** The release's own ZIP ("Source code (zip)" on the release page, which novices do not spot). */
export function zipUrl(releaseUrl: string): string | null {
  const m = releaseUrl.match(/^(https:\/\/github\.com\/[^/]+\/[^/]+)\/releases\/tag\/([^/?#]+)$/);
  return m ? `${m[1]}/archive/refs/tags/${m[2]}.zip` : null;
}

/** How to update, step by step: the release page alone leaves a novice at a list of changes. */
function UpdateDialog({ v, onClose }: { v: SystemVersion; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useDialogFocus(ref, onClose);
  const zip = zipUrl(v.release_url!);
  const code = "rounded bg-slate-100 px-1 font-mono text-[12px] text-slate-700";
  return createPortal(
    <div
      className="fixed inset-0 z-[1100] flex items-center justify-center bg-slate-900/40 p-4"
      role="presentation"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={`Update to scView ${v.latest}`}
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl bg-white p-5 text-sm text-slate-700 shadow-xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-800">scView {v.latest} is out</h2>
            <p className="text-xs text-slate-500">
              You run {v.version}.{" "}
              <a href={v.release_url!} target="_blank" rel="noopener noreferrer" className="text-sky-700 underline">
                What's new
              </a>
            </p>
          </div>
          <button type="button" aria-label="Close" onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-4 w-4" />
          </button>
        </div>

        <h3 className="mb-1 font-medium text-slate-800">If you downloaded a ZIP</h3>
        <ol className="mb-3 ml-5 list-decimal space-y-1">
          <li>
            {zip ? (
              <a href={zip} className="font-medium text-sky-700 underline" data-testid="update-zip">
                Download scView {v.latest} (ZIP)
              </a>
            ) : (
              <>Download the new ZIP from the release page (Assets, "Source code (zip)")</>
            )}{" "}
            and extract it (Windows: right-click the ZIP, <b>Extract All</b>).
          </li>
          <li>
            If you gave scView an API key, copy the <span className={code}>.env</span> file from your old scView
            folder into the new one. Without it, the start script asks for the key again.
          </li>
          <li>
            Start scView from the <b>new</b> folder, as before: double-click{" "}
            <span className={code}>start-windows.bat</span> on Windows, or run{" "}
            <span className={code}>./start.sh</span> on macOS and Linux. Starting the old folder keeps the old
            version.
          </li>
        </ol>

        <h3 className="mb-1 font-medium text-slate-800">If you used git</h3>
        <p className="mb-3">
          Run <span className={code}>git pull</span> in the scView folder, then start scView as before.
        </p>

        <p className="text-xs text-slate-500">
          Your datasets are kept: they live in Docker, not in the scView folder. The first start downloads the new
          version (about 1 GB), a few minutes. More in the README, under Updating.
        </p>
      </div>
    </div>,
    document.body,
  );
}

/** scView's version at the bottom of the sidebar, with update steps when a newer release is out. */
export function VersionInfo({ collapsed }: { collapsed: boolean }) {
  const [v, setV] = useState<SystemVersion | null>(null);
  const [open, setOpen] = useState(false);

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
  const button = update ? (
    <button
      type="button"
      onClick={() => setOpen(true)}
      title={`scView ${v.latest} is out (this is ${v.version}): how to update`}
      aria-label={`scView ${v.latest} is available: how to update`}
      className="flex items-center gap-1 text-emerald-400 hover:text-emerald-300"
    >
      <ArrowUpCircle className="h-3.5 w-3.5 shrink-0" />
      {!collapsed && <span>{v.latest} available</span>}
    </button>
  ) : null;
  const dialog = open && update ? <UpdateDialog v={v} onClose={() => setOpen(false)} /> : null;

  if (collapsed)
    return button ? (
      <div className="flex justify-center py-1.5 text-[11px]">
        {button}
        {dialog}
      </div>
    ) : null;
  return (
    <div
      className="flex items-center justify-between px-4 pb-2 text-[11px] text-sidebar-foreground/50"
      data-testid="version-info"
    >
      <span title={v.check === "off" ? "Update check off (UPDATE_CHECK=false)" : undefined}>scView {v.version}</span>
      {button}
      {dialog}
    </div>
  );
}
