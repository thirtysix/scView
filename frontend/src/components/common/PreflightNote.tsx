import { useEffect, useState } from "react";
import { AlertTriangle, Box, Gauge, Timer } from "lucide-react";

import {
  getIngestPreflight,
  getRunPreflight,
  type IngestPreflight,
  type LeidenChoice,
  type PreflightVerdict,
  type Umap3dChoice,
} from "@/api/preflight";
import { FAST_LEIDEN, FULL_LEIDEN } from "@/lib/fastLeiden";
import type { Umap3dSetting } from "@/lib/umap3d";
import { FastLeidenInfo } from "./FastLeidenInfo";
import { InfoTip } from "./InfoTip";

const GB = 1024 ** 3;
const gb = (bytes: number) => (bytes / GB >= 10 ? (bytes / GB).toFixed(0) : (bytes / GB).toFixed(1));

const LEVEL_STYLE: Record<PreflightVerdict["level"], string> = {
  fits: "text-slate-500",
  tight: "text-amber-700",
  wont_fit: "rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-amber-800",
  unknown: "text-slate-400",
};

/** The tooltip: what the estimate adds up, against what scView has, and how far to trust it. */
function Breakdown({ verdict }: { verdict: PreflightVerdict }) {
  const v = verdict;
  return (
    <span className="block space-y-1">
      {v.breakdown.map((b) => (
        <span key={b.label} className="flex justify-between gap-2">
          <span>{b.label}</span>
          <span className="tabular-nums">{gb(b.bytes)} GB</span>
        </span>
      ))}
      {v.available_bytes != null && v.limit_bytes != null && (
        <span className="block">
          Free now: {gb(v.available_bytes)} GB of {gb(v.limit_bytes)} GB
          {v.limited ? " (the SCVIEW_MEMORY_GB limit)" : " (no limit is set: the memory this machine gives scView)"};
          file cache the system can drop is counted as free.
        </span>
      )}
      {v.cached.count > 0 && (
        <span className="block">
          Other datasets held in memory: {v.cached.count}, about {gb(v.cached.bytes)} GB
          {v.cached.frees_enough ? "; closing them would free enough." : "."}
        </span>
      )}
      <span className="block">
        Memory is estimated from the data's shape and storage (cells, genes, stored values, read
        from the file's header) with coefficients measured on test datasets, where it came within{" "}
        {Math.round(v.error_band * 100)}% of the measured peak.
      </span>
      <span className="block">
        Time is rough: measured with 2 CPUs (more CPUs barely speed the steps up; fewer slow them
        down). In a virtual machine
        (Docker Desktop or Podman on Windows) steps can take much longer: a Windows test VM ran about
        10 times slower than its host.
      </span>
      {v.notes.map((n) => (
        <span key={n} className="block">
          {n}
        </span>
      ))}
      {v.level === "tight" &&
        v.suggestions.map((s) => (
          <span key={s} className="block">
            If it does not fit: {s}
          </span>
        ))}
    </span>
  );
}

/** One sentence on whether a load or run fits in memory, with the breakdown in a tooltip.
 *  It warns, never blocks: a won't-fit verdict lists what to change. */
export function PreflightNote({ verdict, label }: { verdict: PreflightVerdict; label?: string }) {
  const Icon = verdict.level === "fits" || verdict.level === "unknown" ? Gauge : AlertTriangle;
  return (
    <div
      className={`text-[11px] leading-snug ${LEVEL_STYLE[verdict.level]}`}
      data-level={verdict.level}
      role={verdict.level === "wont_fit" ? "alert" : undefined}
    >
      <p className="flex items-start gap-1.5">
        <Icon className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>
          {label && <span className="font-medium">{label}: </span>}
          {verdict.sentence}{" "}
          <InfoTip width={320} align="right" label="How this estimate is made">
            <Breakdown verdict={verdict} />
          </InfoTip>
        </span>
      </p>
      {verdict.level === "wont_fit" && verdict.suggestions.length > 0 && (
        <ul className="mt-1 list-disc space-y-0.5 pl-8">
          {verdict.suggestions.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** When clustering with Leiden's full optimisation would take over 10 minutes: the offer of
 *  fast Leiden, one click to opt in (and back), with what it changes in a tooltip. Nothing
 *  for a run that is quick enough, unless fast Leiden is already chosen. */
export function FastLeidenOffer({
  choice,
  onChange,
}: {
  choice: LeidenChoice;
  onChange?: (iterations: number) => void;
}) {
  const fast = choice.iterations === FAST_LEIDEN;
  if (!choice.offer_fast) return null;
  const cells = choice.n_cells.toLocaleString();
  return (
    <div
      className="mt-1 rounded-md border border-sky-200 bg-sky-50 px-2.5 py-1.5 text-[11px] leading-snug text-sky-900"
      data-testid="fast-leiden-offer"
    >
      <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
        <Timer className="h-3.5 w-3.5 shrink-0" aria-hidden />
        {fast ? (
          <span>
            Clustering {cells} cells with fast Leiden: {choice.fast_text} (the full optimisation:{" "}
            {choice.full_text}).
          </span>
        ) : (
          <span>
            Clustering {cells} cells with Leiden&apos;s full optimisation may take {choice.full_text}. Fast Leiden:{" "}
            {choice.fast_text}.
          </span>
        )}
        {onChange && (
          <button
            type="button"
            onClick={() => onChange(fast ? FULL_LEIDEN : FAST_LEIDEN)}
            className={
              fast
                ? "text-sky-800 underline hover:text-sky-950"
                : "rounded border border-sky-300 bg-white px-1.5 py-0.5 font-medium text-sky-800 hover:bg-sky-100"
            }
          >
            {fast ? "Use the full optimisation" : "Use fast Leiden"}
          </button>
        )}
        {/* Opens rightwards: the icon sits near the card's left edge, beside the sidebar */}
        <FastLeidenInfo />
      </p>
    </div>
  );
}

/** When the run leaves out the 3D UMAP (large-dataset mode) or it would take long: what it
 *  costs, one click to add it (or leave it out again). */
export function Umap3dNote({
  choice,
  onChange,
}: {
  choice: Umap3dChoice;
  onChange?: (setting: Umap3dSetting) => void;
}) {
  if (!choice.offer) return null;
  const cells = choice.n_cells.toLocaleString();
  return (
    <div
      className="mt-1 rounded-md border border-sky-200 bg-sky-50 px-2.5 py-1.5 text-[11px] leading-snug text-sky-900"
      data-testid="umap-3d-note"
    >
      <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
        <Box className="h-3.5 w-3.5 shrink-0" aria-hidden />
        {choice.on ? (
          <span>
            The 3D UMAP of {cells} cells adds {choice.extra_text} to the run.
          </span>
        ) : (
          <span>
            {choice.setting === "auto" ? "Large dataset: no 3D UMAP in this run" : "No 3D UMAP in this run"}{" "}
            (the 3D view needs one). Adding it would take {choice.extra_text} more.
          </span>
        )}
        {onChange && (
          <button
            type="button"
            onClick={() => onChange(choice.on ? (choice.default_on ? "off" : "auto") : "on")}
            className={
              choice.on
                ? "text-sky-800 underline hover:text-sky-950"
                : "rounded border border-sky-300 bg-white px-1.5 py-0.5 font-medium text-sky-800 hover:bg-sky-100"
            }
          >
            {choice.on ? "Leave it out" : "Add the 3D UMAP"}
          </button>
        )}
      </p>
    </div>
  );
}

/** The run estimate for the selected steps, next to the pipeline Run button. Refetched
 *  (debounced) when the steps, the parameters it depends on, or `refreshKey` change.
 *  `onLeidenIterations`: the run's clustering setting, for the fast-Leiden offer's button;
 *  `onUmap3d`: the UMAP step's 3D setting, for the 3D UMAP note's button. */
export function PreflightRunNote({
  datasetId,
  steps,
  params,
  refreshKey = 0,
  onLeidenIterations,
  onUmap3d,
}: {
  datasetId: string;
  steps: string[];
  params: Record<string, unknown> | null;
  refreshKey?: number;
  onLeidenIterations?: (iterations: number) => void;
  onUmap3d?: (setting: Umap3dSetting) => void;
}) {
  const [verdict, setVerdict] = useState<PreflightVerdict | null>(null);
  const [leiden, setLeiden] = useState<LeidenChoice | null>(null);
  const [umap3d, setUmap3d] = useState<Umap3dChoice | null>(null);
  const key = steps.join(",");
  const paramKey = JSON.stringify(params ?? {});

  useEffect(() => {
    if (!key) {
      setVerdict(null);
      setLeiden(null);
      setUmap3d(null);
      return;
    }
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      getRunPreflight(datasetId, key.split(","), JSON.parse(paramKey), ctl.signal)
        .then((r) => {
          setVerdict(r.verdict);
          setLeiden(r.leiden ?? null);
          setUmap3d(r.umap_3d ?? null);
        })
        .catch(() => {
          /* advisory only: an older backend or a restart leaves no note */
          if (!ctl.signal.aborted) {
            setVerdict(null);
            setLeiden(null);
            setUmap3d(null);
          }
        });
    }, 300);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [datasetId, key, paramKey, refreshKey]);

  if (!verdict) return null;
  return (
    <>
      <PreflightNote verdict={verdict} />
      {leiden && <FastLeidenOffer choice={leiden} onChange={onLeidenIterations} />}
      {umap3d && <Umap3dNote choice={umap3d} onChange={onUmap3d} />}
    </>
  );
}

/** One note per staged sample (and one for the merge) before an ingest commit. */
export function PreflightIngestNotes({
  sessionId,
  refreshKey = 0,
}: {
  sessionId: string;
  /** A change refetches (the staged files changed) */
  refreshKey?: number | string;
}) {
  const [data, setData] = useState<IngestPreflight | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    getIngestPreflight(sessionId, ctl.signal)
      .then(setData)
      .catch(() => {
        if (!ctl.signal.aborted) setData(null);
      });
    return () => ctl.abort();
  }, [sessionId, refreshKey]);

  if (!data || data.units.length === 0) return null;
  const many = data.units.length > 1;
  return (
    <div className="space-y-1.5" aria-label="Memory estimate">
      {data.units.map((u) => (
        <PreflightNote key={u.label} verdict={u.verdict} label={many ? u.label : undefined} />
      ))}
      {data.merge && <PreflightNote verdict={data.merge} label="Merging them" />}
    </div>
  );
}
