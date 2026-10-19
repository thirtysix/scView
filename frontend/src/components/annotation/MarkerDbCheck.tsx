import { useEffect, useRef, useState } from "react";
import { Database, Loader2, RotateCcw } from "lucide-react";
import {
  getMarkerDbStatus,
  runMarkerDbCheck,
  type MarkerDbLabel,
  type MarkerDbResult,
  type MarkerDbSource,
  type MarkerDbStatus,
  type MarkerDbVerdict,
} from "@/api/annotation";
import { formatDate, formatPValue, prettyObsLabel } from "@/lib/formatting";

const ROWS = 12; // more behind "show all"

const VERDICT_STYLE: Record<MarkerDbVerdict, string> = {
  supported: "bg-emerald-50 text-emerald-700 border-emerald-200",
  no_support: "bg-amber-50 text-amber-800 border-amber-200",
  too_few: "bg-slate-50 text-slate-500 border-slate-200",
  not_in_db: "bg-slate-50 text-slate-500 border-slate-200",
  no_markers: "bg-slate-50 text-slate-500 border-slate-200",
};

const VERDICT_TITLE: Record<MarkerDbVerdict, string> = {
  supported: "Some of the label's database markers are among this group's top markers",
  no_support:
    "None of the label's database markers is among this group's top 25 markers: worth a look, not proof the label is wrong",
  too_few: "The matched database type has fewer than 5 genes measured here: too few to judge",
  not_in_db: "No PanglaoDB or CellMarker 2.0 cell type matches this name",
  no_markers: "The group has no marker genes that pass the filters",
};

/** The server's reason from an apiFetch error ("API error 503: {detail}"). */
function errorText(e: unknown): string {
  const body = (e instanceof Error ? e.message : String(e)).replace(/^API error \d+: /, "");
  try {
    return String(JSON.parse(body).detail ?? body);
  } catch {
    return body;
  }
}

/** "PanglaoDB (27 Mar 2020) + CellMarker 3.0 (2026-07-22 release, unpublished) + ACT (21 Feb 2024);
 * fetched 19 Oct 2026", from the sources on disk and in use. */
export function markerDbSourceText(sources: MarkerDbSource[]): string {
  const have = inUse(sources);
  if (!have.length) return "";
  const names = have.map((s) => (s.name === "cellmarker2" ? s.label : `${s.label} (${s.release})`)).join(" + ");
  const dates = have.map((s) => s.fetched_at).filter((d): d is string => !!d).sort();
  return `${names}${dates.length ? `; fetched ${formatDate(dates[dates.length - 1])}` : ""}`;
}

/** Sources on disk and in use (CellMarker 2.0 sits unused once 3.0 is in). */
function inUse(sources: MarkerDbSource[]): MarkerDbSource[] {
  return sources.filter((s) => s.downloaded && s.needed !== false);
}

function mb(bytes: number): string {
  return (bytes / 1e6).toFixed(0);
}

function genesText(genes: string[], max = 6): string {
  return genes.length > max ? `${genes.slice(0, max).join(", ")} +${genes.length - max}` : genes.join(", ");
}

/**
 * Does each label's database marker list show up among its group's markers?
 * (marker-db extension of C005.) Matches every label of `column` by name to
 * PanglaoDB, CellMarker (3.0, or 2.0 when 3.0 can't be fetched) and ACT, gives a
 * verdict per label and the 3 database types its markers fit best. The
 * databases are fetched once, on first use; the download's progress is polled.
 */
export function MarkerDbCheck({ datasetId, column }: { datasetId: string; column: string }) {
  const [status, setStatus] = useState<MarkerDbStatus | null>(null);
  const [species, setSpecies] = useState<"" | "human" | "mouse">("");
  const [result, setResult] = useState<MarkerDbResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allRows, setAllRows] = useState(false);
  const request = useRef(0); // answers for an older dataset or column are dropped

  useEffect(() => {
    let live = true;
    setStatus(null);
    setSpecies("");
    getMarkerDbStatus(datasetId)
      .then((s) => live && s?.sources && setStatus(s))
      .catch(() => undefined); // an older backend without the route: the button still works
    return () => {
      live = false;
    };
  }, [datasetId]);

  // While a check runs (the first one downloads), follow the download's progress.
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      getMarkerDbStatus(datasetId).then((s) => s?.sources && setStatus(s), () => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, [running, datasetId]);

  useEffect(() => {
    request.current++;
    setResult(null);
    setError(null);
    setAllRows(false);
  }, [datasetId, column]);

  const run = async (retry = false) => {
    const id = ++request.current;
    setRunning(true);
    setError(null);
    try {
      const r = await runMarkerDbCheck(datasetId, column, {
        ...(species ? { species } : {}),
        ...(retry ? { retry: true } : {}),
      });
      if (id !== request.current) return;
      setResult(r);
    } catch (e) {
      if (id === request.current) setError(errorText(e));
    } finally {
      if (id === request.current) setRunning(false);
      getMarkerDbStatus(datasetId).then((s) => s?.sources && setStatus(s), () => undefined);
    }
  };

  const ready = status?.sources.some((s) => s.downloaded) ?? false;
  const failed = status?.sources.filter((s) => !s.downloaded && s.error) ?? [];
  const missing = status?.sources.filter((s) => !s.downloaded && s.needed !== false && !s.error) ?? [];
  const fetching = status?.sources.find((s) => s.progress);
  const cellmarker =
    result?.cellmarker ??
    status?.cellmarker ??
    inUse(status?.sources ?? []).find((s) => s.name.startsWith("cellmarker"))?.label ??
    null;
  const sourceLine = markerDbSourceText(result?.sources ?? status?.sources ?? []);
  const rows = result ? (allRows ? result.labels : result.labels.slice(0, ROWS)) : [];

  return (
    <div className="mt-3 space-y-2 border-t border-slate-100 pt-3" data-testid="marker-db-check">
      <div className="flex items-center gap-2">
        <Database className="h-4 w-4 text-primary" />
        <span className="text-xs font-semibold text-slate-800">Marker database check</span>
        <span className="text-[11px] text-slate-400">{prettyObsLabel(column)}</span>
      </div>
      <p className="text-[11px] text-slate-500">
        Matches each label to PanglaoDB, CellMarker and ACT cell types and asks whether the database&apos;s markers for
        it are among the group&apos;s own top 25 markers, then lists the 3 PanglaoDB / curated CellMarker types the
        group&apos;s markers fit best.
      </p>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <label htmlFor="mdb-species" className="text-slate-600">
          Species
        </label>
        <select
          id="mdb-species"
          value={species}
          onChange={(e) => setSpecies(e.target.value as "" | "human" | "mouse")}
          className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
        >
          <option value="">
            {status?.dataset_species && status.dataset_species !== "unknown"
              ? `From the genes (${status.dataset_species})`
              : "From the genes"}
          </option>
          <option value="human">Human</option>
          <option value="mouse">Mouse</option>
        </select>
        <button
          type="button"
          onClick={() => void run(false)}
          disabled={running || !column || status?.state === "disabled"}
          className="inline-flex items-center gap-1.5 rounded border border-slate-200 px-2.5 py-1 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
        >
          {running && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {ready ? "Check against marker databases" : "Download databases and check"}
        </button>
      </div>
      {!ready && status?.state !== "disabled" && !running && (
        <p className="text-[11px] text-slate-400">
          First use downloads PanglaoDB (0.3 MB), CellMarker 3.0 (72 MB; CellMarker 2.0, 10 MB, if that fails) and ACT
          (0.8 MB in 498 files) from their own sites into the data volume.
        </p>
      )}
      {running && (fetching || !ready) && (
        <p className="text-[11px] text-slate-500" data-testid="marker-db-progress">
          {fetching?.progress
            ? `Downloading ${fetching.label}: ${mb(fetching.progress.done)} of ${mb(fetching.progress.total)} MB…`
            : "Downloading the databases (first use)…"}
        </p>
      )}
      {status?.state === "disabled" && !ready && (
        <p className="text-[11px] text-slate-500" data-testid="marker-db-disabled">
          The marker databases are not downloaded and downloading is turned off (MARKER_DB_AUTO_DOWNLOAD=false).
        </p>
      )}
      {(error || failed.length > 0) && !running && (
        <div role="alert" className="flex flex-wrap items-center gap-2 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          <span>
            {error ??
              `${failed.map((s) => s.label).join(" and ")} could not be downloaded${
                failed[0]?.error ? ` (${failed[0].error})` : ""
              }.${
                failed.some((s) => s.name === "cellmarker3") && cellmarker === "CellMarker 2.0"
                  ? " Using CellMarker 2.0 instead."
                  : ""
              }`}
          </span>
          {status?.state !== "disabled" && (
            <button
              type="button"
              onClick={() => void run(true)}
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              <RotateCcw className="h-3 w-3" /> Retry download
            </button>
          )}
        </div>
      )}

      {result && (
        <div className="space-y-1.5" data-testid="marker-db-result">
          <p className="text-[11px] text-slate-600">
            {result.verdict_counts.supported} of {result.labels.length} labels supported,{" "}
            {result.verdict_counts.no_support} with no database support, {result.verdict_counts.too_few} with too few
            database markers, {result.verdict_counts.not_in_db} not in the databases ({result.species} rows).
            {result.species_note ? ` ${result.species_note}` : ""}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]" data-testid="marker-db-labels">
              <thead>
                <tr className="text-left text-slate-400">
                  <th className="font-medium">Label</th>
                  <th className="pl-2 font-medium">Database</th>
                  <th className="pl-2 font-medium">Matched type</th>
                  <th className="pl-2 font-medium" title="The label's database markers among the group's top 25">
                    Shared markers
                  </th>
                  <th className="pl-2 font-medium" title="Database types the group's markers fit best">
                    Best database types
                  </th>
                </tr>
              </thead>
              <tbody className="text-slate-600">
                {rows.map((r) => (
                  <LabelRow key={r.label} row={r} />
                ))}
              </tbody>
            </table>
          </div>
          {result.labels.length > ROWS && (
            <button type="button" onClick={() => setAllRows((v) => !v)} className="text-[11px] text-primary hover:underline">
              {allRows ? "Show fewer" : `Show all ${result.labels.length} labels`}
            </button>
          )}
          <p className="text-[11px] text-slate-400">
            {result.note} {result.elapsed_seconds} s.
          </p>
        </div>
      )}

      {sourceLine && (
        <p className="text-[11px] text-slate-400" data-testid="marker-db-source">
          {sourceLine}.{" "}
          {inUse(result?.sources ?? status?.sources ?? []).map((s, i) => (
              <span key={s.name}>
                {i > 0 && " · "}
                <a href={s.home} target="_blank" rel="noreferrer" className="hover:text-slate-600 hover:underline">
                  {s.label}
                </a>{" "}
                (
                <a
                  href={`https://doi.org/${s.doi}`}
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-slate-600 hover:underline"
                >
                  {s.paper}
                </a>
                )
              </span>
            ))}
          .{result && ` Verdicts use ${result.verdict_sources.join(" + ")} (${result.species}).`} Suggestions rank{" "}
          {(
            result?.suggestion_sources ??
            (cellmarker === "CellMarker 2.0" ? ["PanglaoDB", "CellMarker 2.0"] : ["PanglaoDB", "CellMarker 3.0 (curated)"])
          ).join(" + ")}{" "}
          types only.
          {missing.length > 0 && ` ${missing.map((s) => s.label).join(" and ")} not downloaded.`}{" "}
          None of the databases states a licence; check reuse terms with them.
        </p>
      )}
    </div>
  );
}

function LabelRow({ row }: { row: MarkerDbLabel }) {
  return (
    <tr className="align-top">
      <td className="max-w-[9rem] truncate pr-2" title={row.label}>
        {row.label}
      </td>
      <td className="pl-2">
        <span
          className={`inline-block whitespace-nowrap rounded-full border px-1.5 py-px text-[10px] ${VERDICT_STYLE[row.verdict]}`}
          title={VERDICT_TITLE[row.verdict]}
          data-verdict={row.verdict}
        >
          {row.verdict_text}
        </span>
      </td>
      <td className="max-w-[11rem] pl-2" title={row.match.types.join(", ")}>
        {row.db_type ? (
          <>
            <span className="text-slate-700">{row.db_type}</span>
            {row.match.how && row.match.how !== "exact" && (
              <span className={`block text-[10px] ${row.match.via_parent ? "text-amber-700" : "text-slate-400"}`}>
                {row.match.how_text}
              </span>
            )}
          </>
        ) : (
          <span className="text-slate-400">–</span>
        )}
      </td>
      <td className="max-w-[12rem] pl-2" title={row.overlap.join(", ")}>
        {row.db_type ? (
          <>
            {row.overlap.length ? genesText(row.overlap) : <span className="text-slate-400">none</span>}
            <span className="block text-[10px] text-slate-400">
              {row.overlap.length} of {row.n_db_genes} measured
              {row.p !== null && row.overlap.length ? `, p ${formatPValue(row.p)}` : ""}
            </span>
          </>
        ) : (
          <span className="text-slate-400">–</span>
        )}
      </td>
      <td className="pl-2">
        {row.suggestions.length ? (
          <ol className="space-y-0.5">
            {row.suggestions.map((s) => (
              <li key={s.cell_type} title={`${s.overlap.join(", ")} (p ${formatPValue(s.p)}; ${s.sources.join(", ")})`}>
                <span className="text-slate-700">{s.cell_type}</span>{" "}
                <span className="text-[10px] text-slate-400">{genesText(s.overlap, 4)}</span>
              </li>
            ))}
          </ol>
        ) : (
          <span className="text-slate-400">–</span>
        )}
      </td>
    </tr>
  );
}
