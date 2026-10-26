import { useCallback, useEffect, useMemo, useState } from "react";
import { Archive, Download, GitCompare, Loader2, Pin, PinOff, RotateCcw, Trash2 } from "lucide-react";

import { downloadUrl } from "@/api/export";
import {
  compareWithCurrent,
  deleteResult,
  kindLabel,
  listResults,
  pinResult,
  RESTORABLE_KINDS,
  restoreResult,
  resultsZipUrl,
  resultTableUrl,
  type ResultComparison,
  type StoredResult,
  type StoredResultsList,
} from "@/api/results";
import { formatBytes } from "@/lib/formatting";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function errorText(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  const body = msg.match(/^API error \d+: (.*)$/s)?.[1];
  if (body === undefined) return msg;
  try {
    return String(JSON.parse(body).detail ?? body);
  } catch {
    return body;
  }
}

/** The obs column a stored markers / enrichment result is about. */
export function columnOf(r: StoredResult): string {
  return r.kind === "enrichment" ? (r.key.split("|")[0] ?? r.key) : r.key;
}

/**
 * "Stored results" (results store, phase 3): the dataset's stored results, newest first,
 * with a stale badge for one computed on an earlier version of the data. Each row
 * downloads as .tsv.gz and can be pinned (never dropped) or deleted; earlier markers
 * and enrichment can be compared with the current ones and made current again.
 */
export function ResultsList({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<StoredResultsList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kind, setKind] = useState<string>("all");
  const [busy, setBusy] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; text: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [compare, setCompare] = useState<{ id: string; data: ResultComparison | null } | null>(null);
  const bump = useObsRevisionStore((s) => s.bump);

  const load = useCallback(() => {
    listResults(datasetId)
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(errorText(e)));
  }, [datasetId]);

  useEffect(() => {
    setData(null);
    setKind("all");
    setCompare(null);
    load();
  }, [load]);

  const kinds = useMemo(() => [...new Set((data?.results ?? []).map((r) => r.kind))], [data]);
  const rows = (data?.results ?? []).filter((r) => kind === "all" || r.kind === kind);

  const act = async (r: StoredResult, what: string, fn: () => Promise<unknown>, after?: () => void) => {
    setBusy(`${r.id}:${what}`);
    setRowError(null);
    setNotice(null);
    try {
      await fn();
      after?.();
      load();
    } catch (e) {
      setRowError({ id: r.id, text: errorText(e) });
    } finally {
      setBusy(null);
    }
  };

  const toggleCompare = async (r: StoredResult) => {
    if (compare?.id === r.id) {
      setCompare(null);
      return;
    }
    setCompare({ id: r.id, data: null });
    setRowError(null);
    try {
      const data = await compareWithCurrent(datasetId, r.id);
      setCompare({ id: r.id, data });
    } catch (e) {
      setCompare(null);
      setRowError({ id: r.id, text: errorText(e) });
    }
  };

  if (error) return <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>;
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin text-primary" /> Loading results…
      </div>
    );
  }

  const icon = "rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800 disabled:opacity-40";
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4" aria-label="Stored results">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-700">
            <Archive className="h-4 w-4" /> Stored results
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            Results kept on disk beside the dataset: they survive a restart, and earlier markers and enrichment
            are kept when a re-run replaces them (the last {data.keep_superseded} each). {data.results.length}{" "}
            stored, {formatBytes(data.total_bytes)} of {formatBytes(data.max_bytes)}.
          </p>
        </div>
        <button
          className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          disabled={data.results.length === 0}
          onClick={() => downloadUrl(resultsZipUrl(datasetId))}
          title="Every result as .tsv.gz, with a manifest.json of their parameters"
        >
          <Download className="h-3.5 w-3.5" /> All results (.zip)
        </button>
      </div>

      {notice && (
        <div role="status" className="mt-2 rounded-lg bg-emerald-50 p-2 text-xs text-emerald-800">
          {notice}
        </div>
      )}

      {data.results.length === 0 ? (
        <p className="mt-3 text-xs text-slate-500">
          No results stored yet. DE, ligand-receptor, composition and the other analyses are kept here once run.
        </p>
      ) : (
        <>
          {kinds.length > 1 && (
            <div className="mt-3 flex flex-wrap gap-1" role="group" aria-label="Filter by kind">
              {["all", ...kinds].map((k) => (
                <button
                  key={k}
                  onClick={() => setKind(k)}
                  aria-pressed={kind === k}
                  className={`rounded-full px-2 py-0.5 text-xs ${kind === k ? "bg-primary text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}
                >
                  {k === "all" ? "All" : kindLabel(k)}
                </button>
              ))}
            </div>
          )}
          <ul className="mt-2 divide-y divide-slate-100 text-xs" aria-label="Stored results list">
            {rows.map((r) => {
              const restorable = RESTORABLE_KINDS.has(r.kind) && r.status !== "primary";
              const isBusy = (what: string) => busy === `${r.id}:${what}`;
              return (
                <li key={r.id} className="py-2" data-testid="stored-result">
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                      {kindLabel(r.kind)}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <span className="font-medium text-slate-800">{r.label || r.key}</span>
                        {r.status === "pinned" && (
                          <span className="rounded bg-sky-100 px-1.5 py-0.5 text-sky-800">pinned</span>
                        )}
                        {r.status === "superseded" && (
                          <span
                            className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-600"
                            title="A newer result of the same kind replaced it"
                          >
                            earlier
                          </span>
                        )}
                        {r.stale && (
                          <span
                            className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800"
                            title="Computed on an earlier version of this dataset (cells filtered, re-clustered, relabelled or re-run since). The co-pilot does not treat it as a current fact."
                          >
                            earlier data
                          </span>
                        )}
                      </div>
                      <div className="mt-0.5 text-slate-500">
                        {fmtTime(r.created_at)} · scView {r.version} · {formatBytes(r.bytes)}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5">
                      {RESTORABLE_KINDS.has(r.kind) && (
                        <button
                          className={icon}
                          onClick={() => toggleCompare(r)}
                          aria-label={`Compare ${r.label} with the current result`}
                          aria-expanded={compare?.id === r.id}
                          title="Compare with the current result"
                        >
                          <GitCompare className="h-3.5 w-3.5" />
                        </button>
                      )}
                      {restorable && (
                        <button
                          className={icon}
                          disabled={busy !== null || r.stale === true}
                          onClick={() =>
                            act(r, "restore", () => restoreResult(datasetId, r.id), () => {
                              bump(datasetId, columnOf(r));
                              setNotice(`${r.label} is the current result again; the one it replaced is kept here.`);
                            })
                          }
                          aria-label={`Make ${r.label} current again`}
                          title={
                            r.stale
                              ? "Computed on earlier groups: it does not describe the groups now"
                              : "Make this the current result again (the current one is kept here)"
                          }
                        >
                          {isBusy("restore") ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                        </button>
                      )}
                      <button
                        className={icon}
                        onClick={() => downloadUrl(resultTableUrl(datasetId, r.id))}
                        aria-label={`Download ${r.label} as .tsv.gz`}
                        title="Download as .tsv.gz"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </button>
                      <button
                        className={icon}
                        disabled={busy !== null}
                        onClick={() => act(r, "pin", () => pinResult(datasetId, r.id, r.status !== "pinned"))}
                        aria-label={r.status === "pinned" ? `Unpin ${r.label}` : `Pin ${r.label}`}
                        title={r.status === "pinned" ? "Unpin (retention rules apply again)" : "Pin: keep until deleted"}
                      >
                        {r.status === "pinned" ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
                      </button>
                      {confirmDelete === r.id ? (
                        <button
                          className="rounded bg-red-600 px-1.5 py-0.5 text-white hover:bg-red-700"
                          onClick={() => {
                            setConfirmDelete(null);
                            void act(r, "delete", () => deleteResult(datasetId, r.id));
                          }}
                          onBlur={() => setConfirmDelete(null)}
                          autoFocus
                        >
                          Delete?
                        </button>
                      ) : (
                        <button
                          className={icon}
                          disabled={busy !== null}
                          onClick={() => setConfirmDelete(r.id)}
                          aria-label={`Delete ${r.label}`}
                          title="Delete"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                  {rowError?.id === r.id && (
                    <div role="alert" className="mt-1 rounded bg-red-50 p-1.5 text-red-700">
                      {rowError.text}
                    </div>
                  )}
                  {compare?.id === r.id && <ComparisonView data={compare.data} />}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}

function ComparisonView({ data }: { data: ResultComparison | null }) {
  if (!data) {
    return (
      <div className="mt-1 flex items-center gap-2 text-slate-500">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Comparing…
      </div>
    );
  }
  if (data.kind === "enrichment") {
    return (
      <div className="mt-1 rounded-lg bg-slate-50 p-2 text-slate-600" data-testid="compare-enrichment">
        <p>
          Significant terms (adjusted p &lt; {data.fdr}): {data.n_sig_earlier} earlier, {data.n_sig_current} now,{" "}
          {data.shared} in both{data.jaccard !== null && ` (Jaccard ${data.jaccard})`}.
        </p>
        {data.only_earlier.length > 0 && <p className="mt-0.5">Only earlier: {data.only_earlier.join(", ")}</p>}
        {data.only_current.length > 0 && <p className="mt-0.5">Only now: {data.only_current.join(", ")}</p>}
      </div>
    );
  }
  const method = (p: Record<string, unknown>) => String(p.method ?? "?");
  return (
    <div className="mt-1 overflow-x-auto rounded-lg bg-slate-50 p-2 text-slate-600" data-testid="compare-markers">
      <p>
        Top {data.n} genes per group shared by the earlier ({method(data.earlier)}) and current (
        {method(data.current)}) markers.
      </p>
      <table className="mt-1 w-full text-left">
        <thead>
          <tr className="text-slate-500">
            <th className="pr-3 font-medium">Group</th>
            <th className="pr-3 font-medium" title={`Of the top ${data.n} genes`}>
              Shared
            </th>
            <th className="pr-3 font-medium">Only earlier</th>
            <th className="font-medium">Only now</th>
          </tr>
        </thead>
        <tbody>
          {data.groups.map((g) => (
            <tr key={g.group} className="align-top">
              <td className="pr-3 font-medium text-slate-700">{g.group}</td>
              <td className="pr-3 tabular-nums">{g.shared === null ? "not now" : `${g.shared}/${data.n}`}</td>
              <td className="pr-3">{g.only_earlier.join(", ")}</td>
              <td>{g.only_current.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
