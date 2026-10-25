import { useEffect, useState, type ReactNode } from "react";
import { AlertTriangle, Copy, Info, Loader2 } from "lucide-react";
import { InfoTip } from "@/components/common/InfoTip";
import { Caption } from "@/components/common/Caption";
import { getStructure, type ColumnSummary, type DatasetStructure, type KeyedArray, type TableSummary, type UnsNode } from "@/api/structure";
import {
  fmtBytes,
  fmtNum,
  matrices,
  shapeText,
  structureText,
  unsLabel,
  valuesKind,
} from "@/lib/objectStructure";

const MODE_LABEL: Record<DatasetStructure["mode"], string> = {
  memory: "in memory",
  backed: "read from disk",
  large: "large: matrices on disk",
};

const th = "px-1.5 py-1 text-left font-medium text-slate-500";
const td = "px-1.5 py-0.5 align-top";
const summaryClass = "cursor-pointer select-none py-1 text-xs font-semibold text-slate-600";

function Section({ title, children, open = false }: { title: string; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="border-t border-slate-100">
      <summary className={summaryClass}>{title}</summary>
      <div className="pb-2">{children}</div>
    </details>
  );
}

function ColumnsTable({ table, label }: { table: TableSummary; label: string }) {
  if (!table.n_columns) return <p className="text-slate-500">No {label} columns.</p>;
  return (
    <>
      <div className="max-h-72 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-white">
            <tr>
              <th className={th}>Column</th>
              <th className={th}>Type</th>
              <th className={`${th} text-right`}>Unique</th>
              <th className={`${th} text-right`}>Missing</th>
              <th className={th}>Examples / range</th>
            </tr>
          </thead>
          <tbody>
            {table.columns.map((c: ColumnSummary) => (
              <tr key={c.name} className={`border-t border-slate-100 ${c.all_missing ? "text-amber-700" : ""}`}>
                <td className={`${td} font-medium`}>{c.name}</td>
                <td className={td} title={c.dtype}>{c.kind}</td>
                <td className={`${td} text-right tabular-nums`}>{c.n_unique.toLocaleString()}</td>
                <td className={`${td} text-right tabular-nums`}>{c.n_missing ? c.n_missing.toLocaleString() : ""}</td>
                <td className={`${td} max-w-[14rem] truncate`} title={c.examples.join(", ")}>
                  {c.kind === "numeric" ? `${fmtNum(c.min)} to ${fmtNum(c.max)}` : c.examples.join(", ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.truncated && (
        <p className="text-slate-500">
          First {table.columns.length} of {table.n_columns} columns listed.
        </p>
      )}
    </>
  );
}

function KeyedList({ items }: { items: KeyedArray[] }) {
  return (
    <ul className="space-y-0.5">
      {items.map((a) => (
        <li key={a.key}>
          <span className="font-medium text-slate-700">{a.key}</span>: {shapeText(a.shape)} {a.kind}
          {a.dtype ? ` ${a.dtype}` : ""}
        </li>
      ))}
    </ul>
  );
}

function UnsTree({ nodes }: { nodes: UnsNode[] }) {
  return (
    <ul className="space-y-0.5 pl-3">
      {nodes.map((n) =>
        n.children ? (
          <li key={n.key}>
            <details>
              <summary className="cursor-pointer">{unsLabel(n)}</summary>
              <UnsTree nodes={n.children} />
              {n.more ? <p className="pl-3 text-slate-500">… {n.more} more</p> : null}
            </details>
          </li>
        ) : (
          <li key={n.key} className="break-all">
            {unsLabel(n)}
          </li>
        ),
      )}
    </ul>
  );
}

/**
 * "Object" card on the Data Assessment tab: what the AnnData object holds (matrices,
 * obs/var columns, obsm/obsp/varm, uns), read cheaply (matrices sampled, never read
 * whole), with obvious problems flagged and a plain-text copy.
 */
export function ObjectInspector({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [data, setData] = useState<DatasetStructure | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getStructure(datasetId)
      .then((d) => {
        // An older server answers with something else: show nothing rather than break.
        if (!cancelled) setData(d && typeof d.n_obs === "number" && Array.isArray(d.flags) ? d : null);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, refreshKey]);

  const copy = async () => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(structureText(data));
      setNotice("Copied");
    } catch {
      setNotice("Copy failed: the browser blocked the clipboard");
    }
    setTimeout(() => setNotice(null), 3000);
  };

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-600" data-testid="object-inspector">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">Object</h3>
          <InfoTip label="About the object summary">
            What the AnnData file holds: the expression matrices (X, .raw, layers), the cell (obs) and gene
            (var) tables, embeddings and graphs (obsm, obsp, varm) and stored results (uns). Matrix values come
            from a sample, so this is quick on any size of dataset.
          </InfoTip>
          {data && (
            <span className="text-slate-500">
              {data.n_obs.toLocaleString()} x {data.n_vars.toLocaleString()}, {MODE_LABEL[data.mode]}
            </span>
          )}
        </div>
        <span className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" aria-label="Loading" />}
          {notice && <span role="status" className="text-slate-500">{notice}</span>}
          <button
            type="button"
            onClick={copy}
            disabled={!data}
            className="inline-flex items-center gap-1 rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
          >
            <Copy className="h-3 w-3" /> Copy as text
          </button>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="rounded border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
          >
            {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-amber-800">
          {error}
        </p>
      )}
      {data && data.flags.length > 0 && (
        <ul className="space-y-1" data-testid="object-flags">
          {data.flags.map((f) => (
            <li
              key={f.text}
              className={`flex gap-1.5 rounded-md px-2 py-1 ${f.level === "warn" ? "bg-amber-50 text-amber-800" : "bg-slate-50 text-slate-600"}`}
            >
              {f.level === "warn" ? (
                <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" aria-label="Warning" />
              ) : (
                <Info className="mt-0.5 h-3 w-3 flex-shrink-0" aria-label="Note" />
              )}
              <span>{f.text}</span>
            </li>
          ))}
        </ul>
      )}
      {open && data && (
        <div>
          <Section title="Matrices" open>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className={th}>Matrix</th>
                    <th className={th}>Shape</th>
                    <th className={th}>Format</th>
                    <th className={`${th} text-right`} title="Stored (non-zero) values and the share of the matrix they fill">
                      Stored
                    </th>
                    <th className={th} title="Smallest and largest value in the sample">Range</th>
                    <th className={th}>Values</th>
                  </tr>
                </thead>
                <tbody>
                  {matrices(data).map(({ name, m }) => (
                    <tr key={name} className="border-t border-slate-100">
                      <td className={`${td} font-medium`}>{name}</td>
                      <td className={`${td} whitespace-nowrap`}>{shapeText(m.shape)}</td>
                      <td className={td}>
                        {m.format} {m.dtype}
                        {m.on_disk ? " (disk)" : ""}
                      </td>
                      <td className={`${td} text-right tabular-nums`}>
                        {m.nnz != null ? `${m.nnz.toLocaleString()} (${fmtNum((m.density ?? 0) * 100, 2)}%)` : "dense"}
                      </td>
                      <td className={`${td} whitespace-nowrap tabular-nums`}>
                        {fmtNum(m.sample.min)} to {fmtNum(m.sample.max)}
                      </td>
                      <td className={td}>{valuesKind(m)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Caption n={[{ value: data.n_obs, unit: "cells" }, { value: data.n_vars, unit: "genes" }]}>
              Range and integer check from up to {data.sampling.values.toLocaleString()} stored values per
              matrix (seed {data.sampling.seed}); the matrices are never read whole.
              {data.file_bytes != null ? ` File ${fmtBytes(data.file_bytes)}.` : ""}
            </Caption>
          </Section>
          <Section title={`Cells (obs): ${data.obs.n_columns} columns`}>
            <ColumnsTable table={data.obs} label="obs" />
            <p className="mt-1 text-slate-500">Cell names, e.g. {data.obs_names.examples.join(", ")}</p>
          </Section>
          <Section title={`Genes (var): ${data.var.n_columns} columns`}>
            <ColumnsTable table={data.var} label="var" />
            <p className="mt-1 text-slate-500">Gene names, e.g. {data.var_names.examples.join(", ")}</p>
          </Section>
          <Section title={`Embeddings and graphs: ${data.obsm.length + data.obsp.length + data.varm.length + data.varp.length}`}>
            {(["obsm", "obsp", "varm", "varp"] as const).map((k) =>
              data[k].length ? (
                <div key={k} className="mb-1">
                  <p className="font-medium text-slate-700">{k}</p>
                  <KeyedList items={data[k]} />
                </div>
              ) : null,
            )}
          </Section>
          <Section title={`Stored results (uns): ${data.uns.length} keys`}>
            <UnsTree nodes={data.uns} />
          </Section>
        </div>
      )}
    </div>
  );
}
