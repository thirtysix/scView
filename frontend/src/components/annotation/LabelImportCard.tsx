import { useEffect, useRef, useState } from "react";
import { FileUp, Loader2 } from "lucide-react";
import { importLabelFile } from "@/api/labels";
import { Help } from "@/components/common/Help";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { columnNameError } from "@/lib/labels";
import {
  BY_ORDER,
  defaultColumns,
  describeReport,
  inCardWords,
  parseHeader,
  readFirstLine,
  suggestImportName,
  type LabelImportReport,
} from "@/lib/labelImport";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

/** Exact counts: a match report is read number by number. */
const count = (v: number) => v.toLocaleString("en-US");

/**
 * "Import labels from a file": a per-cell label table (an author annotation, a
 * collaborator's file) into a new user label column. Check first: the backend's dry run
 * says how many rows match, which name no cell and which cells the table leaves out, and
 * refuses a table whose barcodes do not fit (core/label_import.py). Then import.
 */
export function LabelImportCard({ datasetId }: { datasetId: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const existing = (dataset?.obs_columns ?? []).map((c) => c.name);
  const refresh = useDatasetRefresh();
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[] | null>(null);
  const [barcode, setBarcode] = useState<string>(BY_ORDER);
  const [label, setLabel] = useState("");
  const [name, setName] = useState("");
  const [strip, setStrip] = useState(false);
  const [minMatch, setMinMatch] = useState(90);
  const [busy, setBusy] = useState<"check" | "import" | null>(null);
  const [report, setReport] = useState<LabelImportReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<LabelImportReport | null>(null);

  useEffect(() => {
    setFile(null);
    setColumns(null);
    setReport(null);
    setError(null);
    setSaved(null);
  }, [datasetId]);

  const pickFile = async (f: File | null) => {
    setFile(f);
    setReport(null);
    setError(null);
    setSaved(null);
    setColumns(null);
    if (!f) return;
    const line = await readFirstLine(f);
    const header = line ? parseHeader(line, f.name) : null;
    const cols = header?.columns ?? [];
    setColumns(cols);
    const picks = defaultColumns(cols);
    setBarcode(picks.barcode);
    setLabel(picks.label);
    setName(suggestImportName(picks.label, existing));
  };

  // Any change of what would be sent makes an earlier check stale.
  const changed = () => {
    setReport(null);
    setError(null);
  };

  const nameProblem = columnNameError(name, existing);
  const ready = !!file && !!label && !nameProblem && busy === null;

  const run = async (dryRun: boolean) => {
    if (!file || !ready) return;
    setBusy(dryRun ? "check" : "import");
    setError(null);
    try {
      const r = await importLabelFile(datasetId, file, {
        column: name.trim(),
        barcodeColumn: barcode === BY_ORDER ? null : barcode,
        labelColumn: label,
        stripSuffix: strip,
        minMatch: minMatch / 100,
        dryRun,
      });
      if (dryRun) {
        setReport(r);
      } else {
        setSaved(r);
        setReport(null);
        setFile(null);
        setColumns(null);
        if (fileInput.current) fileInput.current.value = "";
        await refresh(datasetId, r.column);
      }
    } catch (e) {
      setReport(null);
      setError(inCardWords(e instanceof Error ? e.message : "The import failed."));
    } finally {
      setBusy(null);
    }
  };

  const colourBy = (column: string) => {
    useSettingsStore.getState().setColorBy(column);
    useViewStore.getState().setPanel("unified");
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4" data-testid="label-import">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-800">
        <FileUp className="h-4 w-4" />
        <Help id="labelimport.card">Import labels from a file</Help>
      </div>
      <div className="space-y-2 text-xs">
        <label className="flex flex-wrap items-center gap-2">
          <span className="text-slate-600">Label table (CSV or TSV, .gz too)</span>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.tsv,.txt,.tab,.gz"
            aria-label="Label table file"
            onChange={(e) => void pickFile(e.target.files?.[0] ?? null)}
            className="text-xs text-slate-600"
          />
        </label>

        {file && columns && (
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1">
              <Help id="labelimport.barcodes">Cell names</Help>
              <select
                className="rounded border border-slate-300 bg-white px-1 py-0.5 text-slate-700"
                value={barcode}
                aria-label="Cell name column"
                onChange={(e) => {
                  setBarcode(e.target.value);
                  changed();
                }}
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
                <option value={BY_ORDER}>none: one row per cell, in order</option>
              </select>
            </label>
            <label className="flex items-center gap-1">
              Labels
              <select
                className="rounded border border-slate-300 bg-white px-1 py-0.5 text-slate-700"
                value={label}
                aria-label="Label column"
                onChange={(e) => {
                  setLabel(e.target.value);
                  setName(suggestImportName(e.target.value, existing));
                  changed();
                }}
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1">
              New column
              <input
                value={name}
                aria-label="New column name"
                onChange={(e) => {
                  setName(e.target.value);
                  changed();
                }}
                className="w-40 rounded border border-slate-300 px-1.5 py-0.5 text-slate-700"
              />
            </label>
            {barcode !== BY_ORDER && (
              <>
                <label className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={strip}
                    onChange={(e) => {
                      setStrip(e.target.checked);
                      changed();
                    }}
                  />
                  <Help id="labelimport.strip">Ignore a -1 or .1 ending</Help>
                </label>
                <label className="flex items-center gap-1">
                  <Help id="labelimport.min-match">Rows that must match</Help>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={minMatch}
                    aria-label="Share of rows that must match (%)"
                    onChange={(e) => {
                      setMinMatch(Math.max(0, Math.min(100, Number(e.target.value) || 0)));
                      changed();
                    }}
                    className="w-14 rounded border border-slate-300 px-1 py-0.5 text-slate-700"
                  />
                  %
                </label>
              </>
            )}
          </div>
        )}

        {file && columns && nameProblem && (
          <p role="alert" className="rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
            {nameProblem}
          </p>
        )}

        {file && columns && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => void run(true)}
              disabled={!ready}
              className="flex items-center gap-1 rounded border border-slate-300 px-2.5 py-1 font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {busy === "check" && <Loader2 className="h-3 w-3 animate-spin" />}
              Check the match
            </button>
            <button
              onClick={() => void run(false)}
              disabled={!ready || !report}
              title={report ? undefined : "Check the match first"}
              className="flex items-center gap-1 rounded bg-blue-500 px-2.5 py-1 font-medium text-white hover:bg-blue-600 disabled:opacity-50"
            >
              {busy === "import" && <Loader2 className="h-3 w-3 animate-spin" />}
              Import into {name.trim() || "the new column"}
            </button>
          </div>
        )}

        {error && (
          <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-[11px] text-red-700">
            {error}
          </p>
        )}

        {report && <ImportReport report={report} />}

        {saved && (
          <p className="flex flex-wrap items-center gap-2 rounded bg-emerald-50 px-2 py-1.5 text-[11px] text-emerald-800">
            Saved {count(saved.n_matched)} cells into <code>{saved.column}</code> ({saved.n_labels} labels with
            Unassigned).
            <button onClick={() => colourBy(saved.column)} className="font-medium text-blue-600 hover:underline">
              Colour by it
            </button>
          </p>
        )}
      </div>
    </div>
  );
}

const SHOWN_LABELS = 20;

function ImportReport({ report }: { report: LabelImportReport }) {
  const lines = describeReport(report);
  const more = (report.n_imported_labels ?? report.labels.length) - report.labels.length;
  return (
    <div className="space-y-1.5 rounded border border-slate-200 bg-slate-50 px-2 py-1.5" data-testid="label-import-report">
      {lines.map((l) => (
        <p key={l} className="text-[11px] text-slate-700">
          {l}
        </p>
      ))}
      {report.hint && <p className="text-[11px] text-amber-800">{inCardWords(report.hint)}</p>}
      {report.note && <p className="text-[11px] text-slate-500">{report.note}</p>}
      <table className="mt-1 text-[11px]" aria-label="Labels in the table">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="pr-4 font-medium">Label</th>
            <th className="text-right font-medium">Cells</th>
          </tr>
        </thead>
        <tbody>
          {report.labels.slice(0, SHOWN_LABELS).map((r) => (
            <tr key={r.label} className="text-slate-700">
              <td className="pr-4">{r.label}</td>
              <td className="text-right tabular-nums">{count(r.n_cells)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {more > 0 && <p className="text-[11px] text-slate-500">and {more} more labels</p>}
    </div>
  );
}
