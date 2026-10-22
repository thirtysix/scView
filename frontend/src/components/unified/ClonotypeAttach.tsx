import { useState } from "react";
import { Loader2, Upload } from "lucide-react";
import { attachVdj, type VdjFile, type VdjSummary } from "@/api/vdj";
import { errorText } from "@/lib/programs";
import { choiceLabel, fileSentence, needsSample, valueLabel } from "@/lib/vdj";

/**
 * Attach a contig table (Cell Ranger filtered/all_contig_annotations.csv or an
 * AIRR rearrangement TSV) to the open dataset. When its cells come from several
 * libraries the sample must be named, so barcodes are matched within its cells.
 */
export function ClonotypeAttach({
  datasetId,
  summary,
  onAttached,
}: {
  datasetId: string;
  summary: VdjSummary;
  onAttached: (s: VdjSummary) => Promise<void>;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [column, setColumn] = useState("");
  const [sample, setSample] = useState("");
  const [replace, setReplace] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<VdjFile | null>(null);

  const choices = summary.sample_choices;
  const values = choices.find((c) => c.name === column)?.values ?? [];
  const several = needsSample(summary);

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const res = await attachVdj(datasetId, file, {
        sampleColumn: column || undefined,
        sample: column ? sample : undefined,
        replace,
      });
      setDone(res.attached);
      setFile(null);
      await onAttached(res.summary);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-1.5 text-[11px] text-slate-600" data-testid="vdj-attach">
      <label className="flex items-center gap-1.5">
        <span className="shrink-0 font-medium">Contig table</span>
        <input
          aria-label="Contig table file"
          type="file"
          accept=".csv,.tsv,.txt,.gz"
          disabled={busy}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="min-w-0 text-[11px]"
        />
      </label>
      {choices.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span title="The sample these contigs came from: barcodes are matched within its cells only">Sample</span>
          <select
            aria-label="Sample column"
            value={column}
            disabled={busy}
            onChange={(e) => {
              setColumn(e.target.value);
              setSample("");
            }}
            className="rounded border border-slate-300 px-1 py-0.5 text-xs"
          >
            <option value="">{several ? "pick a column…" : "all cells (one sample)"}</option>
            {choices.map((c) => (
              <option key={c.name} value={c.name}>
                {choiceLabel(c)}
              </option>
            ))}
          </select>
          {column && (
            <select
              aria-label="Sample"
              value={sample}
              disabled={busy}
              onChange={(e) => setSample(e.target.value)}
              className="max-w-[45%] rounded border border-slate-300 px-1 py-0.5 text-xs"
            >
              <option value="">pick…</option>
              {values.map((v) => (
                <option key={v} value={v}>
                  {valueLabel(column, v)}
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      {several && !column && (
        <p className="text-amber-700">
          The cells come from several libraries: pick the sample these contigs belong to, so a barcode is only matched
          within that sample&apos;s cells.
        </p>
      )}
      <div className="flex items-center gap-2">
        <label className="flex items-center gap-1" title="Drop this sample's earlier contigs before reading the file">
          <input type="checkbox" checked={replace} disabled={busy} onChange={(e) => setReplace(e.target.checked)} />
          replace earlier contigs
        </label>
        <button
          onClick={submit}
          disabled={!file || busy || (!!column && !sample)}
          className="ml-auto flex items-center gap-1 rounded bg-slate-700 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Upload className="h-3 w-3" />}
          Attach
        </button>
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-800">
          {error}
        </p>
      )}
      {done && (
        <p role="status" className="text-emerald-700">
          {fileSentence(done)}
        </p>
      )}
    </div>
  );
}
