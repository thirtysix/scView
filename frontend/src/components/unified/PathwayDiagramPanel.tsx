import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Loader2, X } from "lucide-react";
import {
  getReactomeDiagram,
  getReactomeValues,
  type PathwayColourSource,
  type PathwayDiagramModel,
  type PathwayValues,
} from "@/api/pathways";
import { listDEResults, type DEResultSummary } from "@/api/deOverlap";
import { Panel } from "@/components/common/Panel";
import { PathwayDiagram } from "@/components/unified/PathwayDiagram";

/** A Reactome row opened from an enrichment table. */
export interface OpenPathway {
  stId: string;
  term: string;
  /** What the enrichment was computed from: the default colouring. */
  source: PathwayColourSource | null;
}

interface Props {
  datasetId: string;
  datasetName?: string | null;
  open: OpenPathway;
  onClose: () => void;
}

const sourceKey = (s: PathwayColourSource | null): string =>
  !s ? "none" : s.kind === "de" ? `de:${s.id}` : `markers:${s.column}/${s.group}`;

/**
 * The Reactome diagram of an enrichment row, in an expandable panel, coloured by a
 * DE result: the one the enrichment came from by default, or any the server still
 * keeps. Sub-pathway boxes open their own diagram (Back returns).
 */
export function PathwayDiagramPanel({ datasetId, datasetName, open, onClose }: Props) {
  const [trail, setTrail] = useState<string[]>([open.stId]);
  const stId = trail[trail.length - 1]!;
  const [model, setModel] = useState<PathwayDiagramModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [deResults, setDeResults] = useState<DEResultSummary[]>([]);
  const [colourKey, setColourKey] = useState<string>(sourceKey(open.source));
  const [values, setValues] = useState<PathwayValues | null>(null);
  const [valuesError, setValuesError] = useState<string | null>(null);

  useEffect(() => {
    setTrail([open.stId]);
    setColourKey(sourceKey(open.source));
  }, [open]);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    setModel(null);
    getReactomeDiagram(stId)
      .then((m) => live && setModel(m))
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [stId]);

  // The DE results the server keeps; without a source of its own the diagram takes the newest.
  useEffect(() => {
    let live = true;
    listDEResults(datasetId)
      .then((r) => {
        if (!live) return;
        setDeResults(r.results);
        if (!open.source && r.results[0]) setColourKey(`de:${r.results[0].result_id}`);
      })
      .catch(() => live && setDeResults([]));
    return () => {
      live = false;
    };
  }, [datasetId, open.source]);

  const options = useMemo(() => {
    const out: { key: string; label: string; source: PathwayColourSource | null }[] = [];
    if (open.source?.kind === "markers") {
      out.push({ key: sourceKey(open.source), label: `${open.source.group} markers (vs rest)`, source: open.source });
    }
    for (const r of deResults) out.push({ key: `de:${r.result_id}`, label: r.label, source: { kind: "de", id: r.result_id } });
    if (open.source?.kind === "de" && !out.some((o) => o.key === sourceKey(open.source))) {
      out.unshift({ key: sourceKey(open.source), label: "This DE result", source: open.source });
    }
    out.push({ key: "none", label: "None (not coloured)", source: null });
    return out;
  }, [deResults, open.source]);

  const colour = options.find((o) => o.key === colourKey)?.source ?? null;

  useEffect(() => {
    let live = true;
    setValues(null);
    setValuesError(null);
    if (!colour || !model || model.too_broad) return;
    getReactomeValues(datasetId, stId, colour)
      .then((v) => live && setValues(v))
      .catch((e: unknown) => live && setValuesError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
    // colourKey stands for colour
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, stId, colourKey, model]);

  const openSub = (id: string) => setTrail((t) => [...t, id]);

  const title = model ? `Reactome: ${model.highlight ? model.highlight.name : model.name}` : `Reactome: ${open.term}`;
  const actions = (
    <>
      {trail.length > 1 && (
        <button type="button" onClick={() => setTrail((t) => t.slice(0, -1))} title="Back to the previous diagram"
          className="inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[11px] text-slate-500 hover:bg-slate-200">
          <ArrowLeft className="h-3 w-3" /> Back
        </button>
      )}
      <button type="button" onClick={onClose} aria-label="Close the diagram" title="Close"
        className="rounded p-1 text-slate-400 hover:bg-slate-200 hover:text-slate-600">
        <X className="h-3.5 w-3.5" />
      </button>
    </>
  );

  return (
    <Panel title={title} actions={actions} className="mt-2" bodyClassName="p-2">
      <div data-testid="pathway-panel" className="flex flex-col gap-1.5 text-xs">
        {loading && (
          <p className="flex items-center gap-1 text-slate-500">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading the diagram (fetched from reactome.org on first use)...
          </p>
        )}
        {error && <p role="alert" className="rounded border border-red-200 bg-red-50 p-2 text-red-700">{error}</p>}
        {model && model.message && (
          <p role="note" className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">{model.message}</p>
        )}
        {model && model.too_broad && (
          <ul className="flex flex-wrap gap-1" aria-label="Sub-pathways">
            {model.suggestions.map((s) => (
              <li key={s.st_id}>
                <button type="button" onClick={() => openSub(s.st_id)}
                  className="rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] text-slate-700 hover:bg-slate-50">
                  {s.name}
                </button>
              </li>
            ))}
          </ul>
        )}
        {valuesError && <p role="alert" className="rounded border border-amber-200 bg-amber-50 p-1.5 text-amber-800">{valuesError}</p>}
        {model && !model.too_broad && (
          <PathwayDiagram
            model={model}
            values={values}
            onOpenPathway={openSub}
            datasetName={datasetName}
            controls={
              <label className="flex items-center gap-1 text-[11px] text-slate-600">
                Colour by
                <select
                  value={colourKey}
                  onChange={(e) => setColourKey(e.target.value)}
                  className="max-w-[260px] rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px]"
                >
                  {options.map((o) => (
                    <option key={o.key} value={o.key}>{o.label}</option>
                  ))}
                </select>
              </label>
            }
          />
        )}
      </div>
    </Panel>
  );
}
