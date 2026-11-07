import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { getLabelMatches, type LabelMatch, type LabelMatches } from "@/api/ontology";
import { InfoTip } from "@/components/common/InfoTip";
import { errorMessage } from "@/lib/errorMessage";

const ROWS = 8; // more behind "show all"

const HOW: Record<LabelMatch["how"], string> = {
  exact_name: "name",
  exact_synonym: "synonym",
  normalised: "normalised",
  single_cell: "single-cell name",
  none: "",
};

/** The match column: how the label met the term, and whether the AI annotator gave it. */
function howText(r: LabelMatch): string {
  if (!r.cl_id) return "";
  return r.source === "model" ? `AI · ${HOW[r.how]}` : HOW[r.how];
}

function howTitle(r: LabelMatch): string | undefined {
  if (!r.cl_id) return undefined;
  const who = r.source === "model" ? "The AI annotator gave this id; the label agrees with it" : "From the label";
  return `${who}: ${r.how_text}${r.cl_name ? ` (${r.cl_name})` : ""}`;
}

/**
 * The Cell Ontology id of each label of an annotation column, by strict name matching
 * only (GET /annotation/ontology): a link to the term's page where a label matches,
 * nothing where it does not. For the AI annotator's columns the model's own id is
 * used when the label agrees with it. `refreshKey` refetches after a new run.
 */
export function AnnotationOntologyIds({
  datasetId,
  column,
  refreshKey = 0,
}: {
  datasetId: string;
  column: string;
  refreshKey?: number;
}) {
  const [data, setData] = useState<LabelMatches | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    getLabelMatches(datasetId, column)
      .then((d) => live && setData(d))
      .catch((e: unknown) => live && setError(errorMessage(e)));
    return () => {
      live = false;
    };
  }, [datasetId, column, refreshKey]);

  if (error) {
    return (
      <p className="mt-3 text-[11px] text-slate-400" data-testid="ontology-ids-error">
        Cell Ontology ids unavailable: {error}
      </p>
    );
  }
  if (!data || data.n_labels === 0) return null;
  const rows = showAll ? data.labels : data.labels.slice(0, ROWS);

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50/60 p-3" data-testid="ontology-ids">
      <div className="mb-1 flex items-center gap-1">
        <p className="text-xs font-semibold text-slate-700">Cell Ontology ids</p>
        <InfoTip label="About the Cell Ontology ids" width={300}>
          Each label is matched to a Cell Ontology (CL) term by name only: the same name or exact synonym, or the
          same once case, plurals, &quot;+&quot; (positive), hyphens and a trailing &quot;cells&quot; are set aside.
          Abbreviations (&quot;CD14 Mono&quot;) and labels CL does not name get no id rather than a guess. The AI
          annotator also gives an id; it is used only when the label agrees with that term. To keep terms with the
          dataset (and for the CELLxGENE export), use &quot;Cell Ontology…&quot; under the legend.
        </InfoTip>
      </div>
      <p className="mb-2 text-[11px] text-slate-500">
        {data.n_matched} of {data.n_labels} labels in <code className="font-mono">{column}</code> matched{data.version ? ` · ${data.version.replace(/^cl\/releases\//, "CL ").replace(/\/cl-basic\.owl$/, "")}` : ""}.
      </p>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="font-medium" title="The label in the annotation column">
              Label
            </th>
            <th className="pl-2 font-medium" title="Cell Ontology term id; links to the term's page (EBI OLS)">
              CL id
            </th>
            <th
              className="pl-2 font-medium"
              title="How the label matched the term: its name, an exact synonym, or the same once case, plurals, + and hyphens are set aside. AI = the annotator's own id, agreeing with the label"
            >
              Match
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="text-slate-700">
              <td className="max-w-[14rem] truncate" title={r.label}>
                {r.label}
              </td>
              <td className="whitespace-nowrap pl-2">
                {r.cl_id && r.url && (
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noreferrer"
                    title={r.cl_name ?? r.cl_id}
                    className="inline-flex items-center gap-0.5 font-mono text-primary hover:underline"
                  >
                    {r.cl_id}
                    <ExternalLink className="h-2.5 w-2.5" />
                  </a>
                )}
              </td>
              <td className="whitespace-nowrap pl-2 text-slate-500" title={howTitle(r)}>
                {howText(r)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.labels.length > ROWS && (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="mt-1 text-[11px] text-primary hover:underline"
        >
          {showAll ? "Show fewer" : `Show all ${data.labels.length}`}
        </button>
      )}
    </div>
  );
}
