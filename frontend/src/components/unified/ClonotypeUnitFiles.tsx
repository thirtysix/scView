import { Dna } from "lucide-react";
import type { IngestUnit } from "@/api/ingest";

/** Import wizard: the TCR/BCR contig tables bundled with a sample (joined to its cells on create). */
export function ClonotypeUnitFiles({ unit }: { unit: IngestUnit }) {
  const files = unit.files.filter((f) => f.role === "vdj_contigs");
  if (files.length === 0) return null;
  return (
    <ul className="mt-1.5 space-y-0.5 pl-6 text-xs" data-testid="unit-vdj-files">
      {files.map((f) => (
        <li key={f.path} className="flex items-center gap-1.5 text-slate-600" title="Joined to this sample's cells by barcode when the dataset is created">
          <Dna className="h-3 w-3 text-violet-500" />
          {f.name} <span className="text-slate-400">(TCR/BCR contigs)</span>
        </li>
      ))}
    </ul>
  );
}
