/**
 * The ingest step's recorded params as History chips read them (backend
 * core/ingestion/session.py `_record_ingest_provenance`, feature_split.py,
 * gene_symbols.py, public_import.py): words, not JSON. The tissue positions and the
 * converted R objects are worded in provenanceParams.ts.
 */

/** Unit formats (backend `UnitFormat`) and "h5ad" (a plain upload's working copy). */
const FORMATS: Record<string, string> = {
  tenx_mex: "10x MEX",
  tenx_h5: "10x HDF5 (.h5)",
  anndata: "AnnData (.h5ad)",
  h5ad: "AnnData (.h5ad)",
  loom: "Loom",
  zarr: "Zarr",
  dense_table: "expression table (CSV/TSV)",
  seurat: "R object (.rds, converted)",
  h5seurat: "Seurat (.h5Seurat)",
  h5mu: "MuData (.h5mu)",
  xenium: "Xenium folder",
};

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Obj) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const count = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

function formatsText(v: unknown): string | null {
  if (!Array.isArray(v)) return null;
  const names = v.map((f) => FORMATS[String(f)] ?? String(f));
  return `${names.length === 1 ? "format" : "formats"}: ${names.join(", ")}`;
}

/** "CITE-seq: 17 antibodies (3 isotype controls), CLR per cell (Seurat margin = 2)". */
function proteinText(v: Obj): string {
  const n = num(v.n_proteins);
  const controls = Array.isArray(v.controls) ? v.controls.length : 0;
  return (
    `CITE-seq: ${n === null ? "antibodies" : count(n, "antibody", "antibodies")}` +
    (controls ? ` (${count(controls, "isotype control")})` : "") +
    (v.normalisation ? `, ${String(v.normalisation)}` : "")
  );
}

/** "ATAC: 105,421 peaks, GRCh38 (assumed from the peak names)". */
function atacText(v: Obj): string {
  const n = num(v.n_peaks);
  const genome = v.genome ? `, ${String(v.genome)}${v.genome_assumed ? " (assumed from the peak names)" : ""}` : "";
  return `ATAC: ${n === null ? "peaks" : count(n, "peak")}${genome}`;
}

/** "gene symbols from var['feature_name']: 21,241 of 24,505 names changed, ids kept in var['gene_ids']". */
function symbolsText(v: Obj): string {
  const s = obj(v.var) ?? obj(v.raw_var) ?? v;
  const renamed = num(s.renamed);
  const total = num(s.n_genes);
  return (
    `gene symbols from ${String(s.from ?? "the gene table")}` +
    (renamed !== null && total !== null ? `: ${renamed.toLocaleString()} of ${count(total, "name")} changed` : "") +
    (s.ids_in ? `, ids kept in ${String(s.ids_in)}` : "")
  );
}

/** "features split: 33,538 genes, 17 antibodies, 4 multiplexing tags". */
function featureSplitText(v: Obj): string {
  const parts: [unknown, string, string][] = [
    [v.n_genes, "gene", "genes"],
    [v.n_antibodies, "antibody", "antibodies"],
    [v.n_multiplexing_tags, "multiplexing tag", "multiplexing tags"],
    [v.n_guides, "guide", "guides"],
    [v.n_peaks, "peak", "peaks"],
  ];
  const shown = parts.flatMap(([n, one, many]) => (num(n) ? [count(num(n)!, one, many)] : []));
  return `features split: ${shown.join(", ") || "none"}${v.note ? ` (${String(v.note)})` : ""}`;
}

/** "CELLxGENE Discover: dataset 5e3b..., DOI 10.1016/..., CC BY 4.0". */
function publicImportText(v: Obj): string {
  return (
    `${String(v.source ?? "public import")}` +
    (v.dataset_id ? `: dataset ${String(v.dataset_id)}` : "") +
    (v.doi ? `, DOI ${String(v.doi)}` : "") +
    (v.license ? `, ${String(v.license)}` : "")
  );
}

/**
 * An ingest param as its chip reads, "" for one not worth a chip (the cells-in-rows
 * default of a table), or null for a key this does not word (the caller's general rule).
 */
export function ingestParamText(key: string, value: unknown): string | null {
  if (key === "formats") return formatsText(value);
  if (key === "genes_in_rows") return value === true ? "table read with genes in rows" : "";
  const v = obj(value);
  if (!v) return null;
  if (key === "protein") return proteinText(v);
  if (key === "atac") return atacText(v);
  if (key === "gene_symbols") return symbolsText(v);
  if (key === "feature_split") return featureSplitText(v);
  if (key === "public_import") return publicImportText(v);
  return null;
}
