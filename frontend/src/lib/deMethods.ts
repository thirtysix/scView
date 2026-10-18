import type { DEGene, DEMethodsResponse, DEResponse } from "@/api/de";
import type { CsvCell } from "@/lib/csv";

/** The cell-level DE tests (backend core/de_methods.py). */
export type CellMethod = "wilcoxon" | "t-test" | "logreg";

export interface CellMethodInfo {
  value: CellMethod;
  label: string;
  /** What its per-gene statistic is. */
  stat: string;
  hasPValues: boolean;
  /** One-line description for the picker's tooltip. */
  help: string;
}

export const CELL_METHODS: CellMethodInfo[] = [
  {
    value: "wilcoxon",
    label: "Wilcoxon rank-sum",
    stat: "z-score",
    hasPValues: true,
    help: "Wilcoxon rank-sum test per gene (scanpy's wilcoxon): no assumption about the shape of the expression distribution.",
  },
  {
    value: "t-test",
    label: "Welch t-test",
    stat: "t statistic",
    hasPValues: true,
    help:
      "Welch's t-test per gene (scanpy's t-test: unequal variances, each group's own size), " +
      "not scanpy's t-test_overestim_var, which inflates the group's variance.",
  },
  {
    value: "logreg",
    label: "Logistic regression",
    stat: "coefficient",
    hasPValues: false,
    help:
      "One logistic regression of group membership on every tested gene at once (scanpy's logreg): " +
      "a coefficient per gene and NO p-values. Genes are ranked by coefficient; it favours genes " +
      "that tell the groups apart jointly, not each gene's own shift.",
  },
];

/** Shown beside the picker: what is asked for but cannot run here. */
export const MAST_NOTE = "MAST is not available: it is an R package, and scView's backend runs Python.";

export function cellMethodInfo(method: string | undefined | null): CellMethodInfo {
  return CELL_METHODS.find((m) => m.value === method) ?? CELL_METHODS[0]!;
}

/** "cell-level Welch t-test", "pseudobulk DESeq2" etc., for captions and the co-pilot. */
export function methodLabel(method: string | undefined | null): string {
  if (method === "pseudobulk") return "pseudobulk DESeq2";
  return cellMethodInfo(method).label;
}

/** The result has per-gene p-values (logistic regression has none). */
export function hasPValues(r: Pick<DEResponse, "method">): boolean {
  return r.method === "pseudobulk" || cellMethodInfo(r.method).hasPValues;
}

/**
 * The genes the table lists. With p-values: significant ones, most significant
 * first. Without (logistic regression): every gene past the |log2FC| cut-off
 * whose coefficient points the same way as its fold change, largest |coefficient|
 * first: there is no significance to filter on.
 */
export function listedGenes(
  r: DEResponse,
  pThreshold: number,
  fcThreshold: number,
  isSignificant: (g: DEGene, p: number, fc: number) => boolean,
): DEGene[] {
  if (hasPValues(r)) {
    return r.genes
      .filter((g) => isSignificant(g, pThreshold, fcThreshold))
      .sort(
        (a, b) =>
          (a.pval_adj ?? 1) - (b.pval_adj ?? 1) || Math.abs(b.logfoldchange) - Math.abs(a.logfoldchange),
      );
  }
  return r.genes
    .filter(
      (g) =>
        g.stat != null &&
        Math.abs(g.logfoldchange) >= fcThreshold &&
        Math.sign(g.stat) === Math.sign(g.logfoldchange),
    )
    .sort((a, b) => Math.abs(b.stat ?? 0) - Math.abs(a.stat ?? 0));
}

const fmt2 = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));

/** The pairwise agreement rows of a methods comparison, labelled. */
export function agreementRows(res: DEMethodsResponse) {
  const label = (m: string) => cellMethodInfo(m).label;
  return res.pairs.map((p) => ({
    key: `${p.a}|${p.b}`,
    pair: `${label(p.a)} vs ${label(p.b)}`,
    up: `${fmt2(p.jaccard_up)} (${p.shared_up})`,
    down: `${fmt2(p.jaccard_down)} (${p.shared_down})`,
    spearman: fmt2(p.spearman),
    jaccardUp: p.jaccard_up,
    jaccardDown: p.jaccard_down,
  }));
}

/** CSV of a methods comparison: the pairs, then the genes only one test ranks in its top N. */
export function agreementCsv(res: DEMethodsResponse): { headers: string[]; rows: CsvCell[][] } {
  const methods = res.methods.map((m) => m.method);
  const rows: CsvCell[][] = res.pairs.map((p) => [
    "pair", `${p.a} vs ${p.b}`, "", p.jaccard_up, p.jaccard_down, p.spearman,
    ...methods.map(() => null),
  ]);
  for (const u of res.unique) {
    rows.push(["only_in_top", u.method, `${u.gene} (${u.side})`, null, null, null,
      ...methods.map((m) => u.ranks[m])]);
  }
  return {
    headers: ["row", "method_or_pair", "gene", `jaccard_up_top${res.top_n}`,
      `jaccard_down_top${res.top_n}`, "spearman", ...methods.map((m) => `rank_${m}`)],
    rows,
  };
}
