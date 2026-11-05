import type { DEResponse, PseudobulkMethod } from "@/api/de";

/** The pseudobulk models offered (backend core/pseudobulk.py), with when to use each. */
export const PB_METHODS: { value: PseudobulkMethod; label: string; when: string }[] = [
  {
    value: "voom",
    label: "limma-voom",
    when: "the default: fast, and held its false discovery rate on real donor splits; one mean-variance trend.",
  },
  {
    value: "voombygroup",
    label: "voomByGroup",
    when: "when one group's donors vary much more than the other's (see the variance check).",
  },
  {
    value: "edger_ql",
    label: "edgeR QL",
    when: "negative binomial quasi-likelihood F-test (edgeR 4): held its FDR like voom; slower.",
  },
  {
    value: "deseq2",
    label: "DESeq2",
    when: "negative binomial model; its FDR ran high (about 1 in 3 false at 2 donors per side) below 6 donors per side.",
  },
];

export const RUV_CHOICES = [0, 1, 2, 3] as const;

/** "DESeq2", "limma-voom", "voomByGroup", "edgeR QL" (older servers: DESeq2). */
export function pbMethodLabel(m: PseudobulkMethod | null | undefined): string {
  return PB_METHODS.find((x) => x.value === m)?.label ?? "DESeq2";
}

/** "Pseudobulk limma-voom (moderated t, TMM, design ~W1 + donor + condition, RUV k = 1)". */
export function describePbModel(r: Pick<DEResponse, "pb_method" | "design" | "ruv_k">): string {
  const test = r.pb_method === "voom" || r.pb_method === "voombygroup"
    ? "moderated t on TMM log-CPM"
    : r.pb_method === "edger_ql"
      ? "quasi-likelihood F-test, TMM"
      : "Wald test";
  const ruv = r.ruv_k ? `, ${r.ruv_k} RUV factor${r.ruv_k === 1 ? "" : "s"}` : "";
  return `Pseudobulk ${pbMethodLabel(r.pb_method)} (${test}${ruv}, design ${r.design ?? "?"})`;
}

/** A correlation for a table cell: "0.62", or "n/a" when it could not be computed. */
export function formatR(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "n/a" : v.toFixed(2);
}
