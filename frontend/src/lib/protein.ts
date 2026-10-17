/**
 * CITE-seq helpers: the gene that encodes each common antibody target (for the
 * protein-vs-RNA view) and a rank correlation.
 */

// Antibody target (as panels name it) -> gene symbols, human then mouse.
// Several targets are isoforms or complexes of one gene (CD45RA/RO: PTPRC).
const TARGET_GENES: Record<string, string[]> = {
  CD3: ["CD3E", "CD3D", "CD3G", "Cd3e"],
  CD4: ["CD4", "Cd4"],
  CD8: ["CD8A", "Cd8a"],
  CD8A: ["CD8A", "Cd8a"],
  CD8B: ["CD8B", "Cd8b1"],
  CD11B: ["ITGAM", "Itgam"],
  CD11C: ["ITGAX", "Itgax"],
  CD14: ["CD14", "Cd14"],
  CD15: ["FUT4", "Fut4"],
  CD16: ["FCGR3A", "FCGR3B", "Fcgr3"],
  CD19: ["CD19", "Cd19"],
  CD20: ["MS4A1", "Ms4a1"],
  CD25: ["IL2RA", "Il2ra"],
  CD27: ["CD27", "Cd27"],
  CD28: ["CD28", "Cd28"],
  CD34: ["CD34", "Cd34"],
  CD38: ["CD38", "Cd38"],
  CD45: ["PTPRC", "Ptprc"],
  CD45RA: ["PTPRC", "Ptprc"],
  CD45RO: ["PTPRC", "Ptprc"],
  CD56: ["NCAM1", "Ncam1"],
  CD57: ["B3GAT1", "B3gat1"],
  CD62L: ["SELL", "Sell"],
  CD69: ["CD69", "Cd69"],
  CD127: ["IL7R", "Il7r"],
  CD161: ["KLRB1", "Klrb1"],
  CD197: ["CCR7", "Ccr7"],
  CCR7: ["CCR7", "Ccr7"],
  CD279: ["PDCD1", "Pdcd1"],
  "PD-1": ["PDCD1", "Pdcd1"],
  PD1: ["PDCD1", "Pdcd1"],
  "PD-L1": ["CD274", "Cd274"],
  TIGIT: ["TIGIT", "Tigit"],
  "HLA-DR": ["HLA-DRA"],
  IGD: ["IGHD"],
  IGM: ["IGHM"],
};

/** Genes that may encode an antibody's target, most likely first. The name's
 *  own spelling is tried last (many targets are named after their gene). */
export function targetGenes(protein: string): string[] {
  const key = protein.replace(/^(anti[-_ ]?|human[-_ ]|mouse[-_ ]|hu[._]|m[._])/i, "").toUpperCase();
  const listed = TARGET_GENES[key] ?? [];
  const own = protein.replace(/^(anti[-_ ]?)/i, "");
  return [...new Set([...listed, own, own.toUpperCase()])];
}

function ranks(v: ArrayLike<number>): Float64Array {
  const n = v.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a]! - v[b]!);
  const r = new Float64Array(n);
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && v[order[j + 1]!] === v[order[i]!]) j++;
    const avg = (i + j) / 2 + 1; // ties share their mean rank
    for (let k = i; k <= j; k++) r[order[k]!] = avg;
    i = j + 1;
  }
  return r;
}

/** Spearman's rank correlation (ties averaged); null without variation. */
export function spearman(x: ArrayLike<number>, y: ArrayLike<number>): number | null {
  const n = Math.min(x.length, y.length);
  if (n < 3) return null;
  const rx = ranks(Array.prototype.slice.call(x, 0, n) as number[]);
  const ry = ranks(Array.prototype.slice.call(y, 0, n) as number[]);
  const mean = (n + 1) / 2;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const a = rx[i]! - mean;
    const b = ry[i]! - mean;
    sxy += a * b;
    sxx += a * a;
    syy += b * b;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
}
