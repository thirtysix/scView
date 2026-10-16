export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

export function formatPValue(p: number): string {
  if (p < 0.001) return p.toExponential(2);
  return p.toFixed(4);
}

/** A fold change to `digits` places; `missing` when there is none (scanpy's NaN arrives as null). */
export function formatFoldChange(v: number | null | undefined, digits: number, missing = "–"): string {
  return v == null || !Number.isFinite(v) ? missing : v.toFixed(digits);
}

/** Short date ("16 Oct 2026" in en-GB) for an ISO timestamp; "" when missing or unparseable. */
export function formatDate(iso: string | null | undefined, locale?: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" });
}

/** File size in decimal units, as file managers show it: 950 B, 12.3 MB, 1.20 GB. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1000) return `${bytes} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let v = bytes;
  let i = -1;
  do {
    v /= 1000;
    i++;
  } while (v >= 1000 && i < units.length - 1);
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`;
}

/**
 * Human-friendly label for an obs column name, keeping the raw name as the value.
 *  - scview_leiden_r0.5        -> "Leiden (res 0.5)"
 *  - scview_louvain_r1.0       -> "Louvain (res 1.0)"
 *  - <grouping>_celltypeAnno   -> "<pretty grouping> · cell types"
 * Anything else is returned unchanged.
 */
export function prettyObsLabel(name: string | null | undefined): string {
  if (!name) return "";
  const anno = name.match(/^(.*)_celltypeAnno$/);
  if (anno) return `${prettyObsLabel(anno[1])} · cell types`;

  const clust = name.match(/^scview_(leiden|louvain)_r([\d.]+)$/);
  if (clust) {
    const raw = clust[1] ?? "";
    const method = raw.charAt(0).toUpperCase() + raw.slice(1);
    return `${method} (res ${clust[2] ?? ""})`;
  }
  return name;
}

/**
 * True if `name` is scanpy's bare 'leiden'/'louvain' column made redundant by a
 * named scview_<method>_r<res> copy present in `allNames` — so it can be hidden.
 */
export function isRedundantClusterCol(name: string, allNames: string[]): boolean {
  if (name !== "leiden" && name !== "louvain") return false;
  return allNames.some((n) => n.startsWith(`scview_${name}_r`));
}

/**
 * Display label for an obsm embedding key:
 *  - X_umap               -> "UMAP"
 *  - X_umap_uncorrected   -> "UMAP (uncorrected)"
 *  - X_pca_harmony        -> "PCA (HARMONY)"
 */
export function prettyEmbeddingLabel(name: string): string {
  const [base = "", ...rest] = name.replace(/^X_/, "").split("_");
  const suffix = rest.filter((r) => r.toLowerCase() !== "3d").join(" ");
  if (!suffix) return base.toUpperCase();
  return `${base.toUpperCase()} (${suffix === "uncorrected" ? suffix : suffix.toUpperCase()})`;
}
