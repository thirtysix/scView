export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

export function formatPValue(p: number): string {
  if (p < 0.001) return p.toExponential(2);
  return p.toFixed(4);
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
    const method = clust[1][0].toUpperCase() + clust[1].slice(1);
    return `${method} (res ${clust[2]})`;
  }
  return name;
}
