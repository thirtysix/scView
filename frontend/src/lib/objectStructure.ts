import type { ColumnSummary, DatasetStructure, KeyedArray, MatrixSummary, UnsNode } from "@/api/structure";

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return "-";
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : Number(v.toFixed(digits)).toString();
}

export function fmtBytes(b: number | null): string {
  if (b == null) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = b;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 10 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export const shapeText = (shape: number[] | null | undefined): string =>
  shape ? shape.map((n) => n.toLocaleString()).join(" x ") : "-";

/** "what the sampled values look like": counts, log-normalised, scaled. */
export function valuesKind(m: MatrixSummary): string {
  if (m.looks_like_counts) return "counts (integers)";
  if (m.sample.negative) return "negative values (scaled?)";
  if (m.sample.max != null && m.sample.max > 0) return "non-integer (normalised?)";
  return "all zero in the sample";
}

/** The matrices in order, each with a display name. */
export function matrices(s: DatasetStructure): { name: string; m: MatrixSummary }[] {
  const out: { name: string; m: MatrixSummary }[] = [];
  if (s.X) out.push({ name: "X", m: s.X });
  if (s.raw) out.push({ name: ".raw", m: s.raw });
  for (const [k, m] of Object.entries(s.layers)) out.push({ name: `layers['${k}']`, m });
  for (const [k, m] of Object.entries(s.sidecars)) out.push({ name: `${k} (sidecar file)`, m });
  return out;
}

export function matrixLine(name: string, m: MatrixSummary): string {
  const stored = m.nnz != null ? `, ${m.nnz.toLocaleString()} stored (${fmtNum((m.density ?? 0) * 100, 2)}%)` : "";
  return `${name}: ${shapeText(m.shape)} ${m.format} ${m.dtype}${m.on_disk ? " on disk" : ""}${stored}; ` +
    `sample ${fmtNum(m.sample.min)} to ${fmtNum(m.sample.max)}, ${valuesKind(m)}`;
}

export function columnLine(c: ColumnSummary): string {
  const range = c.kind === "numeric" ? `, ${fmtNum(c.min)} to ${fmtNum(c.max)}` : "";
  const ex = c.kind !== "numeric" && c.examples.length ? `, e.g. ${c.examples.join(", ")}` : "";
  const miss = c.n_missing ? `, ${c.n_missing.toLocaleString()} missing` : "";
  return `${c.name} (${c.dtype}): ${c.n_unique.toLocaleString()} unique${miss}${range}${ex}`;
}

const keyedLine = (a: KeyedArray) => `${a.key}: ${shapeText(a.shape)} ${a.kind}${a.dtype ? ` ${a.dtype}` : ""}`;

export function unsLabel(n: UnsNode): string {
  if (n.type === "dict") return `${n.key}: dict, ${n.size} key${n.size === 1 ? "" : "s"}`;
  if (n.value !== undefined) return `${n.key}: ${n.type} = ${JSON.stringify(n.value)}`;
  if (n.shape) return `${n.key}: ${n.type} ${shapeText(n.shape)}${n.dtype ? ` ${n.dtype}` : ""}`;
  if (n.size != null) return `${n.key}: ${n.type}, length ${n.size.toLocaleString()}`;
  return `${n.key}: ${n.type}`;
}

function unsLines(nodes: UnsNode[], depth: number, out: string[]): void {
  for (const n of nodes) {
    out.push(`${"  ".repeat(depth)}${unsLabel(n)}`);
    if (n.children) unsLines(n.children, depth + 1, out);
    if (n.more) out.push(`${"  ".repeat(depth + 1)}… ${n.more} more`);
  }
}

/** The whole summary as plain text, for "copy as text". */
export function structureText(s: DatasetStructure): string {
  const out = [
    `AnnData object: ${s.n_obs.toLocaleString()} cells x ${s.n_vars.toLocaleString()} genes` +
      ` (${s.mode === "memory" ? "in memory" : s.mode === "backed" ? "read from disk" : "large: matrices on disk"}` +
      `${s.file_bytes != null ? `, file ${fmtBytes(s.file_bytes)}` : ""})`,
  ];
  if (s.flags.length) {
    out.push("", "Issues:");
    for (const f of s.flags) out.push(`- ${f.text}`);
  }
  out.push("", "Matrices:");
  for (const { name, m } of matrices(s)) out.push(`  ${matrixLine(name, m)}`);
  for (const [label, t] of [["obs", s.obs], ["var", s.var]] as const) {
    out.push("", `${label}: ${t.n_columns} columns${t.truncated ? ` (first ${t.columns.length} listed)` : ""}`);
    for (const c of t.columns) out.push(`  ${columnLine(c)}`);
  }
  for (const [label, items] of [["obsm", s.obsm], ["obsp", s.obsp], ["varm", s.varm], ["varp", s.varp]] as const) {
    if (items.length) {
      out.push("", `${label}:`);
      for (const a of items) out.push(`  ${keyedLine(a)}`);
    }
  }
  if (s.uns.length) {
    out.push("", "uns:");
    unsLines(s.uns, 1, out);
  }
  return out.join("\n");
}
