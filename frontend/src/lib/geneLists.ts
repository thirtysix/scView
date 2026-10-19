/**
 * Named gene lists (the Gene lists tab, stores/geneListsStore.ts): cleaning a
 * list, reading one from a file, writing lists out, and checking a list
 * against the genes a dataset measures.
 *
 * Lists keep gene symbols as entered (a mouse list stays Cd3e); repeats are
 * dropped ignoring case, keeping the first spelling and the order.
 */
import { buildCsv } from "@/lib/csv";
import { parseGeneList } from "@/lib/geneList";

/** Most genes one list holds (the backend's limit for a gene set). */
export const MAX_LIST_GENES = 5000;
/** Most lists kept. */
export const MAX_LISTS = 200;
export const MAX_LIST_NAME = 200;
const MAX_GENE_CHARS = 100;

export interface GeneListDraft {
  name: string;
  genes: string[];
  /** Where the list came from ("Markers: CD14 Mono", "file: hallmark.gmt") */
  source?: string;
}

/** Trimmed, each gene once ignoring case (first spelling kept), at most `max`. */
export function dedupeGenes(genes: Iterable<string>, max = MAX_LIST_GENES): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of genes) {
    const gene = String(raw).trim();
    if (!gene || gene.length > MAX_GENE_CHARS || /\s/.test(gene)) continue;
    const key = gene.toUpperCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(gene);
    if (out.length >= max) break;
  }
  return out;
}

/** A list name on one line, without control characters, at most MAX_LIST_NAME long. */
export function cleanListName(name: string): string {
  // eslint-disable-next-line no-control-regex
  return name.replace(/[\u0000-\u001f\u007f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, MAX_LIST_NAME);
}

/** `name`, or `name (2)`, `name (3)`... whichever no list in `taken` has (ignoring case). */
export function uniqueListName(name: string, taken: Iterable<string>, suffix = (n: number) => ` (${n})`): string {
  const used = new Set([...taken].map((t) => t.toLowerCase()));
  const base = cleanListName(name) || "Gene list";
  if (!used.has(base.toLowerCase())) return base;
  for (let n = 2; ; n++) {
    const next = `${base}${suffix(n)}`;
    if (!used.has(next.toLowerCase())) return next;
  }
}

/** The same genes in the same order, ignoring case (how a list matches another). */
export function sameGenes(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((g, i) => g.toUpperCase() === b[i]!.toUpperCase());
}

// ---------------------------------------------------------------------------
// Reading a file
// ---------------------------------------------------------------------------

/** Enrichr's GMT downloads append a weight to each gene ("TP53,1.0"). */
const WEIGHT = /,\s*[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?$/;
const SET_HEADERS = new Set(["gene_set", "geneset", "gene set", "set", "set_name", "term", "pathway", "name", "gs_name", "list"]);
const GENE_HEADERS = new Set(["gene", "genes", "gene_symbol", "symbol", "gene_name", "genesymbol", "names", "gene_id"]);

const stem = (filename: string) => filename.replace(/^.*[\\/]/, "").replace(/\.[^.]+$/, "") || "Gene list";

/** One row's cells; a cell in double quotes may hold the delimiter ("" is a quote). */
function splitRow(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"' && cell.trim() === "") {
      quoted = true;
      cell = "";
    } else if (ch === delimiter) {
      cells.push(cell.trim());
      cell = "";
    } else cell += ch;
  }
  cells.push(cell.trim());
  return cells;
}

function parseGmt(text: string): GeneListDraft[] {
  const out: GeneListDraft[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const fields = line.split("\t");
    const name = cleanListName(fields[0] ?? "");
    const genes = dedupeGenes(fields.slice(2).map((g) => g.trim().replace(WEIGHT, "")));
    if (name && genes.length > 0) out.push({ name, genes });
  }
  return out;
}

function parseTable(text: string, delimiter: string, name: string): GeneListDraft[] {
  const rows = text
    .split(/\r?\n/)
    .filter((line) => line.trim() && !line.trimStart().startsWith("#"))
    .map((line) => splitRow(line, delimiter));
  if (rows.length === 0) return [];
  const header = rows[0]!.map((c) => c.toLowerCase());
  // gene_set,gene rows (the custom-collection upload format, and this module's CSV export)
  if (header.length >= 2 && SET_HEADERS.has(header[0]!) && GENE_HEADERS.has(header[1]!)) {
    const sets = new Map<string, string[]>();
    for (const row of rows.slice(1)) {
      const set = cleanListName(row[0] ?? "");
      if (!set || !row[1]) continue;
      if (!sets.has(set)) sets.set(set, []);
      sets.get(set)!.push(row[1]);
    }
    return [...sets].map(([set, genes]) => ({ name: set, genes: dedupeGenes(genes) })).filter((l) => l.genes.length > 0);
  }
  // Otherwise the first column is the genes (a DE or marker table works), under an optional header.
  const body = GENE_HEADERS.has(header[0]!) ? rows.slice(1) : rows;
  const genes = dedupeGenes(body.map((r) => r[0] ?? ""));
  return genes.length > 0 ? [{ name, genes }] : [];
}

/**
 * The gene lists in a file: a .gmt gives one list per set; a .csv/.tsv with a
 * gene_set,gene header one list per set, else one list from its first column;
 * a .txt is read as GMT when it has tab-separated sets, else as pasted names.
 * Throws when the file holds no genes.
 */
export function parseGeneListFile(filename: string, text: string): GeneListDraft[] {
  const clean = text.replace(/^\uFEFF/, "");
  const ext = (filename.match(/\.[^.]+$/)?.[0] ?? "").toLowerCase();
  const name = cleanListName(stem(filename));
  const gmtLike = clean.split(/\r?\n/).some((l) => !l.trimStart().startsWith("#") && (l.match(/\t/g)?.length ?? 0) >= 2);
  let lists: GeneListDraft[];
  if (ext === ".gmt" || (ext === ".txt" && gmtLike)) lists = parseGmt(clean);
  else if (ext === ".csv") lists = parseTable(clean, ",", name);
  else if (ext === ".tsv") lists = parseTable(clean, "\t", name);
  else {
    const genes = dedupeGenes(parseGeneList(clean));
    lists = genes.length > 0 ? [{ name, genes }] : [];
  }
  if (lists.length === 0) throw new Error(`${filename}: no genes found in the file.`);
  return lists.map((l) => ({ ...l, source: `file: ${filename.replace(/^.*[\\/]/, "")}` }));
}

// ---------------------------------------------------------------------------
// Writing lists out
// ---------------------------------------------------------------------------

export type GeneListExportFormat = "txt" | "csv" | "gmt";

const oneLine = (s: string) => s.replace(/[\t\r\n]+/g, " ");

/**
 * Lists as file text: txt is one gene per line (the first list only), csv is
 * gene_set,gene rows and gmt one line per list; csv and gmt read back in as
 * the same lists (here, or as a custom collection upload).
 */
export function exportGeneLists(lists: GeneListDraft[], format: GeneListExportFormat): string {
  if (format === "txt") return (lists[0]?.genes ?? []).join("\n") + "\n";
  if (format === "csv") return buildCsv(["gene_set", "gene"], lists.flatMap((l) => l.genes.map((g) => [l.name, g]))) + "\n";
  return lists.map((l) => [oneLine(l.name), oneLine(l.source ?? "scView gene list"), ...l.genes].join("\t")).join("\n") + "\n";
}

/** Save `text` as a file in the browser. */
export function downloadText(filename: string, text: string, type = "text/plain"): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8;` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// A list against a dataset
// ---------------------------------------------------------------------------

export interface GenePresence {
  /** Genes the dataset has under exactly this name, in list order */
  present: string[];
  /** The rest, in list order */
  missing: string[];
  /** Missing genes the dataset has in another case: gene -> the dataset's name (CD3E -> Cd3e) */
  otherCase: Map<string, string>;
}

/** Which of `genes` a dataset measures. Names must match exactly, as the plots and the overlay need. */
export function genePresence(genes: string[], datasetGenes: ReadonlySet<string>, byUpper?: ReadonlyMap<string, string>): GenePresence {
  const present: string[] = [];
  const missing: string[] = [];
  const otherCase = new Map<string, string>();
  for (const g of genes) {
    if (datasetGenes.has(g)) {
      present.push(g);
      continue;
    }
    missing.push(g);
    const hit = byUpper?.get(g.toUpperCase());
    if (hit) otherCase.set(g, hit);
  }
  return { present, missing, otherCase };
}

/** The dataset's gene names by upper-cased name (the first of any that collide). */
export function genesByUpper(genes: Iterable<string>): Map<string, string> {
  const out = new Map<string, string>();
  for (const g of genes) {
    const key = g.toUpperCase();
    if (!out.has(key)) out.set(key, g);
  }
  return out;
}
