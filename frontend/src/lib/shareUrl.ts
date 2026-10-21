/**
 * A link that reopens a view ("Copy link", components/common/ShareLinkButton.tsx).
 * It carries the session (lib/session.ts) without the user's own lists, as JSON
 * in the URL fragment: `#view=v1.z.<base64url>` (deflate-raw), or `#view=v1.j.`
 * (plain JSON) where the browser cannot compress. A fragment is never sent to a
 * server, so the view stays out of nginx and backend logs.
 *
 * Opening a link (hooks/useShareLinkRestore.ts) checks it with `parseSession`,
 * finds its dataset by id, else by the same cells (another scView gives the
 * same file another id), and restores it with `restoreSession`.
 */
import { getDataset, listDatasets } from "@/api/datasets";
import type { DatasetInfo } from "@/api/types";
import { describeFilter } from "@/lib/cellFilter";
import { PANEL_LABELS, type UnifiedSubtab } from "@/lib/constants";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { parseSession, SessionError, type SessionFile, type SessionSelection } from "@/lib/session";
import { restoreSession, type RestoreReport } from "@/lib/sessionRestore";

export const SHARE_PARAM = "view";
const VERSION = "v1";
/** Links longer than this lose the selection, then the camera (`buildShareUrl`). */
export const SHARE_URL_MAX = 6000;
/** A selection filter this short (JSON characters) stays in a link that is too long. */
const SHORT_FILTER = 1000;
/** Fragments past this are not read (a link scView makes is a few thousand characters). */
const MAX_FRAGMENT = 200_000;
/** Decompressed JSON past this is refused, so a crafted link cannot fill memory. */
const MAX_JSON_BYTES = 4 * 1024 * 1024;

/** Exact counts (13,836 cells), not 13.8K. */
const count = (n: number) => n.toLocaleString("en-US");

/** A link that can't be opened, with the reason in plain words. */
export class ShareLinkError extends Error {}

/** Only what a link carries: never the user's gene lists or gates, nor when it was saved. */
export function shareableSession(session: SessionFile): SessionFile {
  const out: SessionFile = { format: session.format, version: session.version };
  if (session.dataset) out.dataset = session.dataset;
  if (session.view) out.view = session.view;
  if (session.selection) out.selection = session.selection;
  if (session.de) out.de = session.de;
  if (session.colors) out.colors = session.colors;
  if (session.communication) out.communication = session.communication;
  return out;
}

// ---------------------------------------------------------------------------
// Bytes <-> text
// ---------------------------------------------------------------------------

export function toBase64Url(bytes: Uint8Array): string {
  let text = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function fromBase64Url(text: string): Uint8Array<ArrayBuffer> {
  if (!/^[A-Za-z0-9_-]*$/.test(text) || text.length % 4 === 1) {
    throw new ShareLinkError("The link is damaged (its view is not valid base64); it may have been cut short.");
  }
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (text.length % 4)) % 4);
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

/** True where the browser has deflate-raw streams (all current browsers). */
export function canCompress(): boolean {
  try {
    if (typeof CompressionStream !== "function" || typeof DecompressionStream !== "function") return false;
    new CompressionStream("deflate-raw");
    new DecompressionStream("deflate-raw");
    return true;
  } catch {
    return false;
  }
}

/** Bytes through a (de)compression stream, stopping past `maxBytes` of output. */
async function pipeBytes(
  bytes: Uint8Array<ArrayBuffer>,
  stream: CompressionStream | DecompressionStream,
  maxBytes: number,
): Promise<Uint8Array<ArrayBuffer>> {
  const writer = stream.writable.getWriter();
  // Not awaited: the reader below drains the output, else a large input stalls.
  writer.write(bytes).catch(() => {});
  writer.close().catch(() => {});
  const reader = stream.readable.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > maxBytes) {
      await reader.cancel().catch(() => {});
      throw new ShareLinkError("The link's view is too large to open.");
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/** The session's JSON as the fragment's value, compressed when the browser can. */
export async function encodeShareValue(session: SessionFile, compress = canCompress()): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(session));
  if (!compress) return `${VERSION}.j.${toBase64Url(json)}`;
  const packed = await pipeBytes(json, new CompressionStream("deflate-raw"), Number.MAX_SAFE_INTEGER);
  return `${VERSION}.z.${toBase64Url(packed)}`;
}

/** The fragment's `view=` value, if it has one. */
function shareValue(hash: string): string | null {
  const parts = hash.replace(/^#/, "").split("&");
  const part = parts.find((p) => p.startsWith(`${SHARE_PARAM}=`));
  return part == null ? null : part.slice(SHARE_PARAM.length + 1);
}

/** True when the URL fragment is a share link's (`#view=...`). */
export function hasShareHash(hash: string): boolean {
  return shareValue(hash) != null;
}

/**
 * The session a share link's fragment carries, or null when the fragment is
 * not a share link. A damaged, oversized or unknown link throws ShareLinkError.
 */
export async function readShareHash(hash: string): Promise<SessionFile | null> {
  const value = shareValue(hash);
  if (value == null) return null;
  if (value.length > MAX_FRAGMENT) throw new ShareLinkError("The link is too long to be one scView made, so it was not opened.");
  const m = /^v(\d+)\.([zj])\.(.*)$/.exec(value);
  if (!m) throw new ShareLinkError("The link is damaged: its view is not in scView's link format.");
  if (`v${m[1]}` !== VERSION) {
    throw new ShareLinkError(`The link was made by a newer scView (link format v${m[1]}); this one reads ${VERSION}.`);
  }
  // A chat app may take the full stop or bracket after a pasted link into it.
  let bytes = fromBase64Url((m[3] ?? "").replace(/[.,;:!?)\]'"]+$/, ""));
  if (m[2] === "z") {
    if (!canCompress()) throw new ShareLinkError("The link is compressed, and this browser cannot decompress it.");
    try {
      bytes = await pipeBytes(bytes, new DecompressionStream("deflate-raw"), MAX_JSON_BYTES);
    } catch (e) {
      if (e instanceof ShareLinkError) throw e;
      throw new ShareLinkError("The link is damaged (its view does not decompress); it may have been cut short.");
    }
  } else if (bytes.length > MAX_JSON_BYTES) {
    throw new ShareLinkError("The link's view is too large to open.");
  }
  let json: unknown;
  try {
    json = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new ShareLinkError("The link is damaged: its view is not readable JSON.");
  }
  // Drop what a link never carries BEFORE validating, so a hand-made link's
  // gates or gene lists are ignored rather than judged.
  if (json && typeof json === "object" && !Array.isArray(json)) {
    const rest: Record<string, unknown> = { ...(json as Record<string, unknown>) };
    for (const k of ["gene_lists", "gates", "saved_at"]) delete rest[k];
    json = rest;
  }
  try {
    return shareableSession(parseSession(json));
  } catch (e) {
    if (e instanceof SessionError) throw new ShareLinkError(`The link's view can't be used: ${e.message}`);
    throw e;
  }
}

// ---------------------------------------------------------------------------
// Making a link
// ---------------------------------------------------------------------------

function selectionCount(sel: Exclude<SessionSelection, { filter: unknown }>): number {
  if ("indices" in sel) return sel.indices.length;
  let n = 0;
  for (const ch of atob(sel.bitmask)) {
    for (let b = ch.charCodeAt(0); b; b &= b - 1) n++;
  }
  return n;
}

/** The selection in words: `the selection of 312 cells`, or its filter. */
export function describeSelection(sel: SessionSelection): string {
  if ("filter" in sel) return `the selection “${describeFilter(sel.filter)}”`;
  const n = selectionCount(sel);
  return `the selection of ${count(n)} ${n === 1 ? "cell" : "cells"}`;
}

const isShortFilter = (sel: SessionSelection | null | undefined) =>
  !!sel && "filter" in sel && JSON.stringify(sel.filter).length <= SHORT_FILTER;

interface DropStep {
  applies: (s: SessionFile) => boolean;
  drop: (s: SessionFile) => SessionFile;
  label: (s: SessionFile) => string;
}

const withoutSelection = (s: SessionFile): SessionFile => {
  const out = { ...s };
  delete out.selection;
  return out;
};

/** What goes first when a link is too long; each step only while it still is. */
const DROP_STEPS: DropStep[] = [
  {
    applies: (s) => !!s.selection && !isShortFilter(s.selection),
    drop: withoutSelection,
    label: (s) => describeSelection(s.selection!),
  },
  {
    applies: (s) => !!s.view?.camera,
    drop: (s) => ({ ...s, view: { ...s.view, camera: null } }),
    label: () => "the zoom and position (camera)",
  },
  {
    // A short filter goes only after the camera.
    applies: (s) => !!s.selection,
    drop: withoutSelection,
    label: (s) => describeSelection(s.selection!),
  },
  {
    // Only a long gene list can still be too long here.
    applies: (s) => s.view?.overlay?.type === "score",
    drop: (s) => ({ ...s, view: { ...s.view, overlay: null } }),
    label: (s) => {
      const o = s.view!.overlay as { name: string; genes: string[] };
      return `the gene-set score overlay “${o.name}” (${count(o.genes.length)} genes)`;
    },
  },
];

type Where = Pick<Location, "origin" | "pathname" | "search">;

/**
 * A link to the session's view on this page. Past `maxLength` characters the
 * selection (unless it is a short filter) and then the camera are left out;
 * `dropped` says what, in words.
 */
export async function buildShareUrl(
  session: SessionFile,
  location: Where,
  { compress = canCompress(), maxLength = SHARE_URL_MAX }: { compress?: boolean; maxLength?: number } = {},
): Promise<{ url: string; dropped: string[] }> {
  const base = `${location.origin}${location.pathname}${location.search}#${SHARE_PARAM}=`;
  let shared = shareableSession(session);
  let url = base + (await encodeShareValue(shared, compress));
  const dropped: string[] = [];
  for (const step of DROP_STEPS) {
    if (url.length <= maxLength) break;
    if (!step.applies(shared)) continue;
    dropped.push(step.label(shared));
    shared = step.drop(shared);
    url = base + (await encodeShareValue(shared, compress));
  }
  return { url, dropped };
}

const SUBTAB_LABELS: Partial<Record<UnifiedSubtab, string>> = {
  markers: "Markers",
  expression: "Expression",
  protein: "Protein",
  genesets: "Gene Sets",
  enrichment: "Enrichment",
  de: "DE",
  activity: "Activity",
  genelists: "Gene lists",
  communication: "Communication",
  report: "Report",
};

/** What opening the link restores, one phrase each (the Copy link popover lists them). */
export function describeSharedView(session: SessionFile): string[] {
  const out: string[] = [];
  const d = session.dataset;
  if (d) {
    const cells = d.n_cells != null ? ` (${count(d.n_cells)} cells)` : "";
    out.push(`dataset “${d.name || d.id}”${cells}`);
  }
  const v = session.view ?? {};
  if (v.panel) {
    const tab = v.panel === "unified" && v.subtab ? ` (${SUBTAB_LABELS[v.subtab] ?? v.subtab} tab)` : "";
    out.push(`the ${PANEL_LABELS[v.panel]} panel${tab}`);
  }
  if (v.embedding) {
    const by = v.color_by ? ` coloured by ${prettyObsLabel(v.color_by)}` : "";
    out.push(`${prettyEmbeddingLabel(v.embedding)}${by}`);
  }
  const o = v.overlay;
  if (o?.type === "expression") out.push(`the ${o.gene} ${o.layer ? "protein" : "expression"} overlay`);
  else if (o?.type === "score") out.push(`the gene-set score overlay “${o.name}”${o.method === "aucell" ? " (AUCell)" : ""} (${count(o.genes.length)} genes)`);
  else if (o?.type === "activity") out.push(`the ${o.source} activity overlay (${o.network === "collectri" ? "CollecTRI" : "PROGENy"})`);
  else if (o?.type === "reliability") out.push(`the plot reliability overlay (${prettyEmbeddingLabel(o.embedding)})`);
  if (v.highlight) out.push(`the highlighted group ${prettyObsLabel(v.highlight.column)} = ${v.highlight.value}`);
  if (v.camera) out.push("the zoom and position");
  if (session.selection) out.push(describeSelection(session.selection));
  if (session.de) out.push("the DE settings");
  if (session.colors) out.push("the colours and point settings");
  return out;
}

// ---------------------------------------------------------------------------
// Opening a link
// ---------------------------------------------------------------------------

export interface ShareDeps {
  getDataset: (id: string) => Promise<DatasetInfo>;
  listDatasets: () => Promise<DatasetInfo[]>;
  restoreSession: (session: SessionFile) => Promise<RestoreReport>;
}

// Read on use, not at import (a test's partial mock of the API may lack some).
const defaultDeps = (): ShareDeps => ({ getDataset, listDatasets, restoreSession });
const is404 = (e: unknown) => e instanceof Error && /^API error 404\b/.test(e.message);
const quote = (s: string) => `“${s}”`;

/**
 * The library's dataset for a link: the one with its id, else one with the
 * same cells (`cells_fingerprint` and `n_cells`). Null when there is none.
 */
export async function matchSharedDataset(
  saved: NonNullable<SessionFile["dataset"]>,
  deps: Pick<ShareDeps, "getDataset" | "listDatasets"> = defaultDeps(),
): Promise<{ dataset: DatasetInfo; by: "id" | "cells" } | null> {
  try {
    return { dataset: await deps.getDataset(saved.id), by: "id" };
  } catch (e) {
    // Only "no such dataset" looks further; a busy or unreachable server is an error.
    if (!is404(e)) throw e;
  }
  if (!saved.cells_fingerprint || saved.n_cells == null) return null;
  const library = await deps.listDatasets();
  for (const d of library) {
    if (d.id === saved.id || d.n_cells !== saved.n_cells || d.status !== "ready") continue;
    // The list knows the fingerprint only once the dataset was opened; else ask.
    let info: DatasetInfo | null = d.cells_fingerprint ? d : null;
    if (!info) info = await deps.getDataset(d.id).catch(() => null);
    if (info?.cells_fingerprint === saved.cells_fingerprint) return { dataset: info, by: "cells" };
  }
  return null;
}

export type ShareOutcome =
  | (RestoreReport & { retry?: false })
  | { headline: string; notes: string[]; level: "error"; retry: boolean };

const reason = (e: unknown) => (e instanceof Error ? e.message.replace(/^API error /, "") : String(e));
/** A server error: the link stays in the address bar, so a reload tries again. */
const retryable = (text: string): ShareOutcome => ({
  headline: `${text} Reload the page to try again.`,
  notes: [],
  level: "error",
  retry: true,
});

/**
 * Open the view a link's fragment carries (null when the fragment is not a
 * link). Never throws: a damaged link, a dataset this library lacks or a server
 * error comes back as the outcome's headline; `retry` says whether reloading
 * the page could help.
 */
export async function openShareHash(hash: string, deps: ShareDeps = defaultDeps()): Promise<ShareOutcome | null> {
  let session: SessionFile | null;
  try {
    session = await readShareHash(hash);
  } catch (e) {
    const headline = e instanceof ShareLinkError ? e.message : `The link could not be read: ${String(e)}`;
    return { headline, notes: [], level: "error", retry: false };
  }
  if (!session) return null;
  const saved = session.dataset;
  let match: Awaited<ReturnType<typeof matchSharedDataset>> = null;
  if (saved) {
    try {
      match = await matchSharedDataset(saved, deps);
    } catch (e) {
      return retryable(`The link's dataset could not be looked up (${reason(e)}).`);
    }
    if (!match) {
      const cells = saved.n_cells != null ? ` (${count(saved.n_cells)} cells)` : "";
      return {
        headline:
          `This link is for dataset ${quote(saved.name || saved.id)}${cells}, which is not in this scView's library. ` +
          "Load that file, then open the link again.",
        notes: [],
        level: "warning",
      };
    }
  }
  try {
    if (!saved || !match || match.by === "id") return await deps.restoreSession(session);
    const report = await deps.restoreSession({ ...session, dataset: { ...saved, id: match.dataset.id } });
    if (saved.name && saved.name !== match.dataset.name) {
      report.headline += ` It has the same cells as ${quote(saved.name)}, the dataset the link was made on.`;
    }
    return report;
  } catch (e) {
    // restoreSession's errors are plain sentences already.
    return retryable(reason(e));
  }
}

/** Take the fragment off the address, so a reload does not apply the link again. */
export function clearShareHash() {
  const { pathname, search } = window.location;
  window.history.replaceState(window.history.state, "", `${pathname}${search}`);
}
