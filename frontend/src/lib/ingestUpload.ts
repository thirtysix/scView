/**
 * Folder uploads for the import wizard: every file goes with its path inside the
 * chosen folder (Cell Ranger's raw and filtered folders both hold a barcodes.tsv.gz),
 * and the files scView never reads are left out before anything is sent (a BAM is
 * gigabytes). The names mirror the backend's `core/ingestion/paths.py`.
 */

export interface UploadItem {
  file: File;
  /** The path inside the uploaded folder ("pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz"), or the name. */
  path: string;
}

export interface SkippedUpload {
  path: string;
  reason: string;
  bytes: number;
}

/** A picked file's path inside the chosen folder (`webkitRelativePath`), else its name. */
export function relativePath(file: File): string {
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return rel || file.name;
}

export function itemsFromFiles(files: Iterable<File>): UploadItem[] {
  return Array.from(files, (file) => ({ file, path: relativePath(file) }));
}

const SYSTEM = "system files";
const ALIGNMENTS = "read alignments (scView starts from the count matrix)";
const REPORT = "Cell Ranger's HTML report";
const LOUPE = "Loupe Browser files";
const MOLECULES = "per-molecule read information, not a count matrix";
const SPATIAL_EXTRA = "Space Ranger alignment images and tables, not used by scView";

const JUNK_NAMES: Record<string, string> = {
  ".ds_store": SYSTEM,
  "thumbs.db": SYSTEM,
  "desktop.ini": SYSTEM,
  "web_summary.html": REPORT,
  "molecule_info.h5": MOLECULES,
  "sample_molecule_info.h5": MOLECULES,
  "aligned_fiducials.jpg": SPATIAL_EXTRA,
  "detected_tissue_image.jpg": SPATIAL_EXTRA,
  "aligned_tissue_image.jpg": SPATIAL_EXTRA,
  "cytassist_image.tiff": SPATIAL_EXTRA,
  "spatial_enrichment.csv": SPATIAL_EXTRA,
  "barcode_fluorescence_intensity.csv": SPATIAL_EXTRA,
  "fiducial_positions_list.txt": SPATIAL_EXTRA,
  "probe_set.csv": "the probe set of the run, not data",
};
const JUNK_SUFFIXES: [string, string][] = [
  [".bam", ALIGNMENTS],
  [".bai", ALIGNMENTS],
  [".bam.bai", ALIGNMENTS],
  [".csi", ALIGNMENTS],
  [".cram", ALIGNMENTS],
  [".crai", ALIGNMENTS],
  [".cloupe", LOUPE],
  [".vloupe", LOUPE],
];

/** Why a file of a folder is not uploaded, or null to upload it. */
export function junkReason(path: string): string | null {
  const name = (path.split(/[\\/]/).pop() ?? path).toLowerCase();
  if (name in JUNK_NAMES) return JUNK_NAMES[name]!;
  if (name.startsWith("._")) return SYSTEM;
  for (const [suffix, reason] of JUNK_SUFFIXES) if (name.endsWith(suffix)) return reason;
  return null;
}

/** The files to send, and the ones left out (with why and their size). */
export function splitJunk(items: UploadItem[]): { keep: UploadItem[]; skipped: SkippedUpload[] } {
  const keep: UploadItem[] = [];
  const skipped: SkippedUpload[] = [];
  for (const it of items) {
    const reason = junkReason(it.path);
    if (reason) skipped.push({ path: it.path, reason, bytes: it.file.size });
    else keep.push(it);
  }
  return { keep, skipped };
}

export function formatBytes(n: number): string {
  for (const [unit, scale] of [["GB", 1024 ** 3], ["MB", 1024 ** 2], ["KB", 1024]] as const) {
    if (n >= scale) return `${(n / scale).toFixed(1)} ${unit}`;
  }
  return `${n} bytes`;
}

/** One sentence for the files left out, grouped by reason. */
export function skippedNote(skipped: SkippedUpload[]): string | null {
  if (skipped.length === 0) return null;
  const groups = new Map<string, SkippedUpload[]>();
  for (const s of skipped) groups.set(s.reason, [...(groups.get(s.reason) ?? []), s]);
  const parts = [...groups].map(([reason, items]) => {
    const names = items.map((s) => s.path.split("/").pop() ?? s.path);
    const shown = names.slice(0, 3).join(", ") + (names.length > 3 ? ` and ${names.length - 3} more` : "");
    return `${shown} (${reason})`;
  });
  const total = skipped.reduce((a, s) => a + s.bytes, 0);
  const n = skipped.length;
  return `Not uploaded: ${n} file${n === 1 ? "" : "s"} scView does not read, ${formatBytes(total)}: ${parts.join("; ")}.`;
}

// --- Drag and drop of folders ------------------------------------------------

interface FsEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  fullPath: string;
}
interface FsFileEntry extends FsEntry {
  file(ok: (f: File) => void, fail?: (e: unknown) => void): void;
}
interface FsDirEntry extends FsEntry {
  createReader(): { readEntries(ok: (e: FsEntry[]) => void, fail?: (e: unknown) => void): void };
}

async function walk(entry: FsEntry, out: UploadItem[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((ok, fail) => (entry as FsFileEntry).file(ok, fail));
    out.push({ file, path: entry.fullPath.replace(/^\/+/, "") || file.name });
    return;
  }
  if (!entry.isDirectory) return;
  const reader = (entry as FsDirEntry).createReader();
  // readEntries returns at most ~100 entries per call: read until it returns none.
  for (;;) {
    const batch = await new Promise<FsEntry[]>((ok, fail) => reader.readEntries(ok, fail));
    if (batch.length === 0) break;
    for (const child of batch) await walk(child, out);
  }
}

/**
 * The dropped files with their folder paths. A dropped folder is walked (Chrome,
 * Edge, Firefox and Safari all give `webkitGetAsEntry`); plain files keep their
 * names. Call it from the drop handler itself: the entries are only readable during
 * the event, so they are taken before the first await.
 */
export async function itemsFromDrop(dt: Pick<DataTransfer, "files"> & { items?: DataTransferItemList }): Promise<UploadItem[]> {
  const entries: FsEntry[] = [];
  for (const item of Array.from(dt.items ?? [])) {
    const get = (item as DataTransferItem & { webkitGetAsEntry?: () => FsEntry | null }).webkitGetAsEntry;
    const entry = item.kind === "file" && get ? get.call(item) : null;
    if (entry) entries.push(entry);
  }
  if (!entries.some((e) => e.isDirectory)) return itemsFromFiles(Array.from(dt.files ?? []));
  const out: UploadItem[] = [];
  for (const e of entries) await walk(e, out);
  return out;
}
