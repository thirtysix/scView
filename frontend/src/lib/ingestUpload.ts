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
const XENIUM_EXTRA = "Xenium transcripts, boundaries, stains and Explorer files, not used by scView";
const XENIUM_ZIP = "a zipped Xenium output folder is not read: unzip it and drop the folder instead";
const HD_RAW = "the raw matrix, every bin of the slide and most of them empty: the filtered one is read";
const HD_COPY = "the same matrix as filtered_feature_bc_matrix.h5, which is read";
const HD_ANALYSIS = "Space Ranger's own analysis (clustering, PCA, UMAP, DE): scView computes these itself";
const HD_OTHER = "a Space Ranger output scView does not read";

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
  "transcripts.parquet": XENIUM_EXTRA,
  "transcripts.csv.gz": XENIUM_EXTRA,
  "morphology.ome.tif": XENIUM_EXTRA,
  "cell_boundaries.parquet": XENIUM_EXTRA,
  "cell_boundaries.csv.gz": XENIUM_EXTRA,
  "nucleus_boundaries.parquet": XENIUM_EXTRA,
  "nucleus_boundaries.csv.gz": XENIUM_EXTRA,
  "analysis_summary.html": XENIUM_EXTRA,
  "gene_panel.json": XENIUM_EXTRA,
  "raw_probe_bc_matrix.h5": "the raw per-probe matrix, not used by scView",
  "feature_slice.h5": "Space Ranger's per-gene slide images, not used by scView",
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
  [".zarr.zip", XENIUM_EXTRA],
];

const HD_DIR = /^square_(\d{3})um$/i;
const HD_SPATIAL = new Set([
  "tissue_positions.parquet",
  "tissue_positions.csv",
  "scalefactors_json.json",
  "tissue_hires_image.png",
  "tissue_lowres_image.png",
]);

/**
 * Inside a Visium HD square_NNNum/ folder only the filtered .h5 and four files of
 * spatial/ are read: why any other is left out; null to keep it or outside such a folder.
 */
function hdJunk(path: string): string | null {
  const parts = path.toLowerCase().split(/[\\/]/);
  const at = parts.slice(0, -1).findIndex((p) => HD_DIR.test(p));
  if (at < 0) return null;
  const dirs = parts.slice(at + 1, -1);
  const name = parts[parts.length - 1]!;
  if (dirs.length === 0 && name.endsWith("filtered_feature_bc_matrix.h5")) return null;
  if (dirs.length === 1 && dirs[0] === "spatial" && HD_SPATIAL.has(name)) return null;
  if (dirs.includes("analysis")) return HD_ANALYSIS;
  if (name.includes("raw") || dirs.some((d) => d.includes("raw"))) return HD_RAW;
  if (dirs.some((d) => d.includes("filtered_feature_bc_matrix"))) return HD_COPY;
  return HD_OTHER;
}

/** Why a file of a folder is not uploaded, or null to upload it. */
export function junkReason(path: string): string | null {
  const name = (path.split(/[\\/]/).pop() ?? path).toLowerCase();
  if (name in JUNK_NAMES) return JUNK_NAMES[name]!;
  if (name.startsWith("._")) return SYSTEM;
  for (const [suffix, reason] of JUNK_SUFFIXES) if (name.endsWith(suffix)) return reason;
  if (name.endsWith(".zip") && name.includes("xenium")) return XENIUM_ZIP; // 10 GB; the server would refuse it
  return hdJunk(path);
}

/** The morphology images of a Xenium folder, in the order they are preferred (`core/xenium.py`). */
const XENIUM_IMAGES = [
  "morphology_focus.ome.tif",
  "morphology_focus/morphology_focus_0000.ome.tif",
  "morphology_mip.ome.tif",
];
const XENIUM_MARK = "experiment.xenium";

/**
 * Files of a Xenium output folder (one holding an experiment.xenium) that are never read: its
 * own analysis, auxiliary outputs and every image but the one used, a few GB of stains. By
 * folder, not by name: "analysis/" is any user's folder name elsewhere.
 */
function xeniumExtras(paths: string[]): Set<string> {
  const split = (p: string) => p.replace(/\\/g, "/");
  const roots = paths
    .map(split)
    .filter((p) => p.toLowerCase().endsWith(`/${XENIUM_MARK}`))
    .map((p) => p.slice(0, -XENIUM_MARK.length));
  const out = new Set<string>();
  if (roots.length === 0) return out;
  const low = new Set(paths.map((p) => split(p).toLowerCase()));
  for (const raw of paths) {
    const p = split(raw);
    const root = roots.find((r) => p.startsWith(r));
    if (!root) continue;
    const inner = p.slice(root.length).toLowerCase();
    const top = inner.split("/")[0];
    if (top === "analysis" || top === "aux_outputs") out.add(raw);
    else if (inner.endsWith(".ome.tif")) {
      const best = XENIUM_IMAGES.find((img) => low.has(`${root}${img}`.toLowerCase()));
      if (inner !== best) out.add(raw);
    }
  }
  return out;
}

/** The files to send, and the ones left out (with why and their size). */
export function splitJunk(items: UploadItem[]): { keep: UploadItem[]; skipped: SkippedUpload[] } {
  const keep: UploadItem[] = [];
  const skipped: SkippedUpload[] = [];
  const extras = xeniumExtras(items.map((it) => it.path));
  for (const it of items) {
    const reason = junkReason(it.path) ?? (extras.has(it.path) ? XENIUM_EXTRA : null);
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

// --- Visium HD: one bin size of a run ----------------------------------------
// The same rules as the backend's spatial.py (hd_folder, hd_pick_bin, unsupported),
// applied before anything is sent: a 2 um matrix, or a whole archive, is never uploaded.

export const HD_DEFAULT_BIN_UM = 16;
const HD_MIN_BIN_UM = 8;
const HD_ARCHIVE = /binned_outputs\.(tar\.gz|tgz|tar)$/i;
// Bins on a whole Visium HD capture area: 3,350 x 3,350 at 2 um.
const HD_AREA_UM = 6700;

export interface HdBinOption {
  um: number;
  /** Size of the bin folder's filtered matrix */
  matrixBytes: number;
  /** Bins on the whole capture area: the most there can be, whatever the tissue covers */
  maxBins: number;
  /** Too many bins to read: offered as a reason, not a choice */
  refused: boolean;
}

/** The run's folder (above square_NNNum/, without binned_outputs) and bin size of a path in such a folder. */
export function hdFolder(path: string): { run: string; um: number } | null {
  const dirs = path.split(/[\\/]/).slice(0, -1);
  const at = dirs.findIndex((d) => HD_DIR.test(d));
  if (at < 0) return null;
  const run = dirs.slice(0, at);
  if (run[run.length - 1]?.toLowerCase() === "binned_outputs") run.pop();
  return { run: run.join("/"), um: parseInt(HD_DIR.exec(dirs[at]!)![1]!, 10) };
}

/** The bin sizes in these files, finest first; empty for anything but a Visium HD run's folder. */
export function hdBinOptions(items: UploadItem[]): HdBinOption[] {
  const sizes = new Map<number, number>();
  for (const it of items) {
    const hit = hdFolder(it.path);
    if (!hit) continue;
    const isMatrix = (it.path.split(/[\\/]/).pop() ?? "").toLowerCase().endsWith("filtered_feature_bc_matrix.h5");
    sizes.set(hit.um, (sizes.get(hit.um) ?? 0) + (isMatrix && !hdJunk(it.path) ? it.file.size : 0));
  }
  return [...sizes]
    .sort(([a], [b]) => a - b)
    .map(([um, matrixBytes]) => ({
      um,
      matrixBytes,
      maxBins: Math.round((HD_AREA_UM / um) ** 2),
      refused: um < HD_MIN_BIN_UM,
    }));
}

/** The size offered first: 8 um, else the finest that can be read. */
export function hdDefaultBin(options: HdBinOption[]): number | null {
  const usable = options.filter((o) => !o.refused);
  return (usable.find((o) => o.um === HD_DEFAULT_BIN_UM) ?? usable[0])?.um ?? null;
}

/** The sentence for files that are not uploaded at all, or null: the server's own refusals, said first. */
export function hdRefusal(items: UploadItem[]): string | null {
  const zipped = items.find((it) => /xenium[^\\/]*\.zip$/i.test(it.path));
  if (zipped) return `${zipped.path.split(/[\\/]/).pop()}: ${XENIUM_ZIP}.`; // said before the size limit
  const archive = items.find((it) => HD_ARCHIVE.test(it.path));
  if (archive) {
    const name = archive.path.split(/[\\/]/).pop() ?? archive.path;
    return `${name} is a Visium HD binned_outputs archive, which scView does not read: unpack it and drop the binned_outputs folder (or one square_NNNum folder) instead.`;
  }
  const options = hdBinOptions(items);
  if (options.length > 0 && hdDefaultBin(options) == null) {
    return `The ${options[0]!.um} um bins of a Visium HD run are millions of observations (6.3 million on a mouse brain section), far past the 1.3 million cells scView has been tested on: use the square_008um or square_016um folder instead.`;
  }
  if (options.length === 0 && items.some((it) => it.path.toLowerCase().split(/[\\/]/).slice(0, -1).includes("segmented_outputs"))) {
    return "Space Ranger 4's segmented_outputs (cell segmentation) is not read yet: upload the run's binned_outputs folder instead, which reads as 8 or 16 um bins.";
  }
  return null;
}

/** One bin size of each run: the rest of the run's folder goes in `skipped`, with why. */
export function keepHdBin(items: UploadItem[], um: number): { keep: UploadItem[]; skipped: SkippedUpload[] } {
  const runs = new Set(items.map((it) => hdFolder(it.path)?.run).filter((r): r is string => !!r));
  const keep: UploadItem[] = [];
  const skipped: SkippedUpload[] = [];
  for (const it of items) {
    const hit = hdFolder(it.path);
    if (hit) {
      if (hit.um === um) keep.push(it);
      else
        skipped.push({
          path: it.path,
          reason: `another bin size of the run: one dataset per bin size, the ${um} um bins are read`,
          bytes: it.file.size,
        });
    } else if ([...runs].some((r) => it.path.startsWith(`${r}/`))) {
      skipped.push({
        path: it.path,
        reason: "the rest of the Space Ranger output folder, not part of the bins read",
        bytes: it.file.size,
      });
    } else keep.push(it);
  }
  return { keep, skipped };
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
