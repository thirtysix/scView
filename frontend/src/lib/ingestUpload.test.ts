import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  HD_DEFAULT_BIN_UM,
  hdSegRuns,
  keepHdCells,
  hdBinOptions,
  hdDefaultBin,
  hdRefusal,
  itemsFromDrop,
  itemsFromFiles,
  junkReason,
  keepHdBin,
  relativePath,
  skippedNote,
  splitJunk,
  type UploadItem,
} from "./ingestUpload";

/** A File as a folder pick gives it: with its path inside the chosen folder. */
function picked(path: string, size = 3): File {
  const f = new File(["x".repeat(size)], path.split("/").pop()!);
  Object.defineProperty(f, "webkitRelativePath", { value: path });
  return f;
}

describe("Xenium output folders", () => {
  const outs = [
    "run/outs/experiment.xenium",
    "run/outs/cell_feature_matrix.h5",
    "run/outs/cells.parquet",
    "run/outs/metrics_summary.csv",
    "run/outs/morphology_focus.ome.tif",
    "run/outs/cell_boundaries.parquet",
    "run/outs/nucleus_boundaries.csv.gz",
  ];
  const heavy = [
    "run/outs/transcripts.parquet",
    "run/outs/transcripts.csv.gz",
    "run/outs/transcripts.zarr.zip",
    "run/outs/morphology.ome.tif",
    "run/outs/morphology_mip.ome.tif",
    "run/outs/cells.zarr.zip",
    "run/outs/analysis_summary.html",
    "run/outs/gene_panel.json",
    "run/outs/analysis/clustering/clusters.csv",
    "run/outs/aux_outputs/overlay.ome.tif",
  ];

  it("sends the few files a run needs and leaves out the heavy ones, with their reason", () => {
    const { keep, skipped } = splitJunk(itemsFromFiles([...outs, ...heavy].map((p) => picked(p))));
    expect(keep.map((i) => i.path)).toEqual(outs);
    expect(skipped.map((s) => s.path)).toEqual(heavy);
    expect(skipped.every((s) => s.reason.includes("Xenium"))).toBe(true);
  });

  it("sends a boundaries .csv.gz only when there is no parquet of it", () => {
    const paths = ["o/experiment.xenium", "o/cell_boundaries.parquet", "o/cell_boundaries.csv.gz", "o/nucleus_boundaries.csv.gz"];
    const { keep, skipped } = splitJunk(itemsFromFiles(paths.map((p) => picked(p))));
    expect(keep.map((i) => i.path)).toEqual(["o/experiment.xenium", "o/cell_boundaries.parquet", "o/nucleus_boundaries.csv.gz"]);
    expect(skipped.map((s) => s.reason)).toEqual(["the same outlines as cell_boundaries.parquet, which is read"]);
  });

  it("keeps the first stain of XOA 2+ and the mip image when there is no focus image", () => {
    const paths = [
      "o/experiment.xenium",
      "o/morphology_focus/morphology_focus_0000.ome.tif",
      "o/morphology_focus/morphology_focus_0001.ome.tif",
      "o/morphology_mip.ome.tif",
    ];
    const { keep } = splitJunk(itemsFromFiles(paths.map((p) => picked(p))));
    expect(keep.map((i) => i.path)).toEqual(paths.slice(0, 2));
    const mip = splitJunk(itemsFromFiles(["o/experiment.xenium", "o/morphology_mip.ome.tif"].map((p) => picked(p))));
    expect(mip.keep).toHaveLength(2);
  });

  it("leaves a folder called analysis alone unless a Xenium run holds it", () => {
    const paths = ["analysis/pbmc/filtered_feature_bc_matrix.h5", "analysis/notes.ome.tif"];
    expect(splitJunk(itemsFromFiles(paths.map((p) => picked(p)))).keep).toHaveLength(2);
  });

  it("leaves out Cell Ranger ARC's peak files and sample-prefixed reports, keeping its data and side files", () => {
    const arc = "5k_Mouse_Kidney_MO_gemx_";
    for (const p of [`${arc}atac_peaks.bed`, `${arc}atac_peak_annotation.tsv`, `${arc}web_summary.html`,
      `${arc}gex_molecule_info.h5`, `${arc}atac_cut_sites.bigwig`]) {
      expect(junkReason(p), p).not.toBeNull();
    }
    expect(junkReason(`${arc}atac_peaks.bed`)).toContain("same peaks are in the matrix");
    for (const p of [`${arc}filtered_feature_bc_matrix.h5`, `${arc}atac_fragments.tsv.gz`,
      `${arc}atac_fragments.tsv.gz.tbi`, `${arc}per_barcode_metrics.csv`, `${arc}summary.csv`]) {
      expect(junkReason(p), p).toBeNull();
    }
  });

  it("never sends a zipped run", () => {
    expect(junkReason("Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip")).toContain("drop the folder");
    expect(junkReason("samples.zip")).toBeNull();
  });
});

describe("folder upload helpers", () => {
  it("takes each file's path inside the folder, else its name", () => {
    expect(relativePath(picked("pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz"))).toBe(
      "pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz",
    );
    expect(relativePath(new File(["x"], "a.h5ad"))).toBe("a.h5ad");
    expect(itemsFromFiles([picked("o/x.h5")]).map((i) => i.path)).toEqual(["o/x.h5"]);
  });

  it("leaves out BAMs, reports, Loupe and system files, with their size", () => {
    for (const p of ["o/possorted_genome_bam.bam", "o/possorted_genome_bam.bam.bai", "o/.DS_Store",
      "o/web_summary.html", "o/cloupe.cloupe", "o/vdj_t/vloupe.vloupe", "o/molecule_info.h5", "o/._x.csv"]) {
      expect(junkReason(p), p).not.toBeNull();
    }
    for (const p of ["o/filtered_feature_bc_matrix.h5", "o/metrics_summary.csv", "o/vdj_t/filtered_contig_annotations.csv"]) {
      expect(junkReason(p), p).toBeNull();
    }
    const items = itemsFromFiles([
      picked("o/filtered_feature_bc_matrix.h5"),
      picked("o/possorted_genome_bam.bam", 3 * 1024 * 1024),
      picked("o/possorted_genome_bam.bam.bai", 1024 * 1024),
      picked("o/web_summary.html", 10),
    ]);
    const { keep, skipped } = splitJunk(items);
    expect(keep.map((i) => i.path)).toEqual(["o/filtered_feature_bc_matrix.h5"]);
    expect(skippedNote(skipped)).toBe(
      "Not uploaded: 3 files scView does not read, 4.0 MB: possorted_genome_bam.bam, possorted_genome_bam.bam.bai " +
        "(read alignments (scView starts from the count matrix)); web_summary.html (Cell Ranger's HTML report).",
    );
    expect(skippedNote([])).toBeNull();
  });

  it("walks a dropped folder in batches, keeping the paths", async () => {
    const file = (fullPath: string) => ({
      isFile: true, isDirectory: false, name: fullPath.split("/").pop()!, fullPath,
      file: (ok: (f: File) => void) => ok(new File(["x"], fullPath.split("/").pop()!)),
    });
    const dir = (fullPath: string, batches: unknown[][]) => ({
      isFile: false, isDirectory: true, name: fullPath.split("/").pop()!, fullPath,
      createReader: () => {
        let i = 0;
        return { readEntries: (ok: (e: unknown[]) => void) => ok(batches[i++] ?? []) };
      },
    });
    const raw = dir("/outs/raw_feature_bc_matrix", [[file("/outs/raw_feature_bc_matrix/barcodes.tsv.gz")]]);
    const outs = dir("/outs", [
      [file("/outs/filtered_feature_bc_matrix/barcodes.tsv.gz")],
      [raw],
    ]);
    const dt = {
      files: [] as unknown as FileList,
      items: [{ kind: "file", webkitGetAsEntry: () => outs }] as unknown as DataTransferItemList,
    };
    const items = await itemsFromDrop(dt);
    expect(items.map((i) => i.path)).toEqual([
      "outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
      "outs/raw_feature_bc_matrix/barcodes.tsv.gz",
    ]);
    // Plain files dropped (no folder): their names, from dataTransfer.files.
    const plain = await itemsFromDrop({ files: [new File(["x"], "a.h5ad")] as unknown as FileList });
    expect(plain.map((i) => i.path)).toEqual(["a.h5ad"]);
  });

  it("skips the same names as the server (core/ingestion/paths.py)", () => {
    const py = readFileSync(resolve(__dirname, "../../../backend/src/scview/core/ingestion/paths.py"), "utf8");
    const block = (start: string) => py.slice(py.indexOf(start), py.indexOf("\n}", py.indexOf(start)) + 2);
    const names = [...block("_JUNK_NAMES = {").matchAll(/"([^"]+)":/g)].map((m) => m[1]!);
    const suffixes = [...py.slice(py.indexOf("_JUNK_SUFFIXES = (")).split("\n)\n")[0]!.matchAll(/\("([^"]+)",/g)]
      .map((m) => m[1]!);
    expect(names.length).toBeGreaterThan(10);
    expect(suffixes.length).toBeGreaterThan(5);
    for (const n of names) expect(junkReason(`outs/spatial/${n}`), n).not.toBeNull();
    for (const x of suffixes) expect(junkReason(`outs/file${x}`), x).not.toBeNull();
  });
});

describe("Visium HD folders", () => {
  const f = (path: string, size = 10): UploadItem => ({ file: new File([new Uint8Array(size)], path.split("/").pop()!), path });
  const bins = (run: string, um: number, size = 10) => {
    const d = `${run}square_${String(um).padStart(3, "0")}um`;
    return [
      f(`${d}/filtered_feature_bc_matrix.h5`, size),
      f(`${d}/raw_feature_bc_matrix.h5`, 999),
      f(`${d}/raw_probe_bc_matrix.h5`, 999),
      f(`${d}/filtered_feature_bc_matrix/matrix.mtx.gz`),
      f(`${d}/analysis/pca/components.csv`),
      f(`${d}/cloupe.cloupe`),
      f(`${d}/spatial/tissue_positions.parquet`),
      f(`${d}/spatial/scalefactors_json.json`),
      f(`${d}/spatial/tissue_hires_image.png`),
      f(`${d}/spatial/tissue_lowres_image.png`),
      f(`${d}/spatial/aligned_tissue_image.jpg`),
      f(`${d}/spatial/detected_tissue_image.jpg`),
      f(`${d}/spatial/cytassist_image.tiff`),
    ];
  };

  it("leaves everything out of a bin folder but its filtered matrix and four spatial files", () => {
    const { keep, skipped } = splitJunk(bins("", 8));
    expect(keep.map((k) => k.path).sort()).toEqual([
      "square_008um/filtered_feature_bc_matrix.h5",
      "square_008um/spatial/scalefactors_json.json",
      "square_008um/spatial/tissue_hires_image.png",
      "square_008um/spatial/tissue_lowres_image.png",
      "square_008um/spatial/tissue_positions.parquet",
    ]);
    expect(skipped).toHaveLength(8);
    expect(skipped.every((s) => s.reason.length > 0)).toBe(true);
  });

  it("keeps a Cell Ranger raw matrix, which is not junk outside a bin folder", () => {
    expect(junkReason("pbmc/outs/raw_feature_bc_matrix.h5")).toBeNull();
  });

  it("offers the bin sizes of a folder, finest first, 8 um first choice, 2 um refused", () => {
    const options = hdBinOptions([
      ...bins("run/outs/binned_outputs/", 2, 181),
      ...bins("run/outs/binned_outputs/", 8, 112),
      ...bins("run/outs/binned_outputs/", 16, 90),
    ]);
    expect(options.map((o) => [o.um, o.matrixBytes, o.refused])).toEqual([
      [2, 181, true],
      [8, 112, false],
      [16, 90, false],
    ]);
    expect(options[2]!.maxBins).toBe(175_352);
    expect(hdDefaultBin(options)).toBe(HD_DEFAULT_BIN_UM);
    expect(hdDefaultBin(options.filter((o) => o.um !== 16))).toBe(8);
    expect(hdBinOptions([f("pbmc/outs/filtered_feature_bc_matrix.h5")])).toEqual([]);
  });

  it("sends one bin size, and says why the rest of the run is left out", () => {
    const items = [
      ...bins("run/outs/binned_outputs/", 2),
      ...bins("run/outs/binned_outputs/", 8),
      f("run/outs/segmented_outputs/cell_segmentations.geojson"),
      f("run/outs/web_summary.html"),
      f("other/filtered_feature_bc_matrix.h5"),
    ];
    const { keep, skipped } = keepHdBin(items, 8);
    expect(keep.some((k) => k.path.includes("square_002um"))).toBe(false);
    expect(keep.some((k) => k.path.includes("segmented"))).toBe(false);
    expect(keep.some((k) => k.path === "other/filtered_feature_bc_matrix.h5")).toBe(true);
    expect(skipped.find((s) => s.path.includes("square_002um"))!.reason).toMatch(/another bin size/);
    expect(skipped.find((s) => s.path.includes("segmented"))!.reason).toMatch(/rest of the Space Ranger/);
  });

  it("refuses what is not read before anything is sent", () => {
    expect(hdRefusal([f("Visium_HD_Mouse_Brain_binned_outputs.tar.gz")])).toMatch(/unpack it/);
    expect(hdRefusal([f("Xenium_FFPE_Rep1_outs.zip")])).toMatch(/not read: unzip it/);
    expect(hdRefusal([f("spots.zip")])).toBeNull();
    expect(hdRefusal(bins("binned_outputs/", 2))).toMatch(/use the square_008um or square_016um/);
    expect(hdRefusal([...bins("", 2), ...bins("", 16)])).toBeNull();
    expect(hdRefusal([f("outs/segmented_outputs/cell_segmentations.geojson")])).toMatch(/segmented_outputs/);
    expect(hdRefusal([...bins("outs/binned_outputs/", 8), f("outs/segmented_outputs/x.geojson")])).toBeNull();
    expect(hdRefusal([f("pbmc/filtered_feature_bc_matrix.h5")])).toBeNull();
  });

  it("sends only what a segmented_outputs folder needs, by the backend's rule", () => {
    const seg = "run/outs/segmented_outputs/";
    const kept = [
      "filtered_feature_cell_matrix.h5",
      "cell_segmentations.geojson",
      "nucleus_segmentations.geojson",
      "graphclust_annotated_cell_segmentations.geojson",
      "spatial/scalefactors_json.json",
      "spatial/tissue_hires_image.png",
      "spatial/tissue_lowres_image.png",
      "filtered_feature_cell_matrix/matrix.mtx.gz",
    ];
    const left = [
      "raw_feature_cell_matrix.h5",
      "raw_feature_cell_matrix/barcodes.tsv.gz",
      "graphclust_annotated_nucleus_segmentations.geojson",
      "analysis/clustering/gene_expression_graphclust/clusters.csv",
      "spatial/cytassist_image.tiff",
      "spatial/aligned_fiducials.jpg",
      "cloupe_cell.cloupe",
    ];
    for (const name of kept) expect(junkReason(seg + name), name).toBeNull();
    for (const name of left) expect(junkReason(seg + name), name).not.toBeNull();
    const py = readFileSync(resolve(__dirname, "../../../backend/src/scview/core/ingestion/paths.py"), "utf8");
    for (const name of ["cell_segmentations.geojson", "nucleus_segmentations.geojson", "graphclust_annotated_cell_segmentations.geojson"]) {
      expect(py).toContain(`"${name}"`);
    }
  });

  it("a run with a cell segmentation reads as cells and says how to get bins", () => {
    const seg = (run: string) => [
      f(`${run}segmented_outputs/filtered_feature_cell_matrix.h5`),
      f(`${run}segmented_outputs/cell_segmentations.geojson`),
      f(`${run}segmented_outputs/spatial/scalefactors_json.json`),
    ];
    const items = [...seg("run/outs/"), ...bins("run/outs/binned_outputs/", 16), f("run/outs/metrics_summary.csv"), f("other/x.h5")];
    expect(hdSegRuns(items)).toEqual(new Set(["run/outs"]));
    expect(hdRefusal(items)).toBeNull();
    const { keep, skipped } = keepHdCells(items);
    expect(keep.map((k) => k.path).sort()).toEqual([...seg("run/outs/").map((k) => k.path), "other/x.h5"].sort());
    expect(skipped.find((s) => s.path.includes("square_016um"))!.reason).toMatch(/reads as cells; for bins upload binned_outputs/);
    expect(skipped.find((s) => s.path.includes("metrics_summary"))!.reason).toMatch(/rest of the Space Ranger/);
    // Its 2 um bins are no refusal when cells are there to read.
    expect(hdRefusal([...seg("outs/"), ...bins("outs/binned_outputs/", 2)])).toBeNull();
    // Both folders dropped together, at the top.
    const top = keepHdCells([...seg(""), ...bins("binned_outputs/", 16)]);
    expect(top.keep.every((k) => k.path.startsWith("segmented_outputs/"))).toBe(true);
    const chosen = keepHdBin([...seg(""), ...bins("binned_outputs/", 16)], 16);
    expect(chosen.keep.every((k) => k.path.includes("square_016um"))).toBe(true);
  });

  it("the folder's matrix folder is sent only when there is no .h5", () => {
    const seg = "segmented_outputs/";
    const mex = [f(`${seg}filtered_feature_cell_matrix/matrix.mtx.gz`), f(`${seg}filtered_feature_cell_matrix/barcodes.tsv.gz`)];
    const outline = f(`${seg}cell_segmentations.geojson`);
    expect(keepHdCells([...mex, outline]).keep).toHaveLength(3);
    const both = keepHdCells([...mex, outline, f(`${seg}filtered_feature_cell_matrix.h5`)]);
    expect(both.keep).toHaveLength(2);
    expect(both.skipped[0]!.reason).toMatch(/same matrix/);
  });

  it("refuses a segmented_outputs archive and a folder without its matrix or outlines", () => {
    expect(hdRefusal([f("Visium_HD_Mouse_Brain_segmented_outputs.tar.gz")])).toMatch(/segmented_outputs archive/);
    expect(hdRefusal([f("outs/segmented_outputs/filtered_feature_cell_matrix.h5")])).toMatch(/lacks its/);
  });

  it("the refusals say the same as the backend's", () => {
    const py = readFileSync(resolve(__dirname, "../../../backend/src/scview/core/spatial.py"), "utf8").replace(/"\n\s+"/g, "");
    expect(py).toContain("is a Visium HD binned_outputs archive, which scView does not read: unpack it and drop the binned_outputs folder (or one square_NNNum folder) instead.");
    expect(py).toContain("is a Visium HD segmented_outputs archive, which scView does not read: unpack it and drop the segmented_outputs folder instead.");
    expect(py).toContain("This Space Ranger segmented_outputs folder lacks its filtered_feature_cell_matrix.h5 (or that folder) or its cell_segmentations.geojson, and a dataset of cells needs both: upload the whole folder.");
    expect(py).toContain("the run's bins: a run with a cell segmentation reads as cells; for bins upload binned_outputs or one square_NNNum folder alone");
    expect(py).toContain("are millions of observations (6.3 million on a mouse brain section), far past the 1.3 million cells scView has been tested on: use the square_008um or square_016um folder instead.");
  });
});
