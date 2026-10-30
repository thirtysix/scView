import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { itemsFromDrop, itemsFromFiles, junkReason, relativePath, skippedNote, splitJunk } from "./ingestUpload";

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
  ];
  const heavy = [
    "run/outs/transcripts.parquet",
    "run/outs/transcripts.csv.gz",
    "run/outs/transcripts.zarr.zip",
    "run/outs/morphology.ome.tif",
    "run/outs/morphology_mip.ome.tif",
    "run/outs/cell_boundaries.parquet",
    "run/outs/nucleus_boundaries.csv.gz",
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
