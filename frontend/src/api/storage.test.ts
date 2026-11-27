import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { COLUMN_STORED_PHRASE, columnStoredText, isColumnStoredRefusal, isConversionRefusal, runErrorText } from "./pipelineRun";
import { shortDiskText, storeByRowsLabel } from "./storage";

const GB = 1024 ** 3;
const BACKEND = resolve(__dirname, "../../../backend/src/scview");

describe("store by rows: the offer's words", () => {
  it("names the time and the disk", () => {
    expect(storeByRowsLabel({ seconds: 32, disk_bytes: 1.74 * GB })).toBe(
      "Store this dataset by rows (about 30 s, 1.7 GB of disk)",
    );
    expect(storeByRowsLabel({ seconds: 2, disk_bytes: 0.05 * GB })).toBe(
      "Store this dataset by rows (about 5 s, 0.05 GB of disk)",
    );
    expect(storeByRowsLabel({ seconds: 400, disk_bytes: 42 * GB })).toBe(
      "Store this dataset by rows (about 7 min, 42 GB of disk)",
    );
  });

  it("says a short disk with both numbers, and nothing when it fits", () => {
    expect(shortDiskText({ enough_disk: true, disk_bytes: GB, disk_free_bytes: 9 * GB })).toBeNull();
    expect(shortDiskText({ enough_disk: false, disk_bytes: 1.7 * GB, disk_free_bytes: 0.7 * GB })).toBe(
      "It writes a second copy: that needs about 1.7 GB of disk and 0.7 GB is free. Free some disk space first.",
    );
    expect(
      shortDiskText({ enough_disk: false, disk_bytes: 0.2 * GB, disk_free_bytes: 0.4 * GB, disk_floor_bytes: 0.25 * GB }),
    ).toBe(
      "It writes a second copy: that needs about 0.2 GB of disk and 0.4 GB is free (scView leaves 0.25 GB free). Free some disk space first.",
    );
  });
});

describe("the run refusal for a dataset stored by columns", () => {
  const REFUSAL =
    "In memory the run needs about 2.2 GB more at its peak; scView has 2.5 GB free, so it runs in large-dataset " +
    "mode: the matrices stay on disk, read in blocks. That mode reads the matrix in blocks of cells, and this " +
    "dataset's is stored by columns (X, raw/X). Nothing was run: store the dataset by rows first (about 30 s and " +
    "1.7 GB of disk; no value changes), then run it again.";

  it("is told from the conversion refusal and from other errors", () => {
    expect(isColumnStoredRefusal(REFUSAL)).toBe(true);
    expect(isConversionRefusal(REFUSAL)).toBe(false);
    expect(isColumnStoredRefusal("Nothing was run: run it again allowing large-dataset mode to convert it")).toBe(false);
    expect(isColumnStoredRefusal("Unknown pipeline step")).toBe(false);
  });

  it("uses the backend's own phrase", () => {
    const py = readFileSync(resolve(BACKEND, "core/store_by_rows.py"), "utf8");
    expect(py).toContain(`REFUSAL_PHRASE = "${COLUMN_STORED_PHRASE}"`);
  });

  it("where the offer cannot be shown, says where it is, in plain words", () => {
    expect(runErrorText(new Error(REFUSAL))).toBe(columnStoredText());
    expect(columnStoredText()).toContain("Open Data Assessment");
    expect(columnStoredText()).not.toContain("/api/");
    expect(columnStoredText()).not.toContain("store_by_rows");
  });
});
