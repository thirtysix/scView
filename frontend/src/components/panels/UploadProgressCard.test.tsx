import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

vi.mock("@/components/common/PreflightNote", () => ({
  PreflightNote: ({ verdict }: { verdict: { sentence: string } }) => <p data-testid="preflight">{verdict.sentence}</p>,
}));

import { UploadProgressCard } from "./UploadProgressCard";
import type { PreflightVerdict } from "@/api/preflight";

const MB = 1e6;
const one = (loaded: number, total = 331 * MB) => ({ loaded, total, file: "sample.rds", fileIndex: 1, fileCount: 1 });

describe("UploadProgressCard", () => {
  it("shows percent, MB sent, speed and time left on an accessible bar, with Cancel", () => {
    const onCancel = vi.fn();
    render(
      <UploadProgressCard names="sample.rds" phase="uploading" progress={one(152 * MB)} rate={3.2 * MB} left={56}
        rObject onCancel={onCancel} />,
    );
    const bar = screen.getByRole("progressbar", { name: "Upload of sample.rds" });
    expect(bar).toHaveAttribute("aria-valuenow", "45");
    expect(bar).toHaveAttribute("aria-valuetext", "45%, 152 of 331 MB");
    expect(screen.getByTestId("upload-numbers")).toHaveTextContent("45%152 of 331 MB3.2 MB/sabout 1 min left");
    expect(screen.getByTestId("upload-title")).toHaveTextContent("Uploading sample.rds");
    expect(screen.getByRole("status")).toHaveTextContent("Uploading. Keep this page open until it is sent.");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("says it is measuring the speed before it has one", () => {
    render(<UploadProgressCard names="sample.rds" phase="uploading" progress={one(1 * MB)} rate={null} left={null} />);
    expect(screen.getByTestId("upload-numbers")).toHaveTextContent("0%1 of 331 MBmeasuring speed…");
  });

  it("names the file going now in a set of many", () => {
    render(
      <UploadProgressCard names="12 files" phase="uploading" rate={null} left={null}
        progress={{ loaded: 40 * MB, total: 120 * MB, file: "barcodes.tsv.gz", fileIndex: 3, fileCount: 12 }} />,
    );
    expect(screen.getByTestId("upload-file")).toHaveTextContent("File 3 of 12: barcodes.tsv.gz");
  });

  it("once every byte is sent: no Cancel, and the next stage in the live region", () => {
    const { rerender } = render(
      <UploadProgressCard names="3 files" phase="uploading" progress={{ ...one(9 * MB, 9 * MB), fileCount: 3 }}
        rate={2 * MB} left={0} onCancel={() => {}} />,
    );
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByRole("status")).toHaveTextContent("Sent. Checking the files…");
    expect(screen.queryByTestId("upload-file")).toBeNull();
    rerender(
      <UploadProgressCard names="sample.rds" phase="uploading" progress={one(331 * MB)} rate={null} left={null} rObject />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Sent. Saving the file and starting the R converter…");
  });

  it("moves on to Converting with the converter's stage and memory estimate", () => {
    const verdict = { level: "fits", sentence: "Converting sample.rds takes about 2.1 GB." } as PreflightVerdict;
    render(
      <UploadProgressCard names="sample.rds" phase="converting" progress={one(331 * MB)} rate={null} left={null} rObject
        stage="Converting a Seurat object (5,000 cells × 20,000 features): writing the counts assay…" preflight={verdict}
        onCancel={() => {}} />,
    );
    expect(screen.getByTestId("upload-title")).toHaveTextContent("Converting sample.rds");
    expect(screen.getByText("Upload (331 MB)")).toBeTruthy();
    expect(screen.getByRole("status")).toHaveTextContent("writing the counts assay…");
    expect(screen.getByTestId("preflight")).toHaveTextContent("about 2.1 GB");
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });

  it("takes no drop: it says one upload at a time", () => {
    render(<UploadProgressCard names="sample.rds" phase="uploading" progress={one(1)} rate={null} left={null} />);
    const card = screen.getByTestId("upload-card");
    expect(fireEvent.drop(card)).toBe(false); // default prevented: the browser does not open the file
    expect(screen.getByRole("alert")).toHaveTextContent("One upload at a time");
  });
});
