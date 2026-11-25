import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { UploadTask } from "@/hooks/useUploadTask";
import { SHOW_AFTER_MS, UploadProgressCompact, UploadTaskProgress } from "./UploadProgressCompact";

const MB = 1e6;
const one = (loaded: number, total = 900 * MB) => ({ loaded, total, file: "sample.loom", fileIndex: 1, fileCount: 1 });

describe("UploadProgressCompact", () => {
  afterEach(() => vi.useRealTimers());

  it("shows nothing for the first half second, so a quick upload flashes nothing", () => {
    vi.useFakeTimers();
    render(<UploadProgressCompact name="genes.gmt" progress={one(1, 2)} rate={null} left={null} />);
    expect(screen.queryByTestId("upload-compact")).toBeNull();
    act(() => vi.advanceTimersByTime(SHOW_AFTER_MS));
    expect(screen.getByTestId("upload-compact")).toBeInTheDocument();
  });

  it("bar, percent and MB at first; speed and time left once known; Cancel", () => {
    const onCancel = vi.fn();
    const { rerender } = render(
      <UploadProgressCompact name="sample.loom" progress={one(90 * MB)} rate={null} left={null} onCancel={onCancel} delayMs={0} />,
    );
    const bar = screen.getByRole("progressbar", { name: "Upload of sample.loom" });
    expect(bar).toHaveAttribute("aria-valuenow", "10");
    expect(bar).toHaveAttribute("aria-valuetext", "10%, 90 of 900 MB");
    expect(screen.getByTestId("upload-compact-numbers")).toHaveTextContent(/^10%90 of 900 MB$/);
    expect(screen.getByRole("status")).toHaveTextContent("Uploading sample.loom.");
    expect(screen.getByRole("status")).toHaveClass("sr-only");

    rerender(
      <UploadProgressCompact name="sample.loom" progress={one(450 * MB)} rate={3 * MB} left={150} onCancel={onCancel} delayMs={0} />,
    );
    expect(screen.getByTestId("upload-compact-numbers")).toHaveTextContent("50%450 of 900 MB3.0 MB/sabout 3 min left");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("once every byte is sent: no Cancel, no speed, and the next step in the live region", () => {
    render(
      <UploadProgressCompact name="sample.loom" progress={one(900 * MB)} rate={3 * MB} left={0} onCancel={() => {}}
        sentText="Sent. Matching cells and genes…" delayMs={0} />,
    );
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(screen.getByTestId("upload-compact-numbers")).toHaveTextContent(/^100%900 of 900 MB$/);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Sent. Matching cells and genes…");
    expect(status).not.toHaveClass("sr-only");
    expect(status).toHaveAttribute("aria-live", "polite");
  });

  it("names the file going now in a set of several", () => {
    render(
      <UploadProgressCompact name="2 files" delayMs={0} rate={null} left={null}
        progress={{ loaded: 10 * MB, total: 300 * MB, file: "atac_fragments.tsv.gz", fileIndex: 2, fileCount: 2 }} />,
    );
    expect(screen.getByTestId("upload-compact-file")).toHaveTextContent("File 2 of 2: atac_fragments.tsv.gz");
  });

  it("minimal: the bar and percent only, without Cancel or speed", () => {
    render(
      <UploadProgressCompact name="genes.gmt" progress={one(5 * MB, 10 * MB)} rate={2 * MB} left={3} onCancel={() => {}}
        minimal delayMs={0} />,
    );
    expect(screen.getByTestId("upload-compact")).toHaveTextContent("Uploading genes.gmt: 50%");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText(/MB\/s/)).toBeNull();
  });
});

const task = (over: Partial<UploadTask>): UploadTask => ({
  name: null,
  cancelled: null,
  progress: null,
  rate: null,
  left: null,
  run: vi.fn(),
  cancel: vi.fn(),
  ...over,
});

describe("UploadTaskProgress", () => {
  it("shows the running upload with the task's Cancel", () => {
    const t = task({ name: "contigs.csv", progress: one(1 * MB, 4 * MB) });
    render(<UploadTaskProgress task={t} delayMs={0} />);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(t.cancel).toHaveBeenCalled();
  });

  it("says a cancelled upload kept nothing, or only that it was cancelled for a set sent in turn", () => {
    const { rerender } = render(<UploadTaskProgress task={task({ cancelled: "contigs.csv" })} />);
    expect(screen.getByTestId("upload-cancelled")).toHaveTextContent("Upload of contigs.csv cancelled. Nothing was kept.");
    rerender(<UploadTaskProgress task={task({ cancelled: "2 files" })} keptNothing={false} />);
    expect(screen.getByTestId("upload-cancelled")).toHaveTextContent(/^Upload of 2 files cancelled\.$/);
    rerender(<UploadTaskProgress task={task({})} />);
    expect(screen.queryByTestId("upload-cancelled")).toBeNull();
  });
});
