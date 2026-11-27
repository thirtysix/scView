import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { StoragePlan, StoreByRowsProgress } from "@/api/storage";

const api = vi.hoisted(() => ({ getStoragePlan: vi.fn(), storeByRows: vi.fn() }));
vi.mock("@/api/storage", async () => ({ ...(await vi.importActual<object>("@/api/storage")), ...api }));

import { StoreByRowsOffer } from "./StoreByRowsOffer";
import { hoverTip } from "@/test/hoverTip";

const GB = 1024 ** 3;
const PLAN: StoragePlan = {
  dataset_id: "d1", column_stored: ["X", "raw/X"], blocks_large_mode: ["X", "raw/X"], files: 1, seconds: 32,
  disk_bytes: 1.7 * GB, disk_free_bytes: 508 * GB, enough_disk: true, memory_bytes: 0.4 * GB,
  text: "about 30 s, 1.7 GB of disk (508 GB free)",
};

beforeEach(() => vi.clearAllMocks());

describe("StoreByRowsOffer", () => {
  it("offers one action with its time and disk, and a tip on the button itself", async () => {
    render(<StoreByRowsOffer datasetId="d1" plan={PLAN} />);
    expect(screen.getByTestId("store-by-rows-offer")).toHaveTextContent("This dataset is stored by columns.");
    const button = screen.getByRole("button", { name: "Store this dataset by rows (about 30 s, 1.7 GB of disk)" });
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(api.getStoragePlan).not.toHaveBeenCalled(); // the preflight gave the plan
    expect(api.storeByRows).not.toHaveBeenCalled(); // never without the click
    const tip = await hoverTip(button);
    expect(tip).toContain("No value, label or result changes");
    expect(tip).toContain("your uploaded file is not touched");
  });

  it("on a click: progress, then done and the caller's retry", async () => {
    let report!: (p: StoreByRowsProgress) => void;
    let finish!: () => void;
    api.storeByRows.mockImplementation((_id: string, onProgress: (p: StoreByRowsProgress) => void) => {
      report = onProgress;
      return new Promise((resolve) => {
        finish = () => resolve({ converted: ["X", "raw/X"], seconds: 31, bytes_before: 1, bytes_after: 2 });
      });
    });
    const onDone = vi.fn();
    render(<StoreByRowsOffer datasetId="d1" plan={PLAN} doneText="Stored by rows. Running the steps again." onDone={onDone} />);
    fireEvent.click(screen.getByRole("button"));
    expect(api.storeByRows).toHaveBeenCalledWith("d1", expect.any(Function));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
    expect(screen.queryByRole("button")).toBeNull(); // not started twice
    act(() => report({ done: 45, total: 100 }));
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "45");
    expect(screen.getByText(/Storing by rows: 45%\. The dataset stays as it is until the new file is whole\./)).toBeTruthy();
    expect(onDone).not.toHaveBeenCalled();
    await act(async () => finish());
    expect(await screen.findByText("Stored by rows. Running the steps again.")).toBeTruthy();
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("a short disk: the button is off and both numbers are said", () => {
    render(<StoreByRowsOffer datasetId="d1" plan={{ ...PLAN, enough_disk: false, disk_free_bytes: 0.7 * GB }} />);
    expect(screen.getByRole("button")).toBeDisabled();
    expect(screen.getByText(/needs about 1\.7 GB of disk and 0\.7 GB is free\. Free some disk space first\./)).toBeTruthy();
  });

  it("a failure is said and the action offered again; nothing is retried by itself", async () => {
    api.storeByRows.mockRejectedValue(new Error("A run is working on this dataset: store it by rows when the run has finished."));
    const onDone = vi.fn();
    render(<StoreByRowsOffer datasetId="d1" plan={PLAN} onDone={onDone} />);
    fireEvent.click(screen.getByRole("button"));
    expect(await screen.findByRole("alert")).toHaveTextContent("A run is working on this dataset");
    expect(screen.getByRole("button")).toBeEnabled();
    expect(onDone).not.toHaveBeenCalled();
    expect(api.storeByRows).toHaveBeenCalledTimes(1);
  });

  it("without a plan (a refused run) it asks for the cost", async () => {
    api.getStoragePlan.mockResolvedValue(PLAN);
    render(<StoreByRowsOffer datasetId="d1" />);
    expect(screen.getByRole("button", { name: "Store this dataset by rows" })).toBeTruthy();
    await waitFor(() => expect(screen.getByRole("button", { name: /about 30 s, 1\.7 GB of disk/ })).toBeTruthy());
    expect(api.getStoragePlan).toHaveBeenCalledWith("d1", expect.anything());
  });
});
