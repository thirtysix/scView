/** Opening a dataset sends nothing from it to the language model: the banner asks only for the
 *  deterministic insight (the backend route calls no model), and its question goes to the
 *  co-pilot only on a click. */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { InsightBanner } from "./InsightBanner";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

const fetchMock = vi.fn();

beforeEach(() => {
  localStorage.clear();
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(
    new Response(
      JSON.stringify({
        insight: "Your cells split by stim (CTRL, STIM).",
        question: "Compare CTRL and STIM",
        severity: "suggestion",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  useViewStore.setState({ pendingAsk: null });
  useDatasetStore.setState({ currentDatasetId: "d1" });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  useDatasetStore.setState({ currentDatasetId: null });
});

describe("InsightBanner", () => {
  it("asks only for the deterministic insight on open, never a reworded one", async () => {
    render(<InsightBanner />);
    expect(await screen.findByText("Your cells split by stim (CTRL, STIM).")).toBeTruthy();
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls).toHaveLength(1);
    expect(urls[0]).toMatch(/\/datasets\/d1\/assistant\/insight$/);
    expect(urls[0]).not.toMatch(/polish/);
    expect(fetchMock.mock.calls[0]?.[1]?.method ?? "GET").toBe("GET");
    expect(useViewStore.getState().pendingAsk).toBeNull();
  });

  it("hands the question to the co-pilot only on a click", async () => {
    render(<InsightBanner />);
    const chip = await screen.findByRole("button", { name: /Compare CTRL and STIM/ });
    expect(useViewStore.getState().pendingAsk).toBeNull();
    fireEvent.click(chip);
    await waitFor(() => expect(useViewStore.getState().pendingAsk).toBe("Compare CTRL and STIM"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
