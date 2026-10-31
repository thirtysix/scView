import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const getAmbient = vi.fn();
vi.mock("@/api/ambient", () => ({ getAmbient: (...a: unknown[]) => getAmbient(...a) }));

import { useAmbientFlags } from "./AmbientFlag";

describe("useAmbientFlags", () => {
  beforeEach(() => {
    getAmbient.mockReset();
    getAmbient.mockResolvedValue({
      samples: [],
      markers: { n_flagged: 1, approximate: false, groups: { "0": [{ gene: "LYZ", soup_rank: 1, ratio: 2.4 }] } },
    });
  });

  it("asks the backend once markers are there", async () => {
    const { result } = renderHook(() => useAmbientFlags("d1", "leiden", true));
    await waitFor(() => expect(getAmbient).toHaveBeenCalledWith("d1", "leiden"));
    await waitFor(() => expect(result.current.flags.get("0")?.has("LYZ")).toBe(true));
  });

  it("never asks on a large dataset (the backend would answer 422)", async () => {
    const { result } = renderHook(() => useAmbientFlags("d1", "leiden", true, { mode: "large" }));
    await new Promise((r) => setTimeout(r, 20));
    expect(getAmbient).not.toHaveBeenCalled();
    expect(result.current.flags.size).toBe(0);
  });
});
