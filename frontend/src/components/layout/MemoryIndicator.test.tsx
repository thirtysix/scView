import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

const { getSystemMemory } = vi.hoisted(() => ({ getSystemMemory: vi.fn() }));
vi.mock("@/api/system", () => ({ getSystemMemory }));

import { MemoryIndicator, memoryShare } from "./MemoryIndicator";

const GB = 1024 ** 3;

describe("MemoryIndicator", () => {
  beforeEach(() => getSystemMemory.mockReset());

  it("shows use against the container limit and says where it is set", async () => {
    getSystemMemory.mockResolvedValue({
      limit_bytes: 4 * GB, used_bytes: 1.2 * GB, available_bytes: 2.8 * GB, limited: true, source: "container limit",
    });
    render(<MemoryIndicator collapsed={false} />);
    const el = await screen.findByLabelText("Memory 1.2 / 4.0 GB");
    expect(el.getAttribute("title")).toMatch(/SCVIEW_MEMORY_GB/);
    expect(screen.queryByText(/no limit/)).toBeNull();
  });

  it("marks a machine without a limit", async () => {
    getSystemMemory.mockResolvedValue({
      limit_bytes: 64 * GB, used_bytes: 0.5 * GB, available_bytes: 63.5 * GB, limited: false, source: "machine",
    });
    render(<MemoryIndicator collapsed={false} />);
    expect(await screen.findByText("Memory (no limit)")).toBeTruthy();
  });

  it("shows nothing when the backend can't say", () => {
    expect(memoryShare({ limit_bytes: null, used_bytes: 1, available_bytes: null, limited: false, source: "unknown" })).toBeNull();
    expect(memoryShare(null)).toBeNull();
  });
});
