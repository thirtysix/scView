import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { getSystemMemory } = vi.hoisted(() => ({ getSystemMemory: vi.fn() }));
vi.mock("@/api/system", () => ({ getSystemMemory }));

import { MemoryIndicator, memoryShare, restartNotice } from "./MemoryIndicator";

const GB = 1024 ** 3;

describe("MemoryIndicator", () => {
  beforeEach(() => {
    getSystemMemory.mockReset();
    sessionStorage.clear();
  });

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

  it("says the backend was killed and restarted, until dismissed", async () => {
    const restarted = { restarted_at: "2026-10-20T13:00:00+00:00", previous_start: "2026-10-20T12:55:00+00:00" };
    getSystemMemory.mockResolvedValue({
      limit_bytes: 3 * GB, used_bytes: 0.8 * GB, available_bytes: 2.2 * GB, limited: true,
      source: "container limit", unclean_restart: restarted,
    });
    const { unmount } = render(<MemoryIndicator collapsed={false} />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/ran out of memory: it may use 3.0 GB/);
    expect(alert.textContent).toMatch(/SCVIEW_MEMORY_GB/);
    fireEvent.click(screen.getByLabelText("Dismiss"));
    expect(screen.queryByRole("alert")).toBeNull();
    unmount();
    render(<MemoryIndicator collapsed={false} />);  // same restart: stays dismissed
    await screen.findByLabelText(/^Memory /);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("words the notice for a machine without a limit", () => {
    const m = {
      limit_bytes: 64 * GB, used_bytes: GB, available_bytes: 63 * GB, limited: false, source: "machine" as const,
      unclean_restart: { restarted_at: "2026-10-20T13:00:00+00:00", previous_start: null },
    };
    expect(restartNotice(m)).toMatch(/possibly because the computer ran out of memory/);
    expect(restartNotice({ ...m, unclean_restart: null })).toBeNull();
  });
});
