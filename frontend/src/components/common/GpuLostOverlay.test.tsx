import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const probe = vi.hoisted(() => ({ ok: true }));
vi.mock("@/lib/webgl", () => ({ canCreateWebGL: () => probe.ok }));

import { GpuLostOverlay } from "./GpuLostOverlay";

describe("GpuLostOverlay", () => {
  beforeEach(() => {
    probe.ok = true;
  });

  it("remounts the plot when the browser still gives out contexts", () => {
    const onRetry = vi.fn();
    render(<GpuLostOverlay onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("asks for a reload instead of retrying when WebGL is blocked for the page", () => {
    probe.ok = false;
    const onRetry = vi.fn();
    render(<GpuLostOverlay onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Reload page" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("goes straight to reload when the last mount could not create a context", () => {
    render(<GpuLostOverlay onRetry={() => {}} blocked />);
    expect(screen.getByRole("button", { name: "Reload page" })).toBeInTheDocument();
  });
});
