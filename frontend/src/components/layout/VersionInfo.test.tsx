import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

const { getSystemVersion } = vi.hoisted(() => ({ getSystemVersion: vi.fn() }));
vi.mock("@/api/system", () => ({ getSystemVersion }));

import { VersionInfo } from "./VersionInfo";

const BASE = { version: "0.69.1", check: "on" as const, latest: "0.69.1", update_available: false, release_url: null };

// Braces: a function returned from beforeEach is run as its teardown (mockReset returns the mock).
beforeEach(() => {
  getSystemVersion.mockReset();
});

describe("VersionInfo", () => {
  it("shows nothing when the backend cannot say", async () => {
    getSystemVersion.mockRejectedValue(new Error("404"));
    const { container } = render(<VersionInfo collapsed={false} />);
    await vi.waitFor(() => expect(getSystemVersion).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    expect(container).toBeEmptyDOMElement();
  });
  it("shows the version, and no link when this is the latest", async () => {
    getSystemVersion.mockResolvedValue(BASE);
    render(<VersionInfo collapsed={false} />);
    expect(await screen.findByText("scView 0.69.1")).toBeInTheDocument();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("links a newer release, also in the collapsed sidebar", async () => {
    const v = { ...BASE, latest: "0.70.0", update_available: true,
      release_url: "https://github.com/thirtysix/scView/releases/tag/v0.70.0" };
    getSystemVersion.mockResolvedValue(v);
    const { unmount } = render(<VersionInfo collapsed={false} />);
    const link = await screen.findByRole("link", { name: "scView 0.70.0 is available" });
    expect(link).toHaveAttribute("href", v.release_url);
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveTextContent("0.70.0 available");
    expect(link.getAttribute("title")).toContain("git pull");
    unmount();
    render(<VersionInfo collapsed />);
    expect(await screen.findByRole("link", { name: "scView 0.70.0 is available" })).toBeInTheDocument();
    expect(screen.queryByText("scView 0.69.1")).toBeNull();
  });

});
