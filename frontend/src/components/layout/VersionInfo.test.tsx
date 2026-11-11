import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { getSystemVersion } = vi.hoisted(() => ({ getSystemVersion: vi.fn() }));
vi.mock("@/api/system", () => ({ getSystemVersion }));

import { VersionInfo, zipUrl } from "./VersionInfo";

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

  it("opens update steps with the release's ZIP, also from the collapsed sidebar", async () => {
    const v = { ...BASE, latest: "0.70.0", update_available: true,
      release_url: "https://github.com/thirtysix/scView/releases/tag/v0.70.0" };
    getSystemVersion.mockResolvedValue(v);
    const { unmount } = render(<VersionInfo collapsed={false} />);
    const button = await screen.findByRole("button", { name: /scView 0.70.0 is available/ });
    expect(button).toHaveTextContent("0.70.0 available");
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(button);
    const dialog = screen.getByRole("dialog", { name: "Update to scView 0.70.0" });
    expect(screen.getByTestId("update-zip")).toHaveAttribute(
      "href", "https://github.com/thirtysix/scView/archive/refs/tags/v0.70.0.zip");
    expect(screen.getByRole("link", { name: "What's new" })).toHaveAttribute("href", v.release_url);
    expect(dialog).toHaveTextContent("git pull");
    expect(dialog).toHaveTextContent("start-windows.bat");
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    unmount();
    render(<VersionInfo collapsed />);
    fireEvent.click(await screen.findByRole("button", { name: /scView 0.70.0 is available/ }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByText("scView 0.69.1")).toBeNull();
  });

  it("builds the ZIP link only from a GitHub release URL", () => {
    expect(zipUrl("https://github.com/a/b/releases/tag/v1.2.3")).toBe("https://github.com/a/b/archive/refs/tags/v1.2.3.zip");
    expect(zipUrl("https://example.com/releases/tag/v1.2.3")).toBeNull();
  });

});
