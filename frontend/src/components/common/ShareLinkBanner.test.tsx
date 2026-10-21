import { StrictMode } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getDataset: vi.fn(), listDatasets: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset, listDatasets: api.listDatasets }));
const restore = vi.hoisted(() => ({ restoreSession: vi.fn() }));
vi.mock("@/lib/sessionRestore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/sessionRestore")>()),
  restoreSession: restore.restoreSession,
}));

import type { DatasetInfo } from "@/api/types";
import { SESSION_FORMAT, SESSION_VERSION, type SessionFile } from "@/lib/session";
import { buildShareUrl } from "@/lib/shareUrl";
import { ShareLinkBanner } from "./ShareLinkBanner";

const kang: DatasetInfo = {
  id: "kang-id",
  name: "Kang 2018",
  filename: "k.h5ad",
  n_cells: 13836,
  n_genes: 5,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  cells_fingerprint: "fp-kang",
  created_at: "",
};

const session = (id = "kang-id"): SessionFile => ({
  format: SESSION_FORMAT,
  version: SESSION_VERSION,
  dataset: { id, name: "Kang 2018", n_cells: 13836, cells_fingerprint: "fp-kang" },
  view: { panel: "unified", overlay: { type: "expression", gene: "ISG15" } },
});

const here = { origin: "http://localhost:3000", pathname: "/", search: "" };
async function goTo(s: SessionFile) {
  const { url } = await buildShareUrl(s, here);
  window.history.replaceState(null, "", url.slice(here.origin.length));
  return url;
}

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/");
  api.getDataset.mockResolvedValue(kang);
  api.listDatasets.mockResolvedValue([kang]);
  restore.restoreSession.mockResolvedValue({ headline: "Opened “Kang 2018”.", notes: ["Camera: not restored, the embedding differs."], level: "warning" });
});
afterEach(() => {
  window.history.replaceState(null, "", "/");
});

describe("ShareLinkBanner", () => {
  it("opens the link once (even in StrictMode), shows what was not restored, then clears the fragment", async () => {
    await goTo(session());
    render(
      <StrictMode>
        <ShareLinkBanner />
      </StrictMode>,
    );
    const card = await screen.findByRole("status", { name: "Share link" });
    await waitFor(() => expect(card).toHaveTextContent("Opened “Kang 2018”."));
    expect(card).toHaveTextContent("Camera: not restored, the embedding differs.");
    expect(restore.restoreSession).toHaveBeenCalledTimes(1);
    expect(restore.restoreSession.mock.calls[0]![0].view.overlay).toEqual({ type: "expression", gene: "ISG15" });
    expect(window.location.hash).toBe("");
    expect(window.location.pathname).toBe("/");

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("status", { name: "Share link" })).not.toBeInTheDocument();
  });

  it("opens a link pasted into the address bar of an open tab", async () => {
    render(<ShareLinkBanner />);
    expect(restore.restoreSession).not.toHaveBeenCalled();
    await act(async () => {
      await goTo(session());
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    await waitFor(() => expect(restore.restoreSession).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(window.location.hash).toBe(""));
  });

  it("says when the library lacks the dataset, and restores nothing", async () => {
    api.getDataset.mockRejectedValue(new Error("API error 404: not found"));
    api.listDatasets.mockResolvedValue([{ ...kang, id: "paul", n_cells: 2730, cells_fingerprint: "fp-paul" }]);
    await goTo(session("someone-elses-id"));
    render(<ShareLinkBanner />);
    expect(await screen.findByRole("status", { name: "Share link" })).toHaveTextContent(
      "This link is for dataset “Kang 2018” (13,836 cells), which is not in this scView's library.",
    );
    expect(restore.restoreSession).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("");
  });

  it("ignores a damaged link with a notice instead of crashing", async () => {
    window.history.replaceState(null, "", "/#view=v1.z.not*base64");
    render(<ShareLinkBanner />);
    expect(await screen.findByRole("alert", { name: "Share link" })).toHaveTextContent(/link is damaged/);
    expect(restore.restoreSession).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("");
  });

  it("keeps the fragment after a server error, so a reload tries again", async () => {
    api.getDataset.mockRejectedValue(new Error("API error 503: busy"));
    await goTo(session());
    render(<ShareLinkBanner />);
    expect(await screen.findByRole("alert", { name: "Share link" })).toHaveTextContent(/Reload the page to try again/);
    expect(window.location.hash).toMatch(/^#view=v1\./);
  });

  it("does nothing without a link", () => {
    window.history.replaceState(null, "", "/#somewhere");
    const { container } = render(<ShareLinkBanner />);
    expect(container).toBeEmptyDOMElement();
    expect(api.getDataset).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("#somewhere");
  });
});
