import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DiscoverDetail, DiscoverSearch, PublicSource } from "@/api/publicImport";

const api = vi.hoisted(() => ({
  searchDiscover: vi.fn(),
  getDiscoverDataset: vi.fn(),
  importDiscoverDataset: vi.fn(),
  cancelPublicImport: vi.fn(),
  getPublicImportStatus: vi.fn(),
}));
vi.mock("@/api/publicImport", async (orig) => ({
  ...(await orig<typeof import("@/api/publicImport")>()),
  ...api,
}));
const ds = vi.hoisted(() => ({ getDataset: vi.fn(), listDatasets: vi.fn(), uploadDataset: vi.fn() }));
vi.mock("@/api/datasets", () => ds);
vi.mock("@/components/panels/DatasetLibrary", () => ({ DatasetLibrary: () => null }));
vi.mock("@/components/common/PreflightNote", () => ({ PreflightIngestNotes: () => null }));

import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { DataPanel } from "./DataPanel";
import { PublicImport } from "./PublicImport";
import { PublicSourceCard } from "./PublicSourceCard";

const VID = "0d5aa69f-1ad1-470f-9a1b-4a167da3b545";
const REC = {
  dataset_id: "0895c838-e550-48a3-a777-dbcd35d30272",
  dataset_version_id: VID,
  collection_id: "44531dd9-1388-4416-a117-af0a99de2294",
  collection_name: "Healthy human liver",
  collection_doi: "10.1002/hep4.1854",
  collection_doi_label: "Andrews et al. (2022) Hepatology Communications",
  title: "Healthy human liver: B cells",
  citation: "Publication: https://doi.org/10.1002/hep4.1854 Dataset Version: ...",
  organism: ["Homo sapiens"],
  tissue: ["liver", "caudate lobe of liver", "left lobe of liver"],
  disease: ["normal"],
  assay: ["10x 3' v3"],
  n_cell_types: 2,
  suspension_type: ["cell"],
  cell_count: 146,
  size_bytes: 5382428,
  schema_version: "7.1.0",
  published_at: "2026-06-10",
  explorer_url: null,
  license: "CC BY 4.0",
};
const SEARCH: DiscoverSearch = {
  total: 1,
  results: [REC],
  facets: { organism: [["Homo sapiens", 3], ["Mus musculus", 1]], tissue: [["liver", 1]] },
  fetched_at: "2026-10-24T12:00:00+00:00",
  n_datasets: 4,
};
const DETAIL: DiscoverDetail = {
  ...REC,
  download_url: `https://datasets.cellxgene.cziscience.com/${VID}.h5ad`,
  license_url: "https://creativecommons.org/licenses/by/4.0/",
  disk: { free_bytes: 50 * 1024 ** 3, need_bytes: 1.1 * 1024 ** 3, ok: true },
};

describe("PublicImport", () => {
  beforeEach(() => {
    Object.values(api).forEach((f) => f.mockReset());
    api.searchDiscover.mockResolvedValue(SEARCH);
    api.getDiscoverDataset.mockResolvedValue(DETAIL);
    ds.getDataset.mockReset().mockResolvedValue({ id: "new", name: REC.title, status: "ready" });
    ds.listDatasets.mockReset().mockResolvedValue([]);
  });

  it("searches, shows the table and filters, and the details with citation and download address", async () => {
    render(<PublicImport onClose={() => {}} />);
    expect(await screen.findByText("Healthy human liver: B cells")).toBeInTheDocument();
    expect(screen.getByText("146")).toBeInTheDocument();
    expect(screen.getByText("5 MB")).toBeInTheDocument();
    expect(screen.getByText(/1 of 4 matching datasets/)).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Mus musculus (1)" })).toBeInTheDocument();

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Organism"), { target: { value: "Mus musculus" } });
    });
    expect(api.searchDiscover).toHaveBeenLastCalledWith(expect.objectContaining({ organism: "Mus musculus" }));

    fireEvent.change(screen.getByLabelText("Search public datasets"), { target: { value: "liver" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Search/ }));
    });
    expect(api.searchDiscover).toHaveBeenLastCalledWith(expect.objectContaining({ q: "liver" }));

    await act(async () => {
      fireEvent.click(screen.getByText("Healthy human liver: B cells"));
    });
    const detail = await screen.findByTestId("public-detail");
    expect(detail).toHaveTextContent(DETAIL.download_url);
    expect(screen.getByTestId("public-citation")).toHaveTextContent("doi.org/10.1002/hep4.1854");
    expect(detail).toHaveTextContent(VID);
  });

  it("imports with progress, then opens the dataset on Data Assessment", async () => {
    let finish: (id: string) => void = () => {};
    api.importDiscoverDataset.mockImplementation(
      (_v: string, onEvent: (e: { event: string; data: Record<string, unknown> }) => void) => {
        onEvent({ event: "job", data: { job_id: "j1" } });
        onEvent({ event: "stage", data: { stage: "download", message: "Downloading the file" } });
        onEvent({ event: "progress", data: { done: 1024 ** 2, total: 4 * 1024 ** 2 } });
        return new Promise<string>((r) => {
          finish = r;
        });
      },
    );
    api.cancelPublicImport.mockResolvedValue({ cancelled: true });
    render(<PublicImport onClose={() => {}} />);
    const row = await screen.findByText("Healthy human liver: B cells");
    await act(async () => {
      fireEvent.click(row);
    });
    const go = await screen.findByRole("button", { name: /^Import$/ });
    await act(async () => {
      fireEvent.click(go);
    });
    const progress = screen.getByTestId("public-progress");
    expect(progress).toHaveTextContent("Downloading the file");
    expect(progress).toHaveTextContent("1 MB of 4 MB");
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    });
    expect(api.cancelPublicImport).toHaveBeenCalledWith("j1");
    await act(async () => {
      finish("new");
    });
    await waitFor(() => expect(useDatasetStore.getState().currentDataset?.id).toBe("new"));
    expect(useViewStore.getState().activePanel).toBe("assessment");
  });

  it("shows the server's reason when the import fails", async () => {
    api.importDiscoverDataset.mockRejectedValue(new Error("Not enough disk: the import needs about 3 GB"));
    render(<PublicImport onClose={() => {}} />);
    const row = await screen.findByText("Healthy human liver: B cells");
    await act(async () => {
      fireEvent.click(row);
    });
    const go = await screen.findByRole("button", { name: /^Import$/ });
    await act(async () => {
      fireEvent.click(go);
    });
    expect(await screen.findByText(/Not enough disk/)).toBeInTheDocument();
  });
});

describe("DataPanel: the public import entry", () => {
  beforeEach(() => {
    api.searchDiscover.mockReset().mockResolvedValue(SEARCH);
  });

  it("is offered only when the server allows it", async () => {
    api.getPublicImportStatus.mockReset().mockResolvedValue({ enabled: true });
    const { unmount } = render(<DataPanel />);
    const button = await screen.findByRole("button", { name: /Import public dataset/ });
    await act(async () => {
      fireEvent.click(button);
    });
    expect(screen.getByTestId("public-import")).toBeInTheDocument();
    unmount();

    api.getPublicImportStatus.mockReset().mockResolvedValue({ enabled: false });
    render(<DataPanel />);
    await waitFor(() => expect(api.getPublicImportStatus).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /Import public dataset/ })).toBeNull();
  });
});

describe("PublicSourceCard", () => {
  it("shows the citation, version, licence and hash", () => {
    const source: PublicSource = {
      about_title: REC.title,
      about_source: "CELLxGENE Discover",
      about_citation: REC.citation,
      about_publication: REC.collection_doi_label,
      about_doi: REC.collection_doi,
      about_collection: REC.collection_name,
      about_collection_id: REC.collection_id,
      about_dataset_id: REC.dataset_id,
      about_dataset_version_id: VID,
      about_schema_version: "7.1.0",
      about_license: "CC BY 4.0",
      about_url: DETAIL.download_url,
      about_downloaded_at: "2026-10-24T12:00:00+00:00",
      about_sha256: "ab".repeat(32),
      about_size_bytes: "5382428",
      citation_text: "Andrews et al. (2022) Hepatology Communications doi:10.1002/hep4.1854. Licence CC BY 4.0.",
    };
    render(<PublicSourceCard source={source} />);
    expect(screen.getByTestId("public-source-citation")).toHaveTextContent("doi:10.1002/hep4.1854");
    const card = screen.getByTestId("public-source");
    expect(card).toHaveTextContent(VID);
    expect(card).toHaveTextContent("ab".repeat(32));
    expect(card).toHaveTextContent("2026-10-24");
  });
});

describe("importDiscoverDataset (the SSE reader)", () => {
  const realFetch = globalThis.fetch;
  let realImport: typeof import("@/api/publicImport").importDiscoverDataset;
  beforeEach(async () => {
    realImport = (await vi.importActual<typeof import("@/api/publicImport")>("@/api/publicImport"))
      .importDiscoverDataset;
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  function stream(text: string): Response {
    const body = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode(text));
        c.close();
      },
    });
    return new Response(body, { status: 200 });
  }

  it("passes events on and resolves with the dataset id, skipping keepalives", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      stream(
        'event: job\ndata: {"job_id": "j"}\n\n: keepalive\n\nevent: progress\ndata: {"done": 1, "total": 2}\n\n' +
          'event: complete\ndata: {"dataset_id": "d1"}\n\n',
      ),
    );
    const seen: string[] = [];
    const id = await realImport(VID, (e) => seen.push(e.event));
    expect(id).toBe("d1");
    expect(seen).toEqual(["job", "progress", "complete"]);
  });

  it("rejects with the server's message on error, cancel or a refused request", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(stream('event: error\ndata: {"error": "too big"}\n\n'));
    await expect(realImport(VID, () => {})).rejects.toThrow("too big");
    globalThis.fetch = vi.fn().mockResolvedValue(stream('event: cancelled\ndata: {"message": "Import cancelled"}\n\n'));
    await expect(realImport(VID, () => {})).rejects.toThrow("Import cancelled");
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ detail: "turned off" }), { status: 403 }));
    await expect(realImport(VID, () => {})).rejects.toThrow("turned off");
  });
});
