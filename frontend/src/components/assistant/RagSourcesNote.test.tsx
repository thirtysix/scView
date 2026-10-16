import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { RagStatus } from "@/api/assistant";

const { getRagStatus } = vi.hoisted(() => ({ getRagStatus: vi.fn() }));
vi.mock("@/api/assistant", () => ({ getRagStatus }));

import { RagSourcesNote, describeSources, resetRagStatus } from "./RagSourcesNote";

const INDEXED: RagStatus = {
  enabled: true,
  corpora: {
    literature: { documents: 1234, chunks: 5000, last_added: "2026-06-07T10:00:00+00:00" },
    tutorials: { documents: 1, chunks: 40, last_added: "2026-05-01T10:00:00+00:00" },
  },
};

describe("describeSources", () => {
  it("counts papers and tutorial docs and dates the newest ingest", () => {
    const { line, detail } = describeSources(INDEXED);
    expect(line).toMatch(/^Your dataset \+ 1,234 papers \+ 1 tutorial doc indexed · updated /);
    expect(line).toContain("2026");
    expect(detail).toContain("Literature: 1,234 documents in 5,000 chunks");
  });

  it("says plainly when retrieval is off, or unreachable", () => {
    expect(describeSources({ enabled: false, corpora: {}, reason: "no_database" }).line).toMatch(
      /not set up/,
    );
    expect(describeSources({ enabled: false, corpora: {}, reason: "no_api_key" }).detail).toMatch(
      /DEEPINFRA_API_KEY/,
    );
    expect(describeSources({ enabled: true, corpora: {}, error: "database unreachable" }).line).toMatch(
      /unreachable/,
    );
    expect(describeSources({ enabled: true, corpora: {} }).line).toBe(
      "Your dataset + no papers + no tutorial docs indexed",
    );
  });
});

describe("RagSourcesNote", () => {
  beforeEach(() => {
    resetRagStatus();
    getRagStatus.mockReset();
  });

  it("renders the line, and the drawer and panel share one request", async () => {
    getRagStatus.mockResolvedValue(INDEXED);
    render(
      <>
        <RagSourcesNote />
        <RagSourcesNote />
      </>,
    );
    expect(await screen.findAllByText(/1,234 papers/)).toHaveLength(2);
    expect(getRagStatus).toHaveBeenCalledTimes(1);
  });

  it("stays out of the way when the status can't be read", async () => {
    getRagStatus.mockRejectedValue(new Error("offline"));
    const { container } = render(<RagSourcesNote />);
    await Promise.resolve();
    expect(container).toBeEmptyDOMElement();
  });
});
