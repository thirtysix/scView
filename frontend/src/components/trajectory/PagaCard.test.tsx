import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import type { PagaView } from "@/api/trajectory";
import type { ObsColumnInfo } from "@/api/types";
import { PagaCard } from "./PagaCard";

const TIE =
  "The tree depends on a tie: 1 of its edges tie with an edge left out (1 pair), so another tree fits " +
  "PAGA's connectivity as well.";

function view(warnings?: string[]): PagaView {
  return {
    paga: {
      groupby: "cell_type", groups: ["A", "B", "C"], sizes: [10, 20, 30],
      edges: [[0, 1, 1], [1, 2, 1], [0, 2, 1]], tree: [[0, 1], [1, 2]], graph: "existing neighbour graph", n_cells: 60,
      ...(warnings ? { warnings } : {}),
    },
    positions: null, stale: null, root_group: null,
  };
}

function renderCard(v: PagaView) {
  const noop = vi.fn();
  return render(
    <PagaCard
      columns={[{ name: "cell_type" } as ObsColumnInfo]} groupby="cell_type" onGroupbyChange={noop} view={v}
      computing={false} error={null} onCompute={noop} show onShowChange={noop} threshold={0.1}
      onThresholdChange={noop} treeOnly={false} onTreeOnlyChange={noop} nVisible={3}
    />,
  );
}

describe("PagaCard", () => {
  it("says when the spanning tree depends on a tie", () => {
    renderCard(view([TIE]));
    expect(screen.getByTestId("paga-warning")).toHaveTextContent(TIE);
  });

  it("says nothing for a record without warnings, or from before they were kept", () => {
    renderCard(view([]));
    expect(screen.queryByTestId("paga-warning")).toBeNull();
    renderCard(view());
    expect(screen.queryByTestId("paga-warning")).toBeNull();
  });
});
