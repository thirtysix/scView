import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { ScatterExportMenu } from "./ScatterExportMenu";

const open = () => fireEvent.click(screen.getByRole("button", { name: "Download grid" }));
const labels = () => screen.getAllByRole("menuitem").map((m) => m.textContent ?? "");

describe("ScatterExportMenu", () => {
  it("offers PNG, SVG and PDF of one figure, as before paging", () => {
    render(<ScatterExportMenu nPoints={100} onPng={vi.fn()} onFigure={vi.fn(async () => {})} label="Download grid" />);
    open();
    expect(screen.getAllByRole("menuitem")).toHaveLength(3);
    expect(screen.getByRole("menuitem", { name: /SVG \(vector\)/ })).toBeInTheDocument();
    expect(screen.queryByText(/All \d+ panels/)).toBeNull();
  });

  it("a paged grid: the page's three formats, then all panels on one sheet", async () => {
    const onPng = vi.fn();
    const onFigure = vi.fn(async () => {});
    const allPng = vi.fn();
    const allFigure = vi.fn(async () => {});
    render(
      <ScatterExportMenu
        nPoints={100}
        onPng={onPng}
        onFigure={onFigure}
        label="Download grid"
        pageNote="This page, panels 7–12 of 24."
        allPanels={{ count: 24, nPoints: 400, onPng: allPng, onFigure: allFigure }}
      />,
    );
    open();
    expect(screen.getAllByRole("menuitem")).toHaveLength(6);
    expect(labels().join("|")).toMatch(/All 24 panels: PNG.*All 24 panels: SVG \(vector\).*All 24 panels: PDF \(vector\)/);
    expect(screen.getAllByText(/This page, panels 7–12 of 24\./).length).toBeGreaterThan(0);
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /All 24 panels: PDF/ }));
    });
    expect(allFigure).toHaveBeenCalledWith("pdf", "vector");
    expect(onFigure).not.toHaveBeenCalled();
  });

  it("offers all panels with points as an image first when the sheet is large", () => {
    render(
      <ScatterExportMenu
        nPoints={1000}
        onPng={vi.fn()}
        onFigure={vi.fn(async () => {})}
        label="Download grid"
        pageNote="This page."
        allPanels={{ count: 24, nPoints: 300_000, onPng: vi.fn(), onFigure: vi.fn(async () => {}) }}
      />,
    );
    open();
    expect(screen.getByRole("menuitem", { name: /All 24 panels: SVG, points as image/ })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /All 24 panels: SVG, every point vector/ })).toBeInTheDocument();
  });
});
