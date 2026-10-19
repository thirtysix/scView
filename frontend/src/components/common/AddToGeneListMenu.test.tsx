import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { AddToGeneListMenu } from "./AddToGeneListMenu";

const lists = () => useGeneListsStore.getState().lists;

beforeEach(() => {
  localStorage.clear();
  useGeneListsStore.setState({ lists: [], selectedId: null });
});

describe("AddToGeneListMenu", () => {
  it("makes a new list under the offered name, with the source", () => {
    render(<AddToGeneListMenu genes={["CD14", "LYZ", "cd14"]} defaultName="Markers: Mono" source="Markers of Mono" />);
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "New list…" }));
    expect(screen.getByLabelText("New list name")).toHaveValue("Markers: Mono");
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(lists().map((l) => [l.name, l.genes, l.source])).toEqual([["Markers: Mono", ["CD14", "LYZ"], "Markers of Mono"]]);
    expect(screen.getByRole("status")).toHaveTextContent("New list “Markers: Mono”: 2 genes");
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("adds to an existing list only the genes it lacks, reading the genes when picked", () => {
    useGeneListsStore.getState().create("Mine", ["CD14"]);
    let genes = ["CD14"];
    const { rerender } = render(<AddToGeneListMenu genes={() => genes} defaultName="x" label="Up" />);
    genes = ["CD14", "FCGR3A", "LYZ"];
    rerender(<AddToGeneListMenu genes={() => genes} defaultName="x" label="Up" />);
    fireEvent.click(screen.getByRole("button", { name: "Up" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Mine/ }));
    expect(lists()[0]!.genes).toEqual(["CD14", "FCGR3A", "LYZ"]);
    expect(screen.getByRole("status")).toHaveTextContent("Added 2 genes to “Mine”");
    fireEvent.click(screen.getByRole("button", { name: "Up" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Mine/ }));
    expect(screen.getByRole("status")).toHaveTextContent("“Mine” has them all");
  });

  it("closes on Escape or a click outside", () => {
    render(<AddToGeneListMenu genes={["A"]} defaultName="x" />);
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("menu")).toBeNull();
  });
});
