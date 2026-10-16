import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

vi.mock("@/api/client", () => ({ apiFetch: vi.fn().mockResolvedValue({ query: "", results: [] }) }));

import { GeneSearch } from "./GeneSearch";

describe("GeneSearch (multiple)", () => {
  it("hands a pasted list to onSelect whole and empties the box", () => {
    const onSelect = vi.fn();
    render(<GeneSearch datasetId="ds1" onSelect={onSelect} multiple placeholder="genes" />);
    const input = screen.getByPlaceholderText("genes") as HTMLInputElement;
    fireEvent.paste(input, { clipboardData: { getData: () => "CD3E\nMS4A1\nNKG7" } });
    expect(onSelect).toHaveBeenCalledWith("CD3E\nMS4A1\nNKG7");
    expect(input.value).toBe("");
  });

  it("empties the box after Enter on a typed name", () => {
    const onSelect = vi.fn();
    render(<GeneSearch datasetId="ds1" onSelect={onSelect} multiple placeholder="genes" />);
    const input = screen.getByPlaceholderText("genes") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "CD3E" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith("CD3E");
    expect(input.value).toBe("");
  });

  it("leaves a single pasted name to the autocomplete, and is off by default", () => {
    const onSelect = vi.fn();
    render(<GeneSearch datasetId="ds1" onSelect={onSelect} placeholder="genes" />);
    const input = screen.getByPlaceholderText("genes");
    fireEvent.paste(input, { clipboardData: { getData: () => "CD3E\nMS4A1" } });
    expect(onSelect).not.toHaveBeenCalled();
  });
});
