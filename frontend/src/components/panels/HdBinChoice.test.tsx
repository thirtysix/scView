import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { HdBinOption } from "@/lib/ingestUpload";
import { HdBinChoice } from "./HdBinChoice";

const OPTIONS: HdBinOption[] = [
  { um: 2, matrixBytes: 181e6, maxBins: 11_222_500, refused: true },
  { um: 8, matrixBytes: 112e6, maxBins: 701_406, refused: false },
  { um: 16, matrixBytes: 90e6, maxBins: 175_352, refused: false },
];

describe("HdBinChoice", () => {
  it("starts on 16 um, cannot choose 2 um, and 8 um can be chosen", () => {
    const onChoose = vi.fn();
    render(<HdBinChoice options={OPTIONS} onChoose={onChoose} onCancel={vi.fn()} />);
    const [two, eight, sixteen] = screen.getAllByRole("radio") as HTMLInputElement[];
    expect(two!.disabled).toBe(true);
    expect(sixteen!.checked).toBe(true);
    expect(screen.getByText(/clusters follow the anatomy/)).toBeTruthy();
    expect(screen.getByText(/at most 701,406 bins/)).toBeTruthy();
    fireEvent.click(eight!);
    fireEvent.click(screen.getByRole("button", { name: /Upload 8 µm bins/ }));
    expect(onChoose).toHaveBeenCalledWith(8);
  });

  it("cancels", () => {
    const onCancel = vi.fn();
    render(<HdBinChoice options={OPTIONS} onChoose={vi.fn()} onCancel={onCancel} />);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
  });
});
