import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CapNote, Caption, capText, nText } from "./Caption";

describe("capText", () => {
  it("names the ordering and the total when rows were cut", () => {
    expect(capText({ shown: 200, total: 2311, by: "adjusted p" })).toBe("Top 200 of 2,311 by adjusted p.");
    expect(capText({ shown: 200, total: 2311, by: "adjusted p", rest: "the CSV has all" })).toBe(
      "Top 200 of 2,311 by adjusted p; the CSV has all.",
    );
  });

  it("says nothing when every row is shown", () => {
    expect(capText({ shown: 200, total: 200, by: "adjusted p" })).toBeNull();
    expect(capText({ shown: 200, total: 12, by: "adjusted p" })).toBeNull();
  });
});

describe("nText", () => {
  it("formats one or several n parts with thousands separators", () => {
    expect(nText({ value: 1234, unit: "cells" })).toBe("1,234 cells");
    expect(nText([{ value: 1234, of: 13836, unit: "cells" }, { value: 20, unit: "genes" }])).toBe(
      "1,234 of 13,836 cells; 20 genes",
    );
  });
});

describe("Caption", () => {
  it("renders the sentence followed by the n", () => {
    render(
      <Caption n={{ value: 13836, unit: "cells" }} testId="cap">
        Genes detected per cell.
      </Caption>,
    );
    expect(screen.getByTestId("cap")).toHaveTextContent("Genes detected per cell. n = 13,836 cells.");
  });

  it("renders only the n when there is no sentence", () => {
    render(<Caption n={{ value: 5, unit: "genes" }} testId="cap" />);
    expect(screen.getByTestId("cap")).toHaveTextContent(/^n = 5 genes\.$/);
  });
});

describe("CapNote", () => {
  it("shows the honest cap line only when rows were cut", () => {
    const { rerender } = render(<CapNote shown={50} total={120} by="|activity|" rest="the CSV has all" />);
    expect(screen.getByTestId("cap-note")).toHaveTextContent("Top 50 of 120 by |activity|; the CSV has all.");
    rerender(<CapNote shown={50} total={40} by="|activity|" />);
    expect(screen.queryByTestId("cap-note")).toBeNull();
  });
});
