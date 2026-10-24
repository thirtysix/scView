import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RequestError } from "./RequestError";

describe("RequestError", () => {
  it("shows a large-dataset refusal as a note, not an alert", () => {
    render(<RequestError message="Milo is not available for large datasets yet: it needs the counts." />);
    expect(screen.getByRole("note").textContent).toMatch(/^Milo is not available/);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows any other error as an alert, and nothing without one", () => {
    const { container, rerender } = render(<RequestError message="API error 500: boom" />);
    expect(screen.getByRole("alert").textContent).toBe("API error 500: boom");
    rerender(<RequestError message={null} />);
    expect(container.textContent).toBe("");
  });
});
