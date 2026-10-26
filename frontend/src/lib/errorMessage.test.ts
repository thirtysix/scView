import { describe, expect, it } from "vitest";
import { errorMessage } from "./errorMessage";

describe("errorMessage", () => {
  it("gives the server's reason, not the raw response", () => {
    expect(errorMessage(new Error('API error 400: {"detail":"This dataset has no counts."}'))).toBe(
      "This dataset has no counts.",
    );
    expect(errorMessage(new Error("API error 404: Column 'x' not found."))).toBe("Column 'x' not found.");
    expect(errorMessage(new Error("stream ended"))).toBe("stream ended");
    expect(errorMessage("plain")).toBe("plain");
    expect(errorMessage(new Error('API error 422: {"detail":[{"msg":"bad"}]}'))).toBe('[{"msg":"bad"}]');
  });
});
