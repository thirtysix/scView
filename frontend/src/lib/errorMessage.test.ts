import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { errorMessage, errorStatus } from "./errorMessage";

describe("errorMessage", () => {
  it("gives the server's reason, not the raw response", () => {
    expect(errorMessage(new Error('API error 400: {"detail":"This dataset has no counts."}'))).toBe(
      "This dataset has no counts.",
    );
    expect(errorMessage(new Error("API error 404: Column 'x' not found."))).toBe("Column 'x' not found.");
    expect(errorMessage(new Error("stream ended"))).toBe("stream ended");
    expect(errorMessage("plain")).toBe("plain");
    expect(errorMessage(new Error('API error 400: {"detail":{"code":1}}'))).toBe('{"code":1}');
  });

  it("gives a 422 validation list's messages", () => {
    expect(errorMessage(new Error('API error 422: {"detail":[{"msg":"bad"}]}'))).toBe("bad");
    expect(
      errorMessage(new Error('API error 422: {"detail":[{"msg":"Value error, no genes"},{"msg":"too many"}]}')),
    ).toBe("no genes; too many");
  });

  it("reads the status of an apiFetch error", () => {
    expect(errorStatus(new Error('API error 404: {"detail":"x"}'))).toBe(404);
    expect(errorStatus(new Error("network down"))).toBeNull();
    expect(errorStatus("API error 500: x")).toBe(500);
  });
});

/**
 * `e instanceof Error ? e.message : String(e)` shows the raw `API error 400: {"detail": ...}`
 * body to the user. Use `errorMessage(e)` (and `errorStatus(e)` for the code) instead.
 */
const SRC = resolve(__dirname, "..");

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

describe("error text", () => {
  it("no source builds a message from the raw error outside lib/errorMessage.ts", () => {
    const hits = sources(SRC)
      .filter((p) => relative(SRC, p) !== join("lib", "errorMessage.ts"))
      .flatMap((p) =>
        readFileSync(p, "utf8")
          .split("\n")
          .flatMap((line, i) =>
            /instanceof Error\s*\?\s*\w+\.message\s*:\s*String\(/.test(line) ? [`${relative(SRC, p)}:${i + 1}`] : [],
          ),
      );
    expect(hits).toEqual([]);
  });
});
