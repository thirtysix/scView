import { describe, it, expect } from "vitest";
import {
  BACKEND_DOWN_MESSAGE,
  LargeDatasetRefusal,
  apiFetch,
  isLargeRefusal,
  largeRefusalMessage,
  readError,
} from "./client";

const nginx502 = "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body><center><h1>502 Bad Gateway</h1></center></body></html>";

describe("readError", () => {
  it("replaces nginx's error page for a backend that is down", async () => {
    expect(await readError(new Response(nginx502, { status: 502 }))).toBe(BACKEND_DOWN_MESSAGE);
    expect(await readError(new Response("", { status: 504 }))).toBe(BACKEND_DOWN_MESSAGE);
  });

  it("keeps the backend's own error text", async () => {
    const body = JSON.stringify({ detail: "Dataset is busy" });
    expect(await readError(new Response(body, { status: 503 }))).toBe(body);
    expect(await readError(new Response("<html>not found</html>", { status: 404 }))).toBe("<html>not found</html>");
  });
});

describe("large-dataset refusals", () => {
  const refusal = JSON.stringify({
    detail: "Gene programs is not available for large datasets yet: it factorises every cell.",
    large_dataset: true,
  });

  it("reads the plain reason from a 422 marked large_dataset", () => {
    expect(largeRefusalMessage(422, refusal)).toBe(
      "Gene programs is not available for large datasets yet: it factorises every cell.",
    );
    expect(largeRefusalMessage(422, JSON.stringify({ detail: "bad step" }))).toBeNull();
    expect(largeRefusalMessage(400, refusal)).toBeNull();
    expect(largeRefusalMessage(422, "<html>")).toBeNull();
  });

  it("apiFetch throws the reason itself, without the status and JSON", async () => {
    const real = globalThis.fetch;
    globalThis.fetch = (async () => new Response(refusal, { status: 422 })) as typeof fetch;
    try {
      const err = await apiFetch("/x").catch((e: unknown) => e);
      expect(err).toBeInstanceOf(LargeDatasetRefusal);
      expect((err as Error).message).toMatch(/^Gene programs is not available/);
      expect(isLargeRefusal(err)).toBe(true);
      expect(isLargeRefusal((err as Error).message)).toBe(true);
      expect(isLargeRefusal(new Error("API error 500: boom"))).toBe(false);
    } finally {
      globalThis.fetch = real;
    }
  });
});
