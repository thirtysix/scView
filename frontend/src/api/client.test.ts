import { describe, it, expect } from "vitest";
import { BACKEND_DOWN_MESSAGE, readError } from "./client";

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
