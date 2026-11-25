import { describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { UploadOptions } from "@/api/upload";
import { useUploadTask } from "./useUploadTask";

/** A send that waits until the test settles it, and keeps the options it was given. */
function pending() {
  let opts: UploadOptions = {};
  let settle: (v: string) => void = () => {};
  let fail: (e: unknown) => void = () => {};
  const send = (o: UploadOptions) =>
    new Promise<string>((resolve, reject) => {
      opts = o;
      settle = resolve;
      fail = reject;
      o.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
  return { send, opts: () => opts, settle: (v: string) => settle(v), fail: (e: unknown) => fail(e) };
}

describe("useUploadTask", () => {
  it("names the upload while it runs, passes progress on, and clears it at the end", async () => {
    const { result } = renderHook(() => useUploadTask());
    const p = pending();
    let out: Promise<string> = Promise.resolve("");
    act(() => {
      out = result.current.run("sample.loom", p.send);
    });
    expect(result.current.name).toBe("sample.loom");
    act(() => p.opts().onProgress!({ loaded: 100, total: 100, file: "sample.loom", fileIndex: 1, fileCount: 1 }));
    expect(result.current.progress?.loaded).toBe(100);
    await act(async () => {
      p.settle("done");
      await out;
    });
    await expect(out).resolves.toBe("done");
    expect(result.current.name).toBeNull();
    expect(result.current.cancelled).toBeNull();
  });

  it("refuses a second upload while one runs", async () => {
    const { result } = renderHook(() => useUploadTask());
    const p = pending();
    act(() => void result.current.run("a.csv", p.send).catch(() => undefined));
    await expect(result.current.run("b.csv", pending().send)).rejects.toThrow("One upload at a time");
    await act(async () => p.settle("ok"));
  });

  it("Cancel aborts the signal, rejects with an AbortError and keeps the name for the note", async () => {
    const { result } = renderHook(() => useUploadTask());
    const p = pending();
    let out: Promise<string> = Promise.resolve("");
    act(() => {
      out = result.current.run("contigs.csv", p.send);
    });
    const caught = out.catch((e: unknown) => e);
    await act(async () => {
      result.current.cancel();
      await caught;
    });
    expect((await caught as DOMException).name).toBe("AbortError");
    expect(p.opts().signal!.aborted).toBe(true);
    expect(result.current.name).toBeNull();
    expect(result.current.cancelled).toBe("contigs.csv");
  });

  it("an error is no cancel", async () => {
    const { result } = renderHook(() => useUploadTask());
    const p = pending();
    let out: Promise<string> = Promise.resolve("");
    act(() => {
      out = result.current.run("contigs.csv", p.send);
    });
    const caught = out.catch((e: unknown) => e);
    await act(async () => {
      p.fail(new Error("No cell matched."));
      await caught;
    });
    expect(result.current.cancelled).toBeNull();
  });

  it("aborts on unmount only when asked to", () => {
    const keep = pending();
    const a = renderHook(() => useUploadTask());
    act(() => void a.result.current.run("a.loom", keep.send).catch(() => undefined));
    a.unmount();
    expect(keep.opts().signal!.aborted).toBe(false);

    const stop = pending();
    const b = renderHook(() => useUploadTask({ abortOnUnmount: true }));
    act(() => void b.result.current.run("b.loom", stop.send).catch(() => undefined));
    b.unmount();
    expect(stop.opts().signal!.aborted).toBe(true);
  });
});
