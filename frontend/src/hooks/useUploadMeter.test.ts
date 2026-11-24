import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useUploadMeter } from "./useUploadMeter";

const MB = 1e6;
const p = (loaded: number, total = 100 * MB) => ({ loaded, total, file: "sample.rds", fileIndex: 1, fileCount: 1 });

describe("useUploadMeter", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("redraws at most four times a second, and gives speed and time left after a few seconds", () => {
    const { result } = renderHook(() => useUploadMeter());
    act(() => result.current.start());
    act(() => result.current.onProgress(p(0)));
    expect(result.current.progress?.loaded).toBe(0);
    act(() => {
      vi.advanceTimersByTime(100);
      result.current.onProgress(p(0.2 * MB));
    });
    expect(result.current.progress?.loaded).toBe(0); // within 250 ms of the last draw
    for (let i = 0; i < 40; i++) {
      act(() => {
        vi.advanceTimersByTime(100);
        result.current.onProgress(p(0.2 * MB + (i + 1) * 0.2 * MB)); // 2 MB/s
      });
    }
    // drawn up to 250 ms late, so within a few percent of 2 MB/s
    expect(result.current.rate! / (2 * MB)).toBeGreaterThan(0.95);
    expect(result.current.rate! / (2 * MB)).toBeLessThan(1.02);
    expect(result.current.left! / ((100 - 8.2) / 2)).toBeGreaterThan(0.98);
    expect(result.current.left! / ((100 - 8.2) / 2)).toBeLessThan(1.06);
  });

  it("shows a stall: with no progress events the speed falls", () => {
    const { result } = renderHook(() => useUploadMeter());
    act(() => result.current.start());
    for (let i = 0; i <= 50; i++) {
      act(() => {
        result.current.onProgress(p(i * 0.2 * MB));
        vi.advanceTimersByTime(100);
      });
    }
    const before = result.current.rate!;
    act(() => {
      vi.advanceTimersByTime(6000);
    });
    expect(result.current.rate!).toBeLessThan(before / 2);
  });

  it("always draws the last byte, and start clears the last upload", () => {
    const { result } = renderHook(() => useUploadMeter());
    act(() => result.current.start());
    act(() => result.current.onProgress(p(1)));
    act(() => result.current.onProgress(p(100 * MB)));
    expect(result.current.progress?.loaded).toBe(100 * MB);
    act(() => result.current.start());
    expect(result.current.progress).toBeNull();
    expect(result.current.rate).toBeNull();
  });
});
