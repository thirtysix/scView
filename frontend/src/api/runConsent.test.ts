import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { withConversionConsent } from "./pipelineRun";

/* v0.81 (QA round 7, B1): no run converts a dataset to large-dataset layout without a yes.
 * The run routes refuse by default; the UI sends allow_large_mode false first and asks. */

const REFUSAL =
  "In memory the run needs about 4.2 GB more at its peak; scView has 3.0 GB free, so it runs in " +
  "large-dataset mode. Large-dataset mode converts the dataset for good. Nothing was run: run it " +
  "again allowing large-dataset mode to convert it (allow_large_mode true), or free memory first.";

describe("withConversionConsent", () => {
  it("runs without the yes, and asks only when the backend refuses a conversion", async () => {
    const call = vi.fn(async (allow: boolean) => (allow ? "converted" : "in memory"));
    const ask = vi.fn(() => true);
    expect(await withConversionConsent(call, ask)).toBe("in memory");
    expect(call.mock.calls).toEqual([[false]]);
    expect(ask).not.toHaveBeenCalled();
  });

  it("runs again allowing it only after a yes", async () => {
    const call = vi.fn(async (allow: boolean) => {
      if (!allow) throw new Error(REFUSAL);
      return "converted";
    });
    const ask = vi.fn((q: string) => q.length > 0);
    expect(await withConversionConsent(call, ask)).toBe("converted");
    expect(call.mock.calls).toEqual([[false], [true]]);
    expect(ask.mock.calls[0]![0]).toContain("Run it in large-dataset mode anyway?");
    expect(ask.mock.calls[0]![0]).not.toContain("Nothing was run");
  });

  it("keeps the refusal on a no, and other errors as they are", async () => {
    const refused = vi.fn(async (allow: boolean): Promise<string> => {
      throw new Error(allow ? "unexpected" : REFUSAL);
    });
    await expect(withConversionConsent(refused, () => false)).rejects.toThrow("Nothing was run");
    expect(refused.mock.calls).toEqual([[false]]);
    const ask = vi.fn(() => true);
    await expect(
      withConversionConsent(async () => {
        throw new Error("Pipeline run failed: boom");
      }, ask),
    ).rejects.toThrow("boom");
    expect(ask).not.toHaveBeenCalled();
  });
});

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

describe("every UI caller of a run route", () => {
  it("sends allow_large_mode, never leaving the conversion to a default", () => {
    const root = resolve(__dirname, "..");
    const callers: string[] = [];
    for (const file of sources(root)) {
      const text = readFileSync(file, "utf8");
      for (const m of text.matchAll(/\/(assessment\/run(?:-stream)?|rerun)`/g)) {
        const after = text.slice(m.index, m.index + 600);
        callers.push(`${relative(root, file)}:${m[1]}`);
        expect(after, `${relative(root, file)} ${m[1]}`).toMatch(/allow_large_mode: allow(Large)?\b/);
      }
    }
    expect(callers.sort()).toEqual([
      "api/assistant.ts:assessment/run",
      "api/provenance.ts:rerun",
      "components/compare/JointSummary.tsx:assessment/run-stream",
      "components/panels/DataAssessmentPanel.tsx:assessment/run-stream",
      "lib/copilotPlan.ts:assessment/run-stream",
    ]);
  });

  it("asks before it lets a run convert (the helper or the panel's own confirm)", () => {
    const root = resolve(__dirname, "..");
    const read = (p: string) => readFileSync(join(root, p), "utf8");
    for (const p of [
      "components/compare/JointSummary.tsx",
      "components/assistant/AssistantChat.tsx",
      "components/unified/UnifiedEnrichmentSubtab.tsx",
      "components/panels/ProvenancePanel.tsx",
    ]) {
      expect(read(p), p).toContain("withConversionConsent(");
    }
    expect(read("components/panels/DataAssessmentPanel.tsx")).toContain("window.confirm(conversionQuestion(msg))");
  });
});

describe("runErrorText", () => {
  it("turns a declined conversion into a plain sentence and leaves other errors as the server's reason", async () => {
    const { runErrorText, DECLINED_CONVERSION } = await import("@/api/pipelineRun");
    const refusal = new Error(
      'API error 409: {"detail":"In memory the run needs 1.8 GB. Nothing was run: run it again allowing large-dataset mode to convert it (allow_large_mode true), or free memory first."}',
    );
    expect(runErrorText(refusal)).toBe(DECLINED_CONVERSION);
    expect(runErrorText(new Error('API error 400: {"detail":"No clusters."}'))).toBe("No clusters.");
  });
});
