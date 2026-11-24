import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

/**
 * Nothing from a dataset goes to the language model until the user asks (a button, a chat
 * message, an AI run option). These are the client calls whose backend routes call the model;
 * each caller below sends them from a click only. A new caller must do the same: check it,
 * then add it here.
 */
const SRC = resolve(__dirname, "..");

const LLM_CALLS = /\b(assistantChat|assistantChatStream|getMethods|checkClaims|postQcReport|streamQcReport)\(|assessment\/suggest/;

const CLICK_ONLY_CALLERS = [
  "components/assistant/AssistantChat.tsx", // Send, suggestion chips, Write methods, Check claims
  "components/panels/QcReportCard.tsx", // Write report / Regenerate buttons
  "components/panels/DataAssessmentPanel.tsx", // Get AI suggestions button
];

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

describe("language model calls", () => {
  it("are sent only from the reviewed click-only callers", () => {
    const callers = sources(SRC)
      .map((p) => relative(SRC, p))
      .filter((p) => !p.startsWith("api/"))
      .filter((p) => LLM_CALLS.test(readFileSync(join(SRC, p), "utf8")));
    expect(callers.sort()).toEqual([...CLICK_ONLY_CALLERS].sort());
  });
});
