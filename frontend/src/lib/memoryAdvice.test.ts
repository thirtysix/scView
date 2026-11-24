import { afterEach, describe, expect, it } from "vitest";

import { declinedConversion } from "@/api/pipelineRun";
import { isHosted, moreMemoryAdvice, setHosted } from "./memoryAdvice";

const GB = 1024 ** 3;

describe("moreMemoryAdvice", () => {
  afterEach(() => setHosted(false));

  it("says .env on a local install and the site admin on a hosted site", () => {
    expect(moreMemoryAdvice()).toBe(
      "Raise SCVIEW_MEMORY_GB in the .env file in the scView folder, then restart scView.",
    );
    expect(moreMemoryAdvice({ hosted: true, limitBytes: 6 * GB })).toBe(
      "Ask the site admin for more memory (this site has 6.0 GB).",
    );
    expect(moreMemoryAdvice({ hosted: true, limitBytes: 12 * GB, alternative: "leave this step out" })).toBe(
      "Ask the site admin for more memory (this site has 12 GB), or leave this step out.",
    );
  });

  it("follows the mode the gauge last read", () => {
    expect(isHosted()).toBe(false);
    expect(declinedConversion()).toMatch(/raise scView's memory limit/);
    setHosted(true);
    expect(moreMemoryAdvice()).toBe("Ask the site admin for more memory.");
    expect(declinedConversion()).toMatch(/ask the site admin for more memory to run it in memory\.$/);
    setHosted(undefined); // an older backend: no field
    expect(isHosted()).toBe(false);
  });
});
