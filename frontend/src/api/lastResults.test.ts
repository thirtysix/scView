import { describe, it, expect, vi, afterEach } from "vitest";
import { getDeconvolution, getDeconvolutionCheck } from "./deconvolution";
import { getGwasClusters } from "./gwas";
import { getSpatialAutocorr } from "./spatialAutocorr";
import { getSpotCheck } from "./spotDeconvolution";
import { getLastTfMotifs } from "./tfMotifs";

const respond = (body: unknown, status = 200) =>
  vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

// The panels ask for their last result on open: with allow_empty "none yet" is a null, not a 404 in the console.
const GETTERS: [string, () => Promise<unknown>, string][] = [
  ["bulk deconvolution", () => getDeconvolution("d1"), "/datasets/d1/deconvolution?allow_empty=true"],
  ["its self-check", () => getDeconvolutionCheck("d1"), "/datasets/d1/deconvolution/check?allow_empty=true"],
  ["spot composition self-check", () => getSpotCheck("d1"), "/datasets/d1/spot-deconvolution/check?allow_empty=true"],
  ["GWAS traits", () => getGwasClusters("d1", "cell type"), "/datasets/d1/gwas/clusters?allow_empty=true&column=cell%20type"],
  ["TF candidates", () => getLastTfMotifs("d1"), "/datasets/d1/tf-motifs?allow_empty=true"],
  ["Moran's I", () => getSpatialAutocorr("d1"), "/datasets/d1/spatial/autocorr?allow_empty=true"],
];

describe("last-result getters", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each(GETTERS)("%s asks with allow_empty and reads null as none", async (_name, get, url) => {
    const f = respond(null);
    vi.stubGlobal("fetch", f);
    expect(await get()).toBeNull();
    expect(String((f.mock.calls[0] as unknown[])[0])).toContain(url);
  });
});
