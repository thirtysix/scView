/**
 * Whether the browser will give this page a new WebGL context right now.
 *
 * After a GPU reset that the browser blames on the page (a draw that hung the
 * GPU), Chrome refuses every new WebGL context for that page until it is
 * reloaded. Remounting a plot then fails again at once ("WebGL context creation
 * failed"), so a Retry button has to check first and ask for a reload instead.
 * The probe context is released straight away so it does not count against the
 * browser's cap on live contexts.
 */
export function canCreateWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    const gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) as
      | WebGLRenderingContext
      | WebGL2RenderingContext
      | null;
    if (!gl || gl.isContextLost()) return false;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

/** deck.gl's error when the browser would not create a context at all. */
export function isContextCreationFailure(err: Error | null | undefined): boolean {
  return /context creation failed/i.test(err?.message ?? "");
}

/**
 * deck.gl's errors for a lost or refused WebGL context. Not a bare "context"
 * match: a layer error names its layer, and the split grid's grey background
 * layer is called `split-context`.
 */
export function isContextLoss(err: Error | null | undefined): boolean {
  return /webgl context|context (is )?lost|context creation failed/i.test(err?.message ?? "");
}
