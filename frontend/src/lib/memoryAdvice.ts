/** How to give scView more memory, in words: a local install (SCVIEW_MEMORY_GB in .env, then a
 *  restart) or a hosted site (DEPLOYMENT_MODE=public), whose users can do neither and ask the
 *  site admin. The frontend twin of the backend's `memory_budget.more_memory_advice`; the mode
 *  comes from GET /system/memory (`hosted`), which the sidebar's memory gauge polls. */

let hostedSite = false;

/** Called by the memory gauge with each /system/memory reading. */
export function setHosted(hosted: boolean | undefined): void {
  hostedSite = Boolean(hosted);
}

export function isHosted(): boolean {
  return hostedSite;
}

const GB = 1024 ** 3;
export const gbText = (bytes: number) =>
  bytes / GB >= 10 ? (bytes / GB).toFixed(0) : (bytes / GB).toFixed(1);

/** One sentence, ending in a full stop: "Ask the site admin for more memory (this site has 6.0 GB)."
 *  on a hosted site, else the .env advice. `alternative`: what else would do, without its "or". */
export function moreMemoryAdvice(
  opts: { limitBytes?: number | null; alternative?: string; hosted?: boolean } = {},
): string {
  const hosted = opts.hosted ?? hostedSite;
  const tail = opts.alternative ? `, or ${opts.alternative}` : "";
  if (!hosted) return `Raise SCVIEW_MEMORY_GB in the .env file in the scView folder, then restart scView${tail}.`;
  const has = opts.limitBytes ? ` (this site has ${gbText(opts.limitBytes)} GB)` : "";
  return `Ask the site admin for more memory${has}${tail}.`;
}
