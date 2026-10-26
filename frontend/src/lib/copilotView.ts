import type { OnScreen } from "@/api/assistant";
import type { PanelId, UnifiedSubtab } from "@/lib/constants";
import type { OnScreenDE, OnScreenEnrichment } from "@/stores/onScreenStore";

interface Slots {
  de: OnScreenDE | null;
  enrichment: OnScreenEnrichment | null;
  compareTab: string | null;
}

/** The part of the co-pilot's view context that says which tab is open and which
 *  result it shows (backend core/screen_facts.py). Only the open tab's result is
 *  named, so a DE result left behind on another tab is not "this result". */
export function screenContext(
  panel: PanelId,
  subtab: UnifiedSubtab,
  slots: Slots,
): { subtab?: string; on_screen?: OnScreen } {
  if (panel === "unified") {
    if (subtab === "de" && slots.de) {
      return { subtab, on_screen: { kind: "de", id: slots.de.id, label: slots.de.label } };
    }
    if (subtab === "enrichment" && slots.enrichment) {
      return { subtab, on_screen: { kind: "enrichment", ...slots.enrichment } };
    }
    return { subtab };
  }
  if (panel === "compare" && slots.compareTab) {
    return { on_screen: { kind: "compare", tab: slots.compareTab } };
  }
  return {};
}
