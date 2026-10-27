import type { AssistantAction } from "@/api/assistant";
import type { PanelId, UnifiedSubtab } from "@/lib/constants";
import { useComparePrefillStore } from "@/stores/comparePrefillStore";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

/** Do one allow-listed view action the co-pilot returned (nothing here runs a
 *  computation: run actions go through a Confirm card or a plan). */
export function applyViewAction(a: AssistantAction): void {
  const view = useViewStore.getState();
  if (a.type === "set_color_by" && a.column) {
    useSettingsStore.getState().setColorBy(a.column);
    useSelectionStore.getState().setHighlight(null);
    view.setPanel("unified");
  } else if (a.type === "highlight_cluster" && a.column && a.value) {
    useSettingsStore.getState().setColorBy(a.column);
    useSelectionStore.getState().setHighlight({ column: a.column, value: a.value });
    view.setPanel("unified");
  } else if (a.type === "open_panel" && a.panel) {
    view.setPanel(a.panel as PanelId);
  } else if (a.type === "set_embedding" && a.embedding) {
    useSettingsStore.getState().setEmbedding(a.embedding);
    view.setPanel("unified");
  } else if (a.type === "set_subtab" && a.subtab) {
    useUnifiedViewStore.getState().setActiveSubtab(a.subtab as UnifiedSubtab);
    view.setPanel("unified");
  } else if (a.type === "set_groupby" && a.column) {
    useUnifiedViewStore.getState().setGroupByColumn(a.column);
    view.setPanel("unified");
  } else if (a.type === "clear_highlight") {
    useSelectionStore.getState().setHighlight(null);
  } else if (a.type === "clear_overlay") {
    useUnifiedViewStore.getState().clearOverlayState();
  } else if (a.type === "show_gene" && a.gene) {
    view.setPendingGene(a.gene);
    view.setPanel("expression");
  } else if (a.type === "open_de_selection") {
    // DE pre-set to selection vs rest; the user clicks Run there.
    useDESettingsStore.getState().update({ mode: "selection" });
    useUnifiedViewStore.getState().setActiveSubtab("de");
    view.setPanel("unified");
  } else if (a.type === "open_de_compare" && a.column && a.group_a && a.group_b) {
    // DE between two groups, pre-filled; the user clicks Run there. Over the whole
    // dataset: a scope left from before is cleared, and the current highlight is
    // marked as seen so the tab does not turn it into a scope on opening.
    useDESettingsStore.getState().update({
      mode: "groups",
      splitBy: a.column,
      groupA: a.group_a,
      groupB: a.group_b,
      withinCol: "",
      withinVal: "",
      scopedHighlight: useSelectionStore.getState().highlightedGroup,
    });
    useUnifiedViewStore.getState().setActiveSubtab("de");
    view.setPanel("unified");
  } else if (a.type === "open_compare_tab" && a.tab) {
    useComparePrefillStore.getState().setPrefill({
      tab: a.tab,
      condition: a.condition,
      group: a.column,
      sample: a.sample,
    });
    view.setPanel("compare");
  } else if (a.type === "show_marker_plot") {
    view.setMarkerView(a.plot === "dotplot" ? "dotplot" : "heatmap");
    view.setPanel("markers");
  }
}
