export const API_BASE = "/api/v1";

export const PANEL_IDS = [
  "load",
  "assessment",
  "unified",
  "observations",
  "expression",
  "genesets",
  "markers",
  "trajectory",
  "compare",
  "provenance",
  "assistant",
] as const;

export type PanelId = (typeof PANEL_IDS)[number];

export const PANEL_LABELS: Record<PanelId, string> = {
  load: "Data",
  assessment: "Data Assessment",
  unified: "Unified View",
  observations: "Observations",
  expression: "Gene Expression",
  genesets: "Gene Sets & Enrichment",
  markers: "Marker Genes",
  trajectory: "Trajectory",
  compare: "Compare",
  provenance: "History",
  assistant: "AI Co-pilot",
};
