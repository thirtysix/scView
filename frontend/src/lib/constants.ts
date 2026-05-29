export const API_BASE = "/api/v1";

export const PANEL_IDS = [
  "load",
  "ingest",
  "assessment",
  "overview",
  "unified",
  "observations",
  "expression",
  "genesets",
  "markers",
  "trajectory",
  "provenance",
] as const;

export type PanelId = (typeof PANEL_IDS)[number];

export const PANEL_LABELS: Record<PanelId, string> = {
  load: "Load Data",
  ingest: "Add Data",
  assessment: "Data Assessment",
  overview: "Visualizations",
  unified: "Unified View",
  observations: "Observations",
  expression: "Gene Expression",
  genesets: "Gene Sets & Enrichment",
  markers: "Marker Genes",
  trajectory: "Trajectory",
  provenance: "History",
};
