import { apiFetch } from "./client";

/** A Reactome shape: arrow (triangle a-b-c), circle (c, r), double circle (r, r1), stop (line a-b), box (a-b). */
export interface DiagramShape {
  type: "ARROW" | "CIRCLE" | "DOUBLE_CIRCLE" | "STOP" | "BOX";
  empty: boolean;
  a?: [number, number];
  b?: [number, number];
  c?: [number, number];
  r?: number;
  r1?: number;
  s?: string;
}

/** [x1, y1, x2, y2] */
export type Segment = [number, number, number, number];

export type NodeClass =
  | "Protein" | "Complex" | "EntitySet" | "Chemical" | "Gene" | "RNA" | "Entity" | "ProcessNode" | "EncapsulatedNode";

export interface DiagramNode {
  id: number;
  reactome_id: number | null;
  class: NodeClass;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** Primary gene symbols (a complex or set: its members'); absent for chemicals and pathway boxes. */
  genes?: string[];
  /** A pathway box's (or entity's) Reactome id. */
  st_id?: string;
  drug?: boolean;
  disease?: boolean;
  end?: DiagramShape;
}

export interface DiagramReaction {
  id: number;
  reactome_id: number | null;
  label: string;
  type: string;
  segments: Segment[];
  shape?: DiagramShape;
  end?: DiagramShape;
}

export interface DiagramConnector {
  node: number;
  edge: number | null;
  type: "INPUT" | "OUTPUT" | "CATALYST" | "ACTIVATOR" | "INHIBITOR";
  segments: Segment[];
  end?: DiagramShape;
}

export interface DiagramCompartment {
  id: number;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  text: [number, number];
  inset?: { x: number; y: number; w: number; h: number };
}

/** The renderer's model of one Reactome diagram (backend core/reactome.py build_model). */
export interface PathwayDiagramModel {
  /** The diagram drawn (the requested pathway's, or its nearest ancestor's). */
  st_id: string;
  name: string;
  species: "human" | "mouse" | "";
  /** The pathway asked for. */
  requested: string;
  release: number;
  attribution: string;
  licence: string;
  licence_url: string;
  citation: string;
  /** The requested pathway in Reactome's browser. */
  url: string;
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  compartments: DiagramCompartment[];
  nodes: DiagramNode[];
  reactions: DiagramReaction[];
  connectors: DiagramConnector[];
  links: { kind: string; segments: Segment[]; end?: DiagramShape }[];
  n_genes: number;
  /** An overview of sub-pathway boxes only: nothing to colour; open a suggestion instead. */
  too_broad: boolean;
  suggestions: { st_id: string; name: string }[];
  message: string;
  /** A pathway drawn inside an ancestor's diagram: its outline and reactions. */
  highlight: {
    st_id: string;
    name: string;
    /** false: the outline is of the nearest ancestor of the requested pathway this diagram draws. */
    exact: boolean;
    outline: [number, number][];
    reactions: number[];
  } | null;
}

export interface PathwayGeneValue {
  /** The symbol in the DE result (an alias when the primary one is not there). */
  gene: string;
  lfc: number;
  padj: number | null;
}

export interface PathwayNodeValue {
  /** Log fold change; a complex or set: the mean of its measured genes. Null: none measured. */
  value: number | null;
  n_genes: number;
  n_measured: number;
  /** Adjusted p < padj_cutoff; null when the result has no adjusted p-values. */
  n_changed: number | null;
}

export interface PathwayValues {
  st_id: string;
  requested: string;
  label: string;
  source: string;
  genes: Record<string, PathwayGeneValue>;
  nodes: Record<string, PathwayNodeValue>;
  padj_cutoff: number;
  has_padj: boolean;
  n_diagram_genes: number;
  n_measured: number;
  n_changed: number | null;
}

/** What colours a diagram: a stored DE result, or one group's markers against the rest. */
export type PathwayColourSource =
  | { kind: "de"; id: string }
  | { kind: "markers"; column: string; group: string };

export async function getReactomeDiagram(stId: string): Promise<PathwayDiagramModel> {
  return apiFetch<PathwayDiagramModel>(`/pathways/reactome/${encodeURIComponent(stId)}/diagram`);
}

export async function getReactomeValues(
  datasetId: string,
  stId: string,
  source: PathwayColourSource,
): Promise<PathwayValues> {
  const q = new URLSearchParams(
    source.kind === "de" ? { de_result: source.id } : { column: source.column, group: source.group },
  );
  return apiFetch<PathwayValues>(
    `/datasets/${datasetId}/pathways/reactome/${encodeURIComponent(stId)}/values?${q.toString()}`,
  );
}
