/**
 * The workspace as a session (lib/session.ts) and back: `captureSession` reads
 * the stores, `restoreSession` writes them. "Save session" / "Open session"
 * use both; the reload restore (hooks/useSessionRestore.ts) restores its
 * localStorage slice through `restoreSession` too.
 */
import { apiFetch } from "@/api/client";
import { cellsMatching } from "@/api/cells";
import { getDataset } from "@/api/datasets";
import type { DatasetInfo } from "@/api/types";
import { describeFilter } from "@/lib/cellFilter";
import { fromSessionGate, gateDependencies, toSessionGate, validateGates, type Gate } from "@/lib/gates";
import { formatNumber, prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import {
  decodeSelection,
  encodeSelection,
  SESSION_FORMAT,
  SESSION_VERSION,
  type SessionDataset,
  type SessionFile,
  type SessionView,
} from "@/lib/session";
import { COMPARE_MIN_COLUMNS, compareColumnOptions, splitColumnOptions } from "@/lib/splitView";
import { resetViewForDataset, useDatasetStore } from "@/stores/datasetStore";
import { useCommunicationStore } from "@/stores/communicationStore";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { staleReason, useGatesStore } from "@/stores/gatesStore";
import {
  cleanListName,
  dedupeGenes,
  MAX_LISTS,
  sameGenes,
  uniqueListName,
  type GeneListDraft,
} from "@/lib/geneLists";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

/** The workspace now, as a session file. */
export function captureSession(now: Date = new Date()): SessionFile {
  const dataset = useDatasetStore.getState().currentDataset;
  const settings = useSettingsStore.getState();
  const unified = useUnifiedViewStore.getState();
  const split = useSplitViewStore.getState();
  const selection = useSelectionStore.getState();
  const de = useDESettingsStore.getState();
  const comm = useCommunicationStore.getState();

  const view: SessionView = {
    panel: useViewStore.getState().activePanel,
    subtab: unified.activeSubtab,
    embedding: settings.embedding,
    color_by: settings.colorBy,
    color_by_chosen: settings.colorByChosen,
    expression_layer: settings.expressionLayer,
    group_by: unified.groupByColumn,
    split_by: split.splitBy,
    show_context: split.showContext,
    compare_columns: split.compareColumns,
    labels: unified.centroidLabels,
    camera: null,
    highlight: selection.highlightedGroup,
    overlay: null,
    color_clip: unified.colorClip,
  };
  const cam = unified.camera;
  if (dataset && cam && cam.datasetId === dataset.id && cam.embedding === settings.embedding) {
    view.camera = cam.camera;
  }
  const overlay = unified.scatterOverlay;
  if (dataset && overlay && unified.overlayDatasetId === dataset.id) {
    if (overlay.type === "expression") {
      view.overlay = overlay.layer
        ? { type: "expression", gene: overlay.gene, layer: overlay.layer }
        : { type: "expression", gene: overlay.gene };
    }
    else if (overlay.type === "score" && overlay.activity) {
      view.overlay = { type: "activity", ...overlay.activity };
    } else if (overlay.type === "score" && overlay.genes?.length) {
      view.overlay = { type: "score", name: overlay.name, genes: overlay.genes };
    }
  }

  const cells = selection.selectedCellIndices;
  const nCells = dataset?.n_cells ?? 0;
  // The dataset's gates, unless they are stale (drawn on other cells).
  const gateRecord = dataset ? useGatesStore.getState().records[dataset.id] : undefined;
  const gates = dataset && gateRecord && !staleReason(gateRecord, dataset) ? gateRecord.gates : [];
  return {
    format: SESSION_FORMAT,
    version: SESSION_VERSION,
    saved_at: now.toISOString(),
    ...(dataset
      ? {
          dataset: {
            id: dataset.id,
            name: dataset.name,
            n_cells: dataset.n_cells,
            cells_fingerprint: dataset.cells_fingerprint ?? null,
          },
        }
      : {}),
    view,
    selection: selection.selectionFilter
      ? { filter: selection.selectionFilter }
      : cells && cells.size > 0 && nCells > 0
        ? encodeSelection(cells, nCells)
        : null,
    de: {
      mode: de.mode,
      split_by: de.splitBy,
      group_a: de.groupA,
      group_b: de.groupB,
      within_column: de.withinCol,
      within_value: de.withinVal,
      test: de.test,
      replicate_key: de.replicateKey,
      fc_threshold: de.fcThreshold,
      min_pct: de.minPct,
      cell_method: de.cellMethod,
    },
    colors: {
      categorical_palette: settings.categoricalPalette,
      continuous_colormap: settings.continuousColormap,
      diverging_colormap: settings.divergingColormap,
      signed_colormap_mode: settings.signedColormapMode,
      clip: settings.colorClip,
      point_size: settings.pointSize,
      opacity: settings.opacity,
      background: settings.plotBackground,
      max_rendered_cells: settings.maxRenderedCells,
    },
    communication: {
      groupby: comm.groupby,
      subset_column: comm.subsetColumn,
      subset_value: comm.subsetValue,
      expr_prop: comm.exprProp,
    },
    gene_lists: useGeneListsStore
      .getState()
      .lists.map((l) => (l.source ? { name: l.name, genes: l.genes, source: l.source } : { name: l.name, genes: l.genes })),
    ...(gates.length > 0 ? { gates: gates.map(toSessionGate) } : {}),
  };
}

export interface RestoreReport {
  /** What was opened, or why the dataset could not be */
  headline: string;
  /** What was not restored, and why */
  notes: string[];
  level: "ok" | "warning";
}

const quote = (s: string) => `“${s}”`;

/** The saved dataset's cells differ from `d`'s (when the session says what they were). */
function cellsChanged(saved: SessionDataset, d: DatasetInfo): string | null {
  if (saved.n_cells != null && d.n_cells != null && saved.n_cells !== d.n_cells) {
    return `${saved.n_cells.toLocaleString()} cells then, ${d.n_cells.toLocaleString()} now`;
  }
  if (saved.cells_fingerprint && d.cells_fingerprint && saved.cells_fingerprint !== d.cells_fingerprint) {
    return "the same number of cells, but not the same cells";
  }
  return null;
}

async function geneInDataset(datasetId: string, gene: string): Promise<boolean> {
  try {
    const res = await apiFetch<{ results: string[] }>(
      `/datasets/${datasetId}/genes/search?q=${encodeURIComponent(gene)}&limit=100`,
    );
    return (res.results ?? []).includes(gene);
  } catch {
    return true; // can't tell: let the overlay try
  }
}

/**
 * Put a saved session back. The dataset is opened by id; when it is missing,
 * the session goes onto the open dataset if that has the same cells, or as far
 * as it fits otherwise. Cell indices and the camera are used only on the same
 * cells; anything naming a column, gene or embedding the dataset lacks is
 * skipped and reported.
 */
export async function restoreSession(session: SessionFile): Promise<RestoreReport> {
  const notes: string[] = [];
  const saved = session.dataset;
  const current = useDatasetStore.getState().currentDataset;
  let target: DatasetInfo | null = null;
  let sameCells = false;
  let headline = "";
  let level: RestoreReport["level"] = "ok";

  if (saved) {
    const name = quote(saved.name || saved.id);
    let found: DatasetInfo | null = null;
    try {
      found = await getDataset(saved.id);
    } catch (e) {
      // Only "no such dataset" falls back; a busy or unreachable server is an error.
      const message = e instanceof Error ? e.message : String(e);
      if (!/^API error 404\b/.test(message)) {
        throw new Error(`The session's dataset ${name} could not be opened (${message.replace(/^API error /, "")}).`);
      }
    }
    if (found) {
      target = found;
      const change = cellsChanged(saved, found);
      sameCells = change == null;
      headline = `Opened ${quote(found.name)}.`;
      if (change) {
        level = "warning";
        headline = `Opened ${quote(found.name)}, but its cells have changed since the session was saved (${change}).`;
      }
    } else if (current && saved.cells_fingerprint && current.cells_fingerprint === saved.cells_fingerprint) {
      target = current;
      sameCells = true;
      headline = `The session's dataset ${name} is not in this library; the open dataset ${quote(current.name)} has the same cells, so the session was restored on it.`;
    } else if (current) {
      target = current;
      level = "warning";
      headline = `The session's dataset ${name} is not in this library. What fits the open dataset ${quote(current.name)} was restored.`;
    } else {
      level = "warning";
      headline = `The session's dataset ${name} is not in this library, so only its colour settings were restored.`;
    }
  } else if (current) {
    target = current;
    sameCells = true;
    headline = `Restored on ${quote(current.name)}.`;
  } else {
    headline = "Restored the colour settings.";
  }

  applyColors(session);
  // The user's lists, not the dataset's: merged whatever dataset opens.
  applyGeneLists(session, notes);
  if (!target) {
    // Without a dataset only a panel that needs none means anything; a
    // missing dataset leaves the user where they are.
    if (!saved && session.view?.panel) useViewStore.getState().setPanel(session.view.panel);
    return { headline, notes, level };
  }

  // Start from the dataset's own defaults, as when it is opened.
  const datasets = useDatasetStore.getState();
  if (datasets.currentDatasetId === target.id) {
    datasets.setCurrentDataset(target);
    resetViewForDataset(target);
  } else {
    datasets.setCurrentDataset(target);
  }

  await applyView(session, target, sameCells, notes);
  applyDE(session, target, notes);
  applyCommunication(session, target, notes);
  await applySelection(session, target, sameCells, notes);
  applyGates(session, target, sameCells, notes);
  if (session.view?.panel) useViewStore.getState().setPanel(session.view.panel);
  if (notes.length > 0) level = "warning";
  return { headline, notes, level };
}

/**
 * The session's gene lists merged into the user's, never replacing one: a list
 * already here under the same name with the same genes (or restored before as
 * "name (session)") is skipped; one whose name is taken by other genes comes in
 * as "name (session)".
 */
export function applyGeneLists(session: SessionFile, notes: string[]) {
  const saved = session.gene_lists;
  if (!saved || saved.length === 0) return;
  const store = useGeneListsStore.getState();
  const taken = store.lists.map((l) => l.name);
  const fromSession = (n: number) => (n === 2 ? " (session)" : ` (session ${n - 1})`);
  const drafts: GeneListDraft[] = [];
  const renamed: string[] = [];
  for (const l of saved) {
    const genes = dedupeGenes(l.genes);
    const name = cleanListName(l.name) || "Gene list";
    const lower = name.toLowerCase();
    const already = [...store.lists, ...drafts].some((x) => {
      const n = x.name.toLowerCase();
      return (n === lower || n.startsWith(`${lower} (session`)) && sameGenes(x.genes, genes);
    });
    if (already) continue;
    const unique = uniqueListName(name, taken, fromSession);
    if (unique !== name) renamed.push(`${quote(name)} came in as ${quote(unique)}`);
    taken.push(unique);
    drafts.push({ name: unique, genes, ...(l.source ? { source: l.source } : {}) });
  }
  if (drafts.length === 0) return;
  const before = store.selectedId;
  const ids = store.addLists(drafts);
  // Adding lists does not move the Gene lists tab off the list it shows.
  if (before) useGeneListsStore.getState().select(before);
  if (ids.length < drafts.length) {
    notes.push(`Gene lists: ${drafts.length - ids.length} not added, at most ${MAX_LISTS} lists are kept.`);
  }
  if (renamed.length > 0) {
    notes.push(`Gene lists: ${renamed.join(", ")}, since a list of that name with other genes was already here.`);
  }
}

function applyColors(session: SessionFile) {
  const c = session.colors;
  if (!c) return;
  const s = useSettingsStore.getState();
  if (c.categorical_palette) s.setCategoricalPalette(c.categorical_palette);
  if (c.continuous_colormap) s.setContinuousColormap(c.continuous_colormap);
  if (c.diverging_colormap) s.setDivergingColormap(c.diverging_colormap);
  if (c.signed_colormap_mode) s.setSignedColormapMode(c.signed_colormap_mode);
  if (c.clip) s.setColorClip(c.clip);
  if (c.point_size != null) s.setPointSize(Math.min(10, Math.max(1, c.point_size)));
  if (c.opacity != null) s.setOpacity(Math.min(1, Math.max(0.1, c.opacity)));
  if (c.background) s.setPlotBackground(c.background);
  if (c.max_rendered_cells != null) s.setMaxRenderedCells(c.max_rendered_cells);
}

async function applyView(session: SessionFile, d: DatasetInfo, sameCells: boolean, notes: string[]) {
  const v = session.view;
  if (!v) return;
  const columns = new Map((d.obs_columns ?? []).map((c) => [c.name, c]));
  const missing = (what: string, name: string) =>
    notes.push(`${what} ${quote(prettyObsLabel(name))}: this dataset has no such column.`);
  const settings = useSettingsStore.getState();
  const unified = useUnifiedViewStore.getState();
  const split = useSplitViewStore.getState();

  if (v.embedding) {
    if (d.available_embeddings.includes(v.embedding)) settings.setEmbedding(v.embedding);
    else notes.push(`Embedding ${prettyEmbeddingLabel(v.embedding)}: not in this dataset.`);
  }
  if (v.expression_layer) {
    if ((d.expression_layers ?? []).some((l) => l.key === v.expression_layer)) {
      settings.setExpressionLayer(v.expression_layer);
    } else {
      notes.push(`Expression layer ${quote(v.expression_layer)}: not in this dataset; the default is used.`);
    }
  }
  if (v.color_by !== undefined) {
    const chosen = v.color_by_chosen !== false;
    if (v.color_by === "" || columns.has(v.color_by)) {
      if (chosen) settings.setColorBy(v.color_by);
      else settings.setDefaultColorBy(v.color_by);
    } else {
      missing("Colour by", v.color_by);
    }
  }
  if (v.group_by) {
    if (columns.has(v.group_by)) unified.setGroupByColumn(v.group_by);
    else missing("Grouping", v.group_by);
  }
  if (v.split_by !== undefined) {
    const ok = v.split_by == null || splitColumnOptions(d.obs_columns ?? []).some((o) => o.name === v.split_by && !o.disabledReason);
    if (ok) split.setSplitBy(v.split_by ?? null);
    else if (v.split_by) notes.push(`Split by ${quote(prettyObsLabel(v.split_by))}: this dataset can't be split by it.`);
  }
  if (v.show_context !== undefined) split.setShowContext(v.show_context);
  if (v.compare_columns !== undefined) {
    if (v.compare_columns == null) split.setCompareColumns(null);
    else {
      const options = new Set(compareColumnOptions(d.obs_columns ?? [], d.n_cells, d.active_clustering).map((o) => o.name));
      const kept = v.compare_columns.filter((c) => options.has(c));
      const lost = v.compare_columns.filter((c) => !options.has(c));
      if (kept.length >= COMPARE_MIN_COLUMNS) split.setCompareColumns(kept);
      if (lost.length > 0) {
        notes.push(
          `Compare colourings: ${lost.map((c) => quote(prettyObsLabel(c))).join(", ")} not in this dataset` +
            (kept.length < COMPARE_MIN_COLUMNS ? ", so the comparison is off." : "."),
        );
      }
    }
  }
  if (v.labels !== undefined) unified.setCentroidLabels(v.labels);
  if (v.subtab) unified.setActiveSubtab(v.subtab);
  if (v.color_clip !== undefined) unified.setColorClip(v.color_clip);

  if (v.highlight) {
    const col = columns.get(v.highlight.column);
    if (col && (col.values ?? []).includes(v.highlight.value)) {
      useSelectionStore.getState().setHighlight(v.highlight);
    } else {
      notes.push(`Highlighted group ${quote(v.highlight.value)} of ${quote(prettyObsLabel(v.highlight.column))}: not in this dataset.`);
    }
  }

  if (v.overlay) {
    const o = v.overlay;
    if (o.type === "expression" && o.layer && !(d.protein?.names ?? []).includes(o.gene)) {
      notes.push(`Protein overlay ${o.gene}: the protein is not in this dataset.`);
    } else if (o.type === "expression" && !o.layer && !(await geneInDataset(d.id, o.gene))) {
      notes.push(`Gene overlay ${o.gene}: the gene is not in this dataset.`);
    } else {
      unified.setPendingOverlay({ datasetId: d.id, overlay: o });
    }
  }

  if (v.camera) {
    const embedding = useSettingsStore.getState().embedding;
    if (!sameCells) notes.push("Camera: not restored, the cells differ.");
    else if (v.embedding && embedding !== v.embedding) notes.push("Camera: not restored, the embedding differs.");
    else unified.setPendingCamera({ datasetId: d.id, embedding, camera: v.camera });
  }
}

function applyDE(session: SessionFile, d: DatasetInfo, notes: string[]) {
  const de = session.de;
  if (!de) return;
  const columns = new Map((d.obs_columns ?? []).map((c) => [c.name, c]));
  const has = (col?: string, value?: string) =>
    !!col && columns.has(col) && (value === undefined || (columns.get(col)!.values ?? []).includes(value));
  const store = useDESettingsStore.getState();
  store.update({
    ...(de.mode ? { mode: de.mode } : {}),
    ...(de.test ? { test: de.test } : {}),
    ...(de.fc_threshold != null ? { fcThreshold: de.fc_threshold } : {}),
    ...(de.min_pct != null ? { minPct: de.min_pct } : {}),
    ...(de.cell_method ? { cellMethod: de.cell_method } : {}),
  });
  if (de.split_by) {
    if (has(de.split_by)) {
      store.update({ splitBy: de.split_by });
      if (has(de.split_by, de.group_a ?? "") && has(de.split_by, de.group_b ?? "")) {
        store.update({ groupA: de.group_a, groupB: de.group_b });
      }
    } else {
      notes.push(`DE between groups: ${quote(prettyObsLabel(de.split_by))} is not in this dataset.`);
    }
  }
  if (de.within_column) {
    if (has(de.within_column, de.within_value ?? "")) {
      store.update({ withinCol: de.within_column, withinVal: de.within_value });
    } else {
      notes.push(`DE scope ${quote(`${prettyObsLabel(de.within_column)} = ${de.within_value ?? ""}`)}: not in this dataset.`);
    }
  }
  if (de.replicate_key && has(de.replicate_key)) store.update({ replicateKey: de.replicate_key });
  // The restored highlight is already the scope it should be.
  store.update({ scopedHighlight: useSelectionStore.getState().highlightedGroup });
}

/** The Communication tab's settings, where the dataset has the columns; never the result. */
function applyCommunication(session: SessionFile, d: DatasetInfo, notes: string[]) {
  const c = session.communication;
  if (!c) return;
  const store = useCommunicationStore.getState();
  const columns = new Map((d.obs_columns ?? []).map((col) => [col.name, col]));
  store.update({
    groupby: "",
    subsetColumn: "",
    subsetValue: "",
    ...(c.expr_prop != null ? { exprProp: c.expr_prop } : {}),
  });
  if (c.groupby) {
    if (columns.has(c.groupby)) store.update({ groupby: c.groupby });
    else notes.push(`Communication: ${quote(prettyObsLabel(c.groupby))} is not in this dataset.`);
  }
  if (c.subset_column) {
    if ((columns.get(c.subset_column)?.values ?? []).includes(c.subset_value ?? "")) {
      store.update({ subsetColumn: c.subset_column, subsetValue: c.subset_value });
    } else {
      notes.push(
        `Communication cells ${quote(`${prettyObsLabel(c.subset_column)} = ${c.subset_value ?? ""}`)}: not in this dataset.`,
      );
    }
  }
}

async function applySelection(session: SessionFile, d: DatasetInfo, sameCells: boolean, notes: string[]) {
  const sel = session.selection;
  if (!sel) return;
  const store = useSelectionStore.getState();
  if ("filter" in sel) {
    const sentence = describeFilter(sel.filter);
    try {
      const cells = await cellsMatching(d.id, sel.filter);
      if (cells.size > 0) store.setSelection(cells, sel.filter);
      else notes.push(`Selection ${quote(sentence)}: no cells match in this dataset.`);
    } catch (e) {
      notes.push(`Selection ${quote(sentence)}: ${e instanceof Error ? e.message : String(e)}`);
    }
    return;
  }
  const nSaved = "indices" in sel ? sel.indices.length : null;
  if (!sameCells || (d.n_cells != null && sel.n_cells !== d.n_cells)) {
    notes.push(
      `Selection${nSaved != null ? ` of ${formatNumber(nSaved)} cells` : ""}: not restored, it lists cells by position and the cells differ.`,
    );
    return;
  }
  const { cells, ignored } = decodeSelection(sel, d.n_cells ?? 0);
  if (ignored > 0) notes.push(`Selection: ${formatNumber(ignored)} cell positions beyond the dataset's cells were ignored.`);
  if (cells.size > 0) store.setSelection(cells);
}

const gateNames = (gates: { name: string }[]) => gates.map((g) => quote(g.name)).join(", ");

/**
 * The session's gates merged into the dataset's (a gate already here with the
 * same id takes the session's version; others stay). Gates saved by cell
 * position, and gates inside or built from them, need the same cells; on
 * other cells only filter and combined gates come back.
 */
export function applyGates(session: SessionFile, d: DatasetInfo, sameCells: boolean, notes: string[]) {
  const saved = session.gates;
  if (!saved || saved.length === 0) return;
  const { gates: valid, dropped } = validateGates(saved.map(fromSessionGate));
  let gates: Gate[] = valid;
  if (!sameCells) {
    const kept = new Set<string>();
    for (let changed = true; changed; ) {
      changed = false;
      for (const g of gates) {
        if (kept.has(g.id) || g.def.kind === "cells" || !gateDependencies(g).every((id) => kept.has(id))) continue;
        kept.add(g.id);
        changed = true;
      }
    }
    const lost = gates.filter((g) => !kept.has(g.id));
    gates = gates.filter((g) => kept.has(g.id));
    if (lost.length > 0) {
      notes.push(
        `Gates ${gateNames(lost)}: not restored, the cells differ and ${lost.length === 1 ? "it was" : "they were"} saved by cell position (or inside or built from such a gate).`,
      );
    }
  }
  if (gates.length > 0) {
    const result = useGatesStore.getState().mergeGates(d, gates);
    dropped.push(...result.dropped);
    if (result.replacedStale > 0) {
      notes.push(`Gates: ${result.replacedStale} gate${result.replacedStale === 1 ? "" : "s"} saved here on other cells replaced by the session's.`);
    }
  }
  if (dropped.length > 0) {
    notes.push(`Gates not restored: ${dropped.map((x) => `${quote(x.name)} (${x.reason})`).join(", ")}.`);
  }
}
