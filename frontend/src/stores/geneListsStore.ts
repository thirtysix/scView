import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";
import { parseGeneList } from "@/lib/geneList";
import {
  cleanListName,
  dedupeGenes,
  MAX_LISTS,
  parseGeneListFile,
  uniqueListName,
  type GeneListDraft,
} from "@/lib/geneLists";

/**
 * The user's named gene lists (the Unified View's Gene lists tab). Lists are
 * the user's, not a dataset's: they are kept in localStorage, so they survive
 * a reload, a dataset switch and `resetViewForDataset` (which never touches
 * them), and a session file carries them too (lib/sessionRestore.ts).
 */
export interface GeneList {
  id: string;
  name: string;
  /** Gene symbols as entered, each once ignoring case, in order */
  genes: string[];
  source?: string;
  /** ISO timestamps */
  created: string;
  updated: string;
}

export const GENE_LISTS_STORAGE_KEY = "scview.geneLists.v1";

interface GeneListsState {
  lists: GeneList[];
  /** The list the tab shows */
  selectedId: string | null;
  select: (id: string | null) => void;
  /** A new list (its name made unique); returns its id, or null when MAX_LISTS are kept */
  create: (name: string, genes?: string[], source?: string) => string | null;
  rename: (id: string, name: string) => void;
  duplicate: (id: string) => string | null;
  remove: (id: string) => void;
  /** Append genes not already in the list; returns how many were added */
  addGenes: (id: string, genes: string[]) => number;
  removeGene: (id: string, gene: string) => void;
  replaceGenes: (id: string, genes: string[]) => void;
  /** Pasted text as a new list (parseGeneList); null when it holds no genes */
  importText: (name: string, text: string, source?: string) => string | null;
  /** A .txt/.csv/.tsv/.gmt file's lists, one new list each; returns their ids. Throws on a file without genes. */
  importFile: (filename: string, text: string) => string[];
  /** Lists added as they are (names made unique); returns their ids */
  addLists: (drafts: GeneListDraft[]) => string[];
}

const now = () => new Date().toISOString();
let counter = 0;
/** No crypto.randomUUID: it is missing outside secure contexts (http on a LAN address). */
const newId = () => `gl-${Date.now().toString(36)}-${(counter++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

// ---------------------------------------------------------------------------
// Storage that never throws: a blocked or full localStorage leaves the lists
// working for this page, and the tab says they will not be kept.
// ---------------------------------------------------------------------------

interface StorageStatus {
  /** The last save failed (storage blocked or full) */
  failed: boolean;
}

export const useGeneListsStorageStatus = create<StorageStatus>(() => ({ failed: false }));

const setFailed = (failed: boolean) => {
  if (useGeneListsStorageStatus.getState().failed !== failed) useGeneListsStorageStatus.setState({ failed });
};

export const safeLocalStorage: StateStorage = {
  getItem: (key) => {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      window.localStorage.setItem(key, value);
      setFailed(false);
    } catch {
      setFailed(true);
    }
  },
  removeItem: (key) => {
    try {
      window.localStorage.removeItem(key);
    } catch {
      /* nothing to do */
    }
  },
};

type Stored = Pick<GeneListsState, "lists" | "selectedId">;

/** Stored lists checked field by field: anything malformed is dropped, never trusted. */
export function sanitizeStored(raw: unknown): Stored {
  const out: Stored = { lists: [], selectedId: null };
  if (typeof raw !== "object" || raw === null) return out;
  const r = raw as Record<string, unknown>;
  const ids = new Set<string>();
  for (const item of Array.isArray(r.lists) ? r.lists : []) {
    if (out.lists.length >= MAX_LISTS) break;
    if (typeof item !== "object" || item === null) continue;
    const l = item as Record<string, unknown>;
    if (typeof l.id !== "string" || !l.id || l.id.length > 64 || ids.has(l.id)) continue;
    const name = typeof l.name === "string" ? cleanListName(l.name) : "";
    if (!name || !Array.isArray(l.genes)) continue;
    ids.add(l.id);
    out.lists.push({
      id: l.id,
      name,
      genes: dedupeGenes(l.genes.filter((g): g is string => typeof g === "string")),
      ...(typeof l.source === "string" && l.source ? { source: l.source.slice(0, 300) } : {}),
      created: typeof l.created === "string" ? l.created : now(),
      updated: typeof l.updated === "string" ? l.updated : now(),
    });
  }
  out.selectedId = typeof r.selectedId === "string" && ids.has(r.selectedId) ? r.selectedId : null;
  return out;
}

export const useGeneListsStore = create<GeneListsState>()(
  persist(
    (set, get) => {
      const edit = (id: string, change: (l: GeneList) => Partial<GeneList>) =>
        set((s) => ({
          lists: s.lists.map((l) => (l.id === id ? { ...l, ...change(l), updated: now() } : l)),
        }));
      const addLists = (drafts: GeneListDraft[]): string[] => {
        const lists = [...get().lists];
        const ids: string[] = [];
        for (const d of drafts) {
          if (lists.length >= MAX_LISTS) break;
          const t = now();
          const list: GeneList = {
            id: newId(),
            name: uniqueListName(d.name, lists.map((l) => l.name)),
            genes: dedupeGenes(d.genes),
            ...(d.source ? { source: d.source } : {}),
            created: t,
            updated: t,
          };
          lists.push(list);
          ids.push(list.id);
        }
        if (ids.length > 0) set({ lists, selectedId: ids[0]! });
        return ids;
      };
      return {
        lists: [],
        selectedId: null,
        select: (id) => set({ selectedId: id }),
        create: (name, genes = [], source) => addLists([{ name, genes, source }])[0] ?? null,
        rename: (id, name) => {
          const others = get().lists.filter((l) => l.id !== id).map((l) => l.name);
          const clean = cleanListName(name);
          if (clean) edit(id, () => ({ name: uniqueListName(clean, others) }));
        },
        duplicate: (id) => {
          const l = get().lists.find((x) => x.id === id);
          return l ? (addLists([{ name: `${l.name} copy`, genes: l.genes, source: l.source }])[0] ?? null) : null;
        },
        remove: (id) =>
          set((s) => {
            const at = s.lists.findIndex((l) => l.id === id);
            const lists = s.lists.filter((l) => l.id !== id);
            // The neighbour takes the selection, so the tab never shows nothing while lists remain.
            const selectedId =
              s.selectedId === id ? (lists[Math.min(at, lists.length - 1)]?.id ?? null) : s.selectedId;
            return { lists, selectedId };
          }),
        addGenes: (id, genes) => {
          const l = get().lists.find((x) => x.id === id);
          if (!l) return 0;
          const next = dedupeGenes([...l.genes, ...genes]);
          if (next.length > l.genes.length) edit(id, () => ({ genes: next }));
          return next.length - l.genes.length;
        },
        removeGene: (id, gene) => edit(id, (l) => ({ genes: l.genes.filter((g) => g !== gene) })),
        replaceGenes: (id, genes) => edit(id, () => ({ genes: dedupeGenes(genes) })),
        importText: (name, text, source) => {
          const genes = dedupeGenes(parseGeneList(text));
          return genes.length > 0 ? (addLists([{ name, genes, source }])[0] ?? null) : null;
        },
        importFile: (filename, text) => addLists(parseGeneListFile(filename, text)),
        addLists,
      };
    },
    {
      name: GENE_LISTS_STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => safeLocalStorage),
      partialize: (s): Stored => ({ lists: s.lists, selectedId: s.selectedId }),
      merge: (persisted, current) => ({ ...current, ...sanitizeStored(persisted) }),
    },
  ),
);
