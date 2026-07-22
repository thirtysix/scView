# AGENTS.md

Orientation for AI coding agents (and humans) arriving at this repo: what it is,
how to run it, and the conventions worth knowing before making changes.

## What this is

scView: a Docker-based, browser-native visualization tool for single-cell RNA-seq
data. Interactive exploration of embeddings (UMAP/tSNE/PCA), gene expression, cluster
annotation, marker genes, gene sets, pathway enrichment, and trajectory analysis,
served through a FastAPI backend with an AI-assisted data-assessment panel.

## Stack & layout

- **Backend** — FastAPI (Python), entry `backend/src/scview/main.py`, port 8080
- **Frontend** — React 18 + Vite 6 (TypeScript), entry `frontend/src/main.tsx`
- **Converter** — R service, entry `converter/server.R`
- Core backend logic lives in `backend/src/scview/core/` (AnnData I/O, Arrow
  serialization, the QC→normalization→embedding→clustering→enrichment pipeline).

## Run, test, lint

```bash
bash start.sh            # or: make up   — brings up the Docker stack
pytest tests/            # backend tests   (run from backend/)
ruff check src/          # backend lint    (run from backend/)
npm run lint             # frontend lint   (run from frontend/)
```

Prefer `docker compose` for anything touching the full stack; the services expect
each other on their compose network.

## Conventions

- Backend serializes numpy/pandas to Apache Arrow IPC for the frontend; do not add a
  parallel JSON path for large arrays.
- Frontend state is Zustand (`frontend/src/stores/`); keep view state there, not in
  component-local state, for anything the scatter/violin/tabs share.
- The main scatter is deck.gl with a controlled camera (`externalViewState`); the
  cluster mini-map is camera-linked to it. Keep them in sync when touching either.

## Gotchas

- The full-page expand toggle uses a body portal without remount, deliberately, so
  deck.gl/Plotly contexts survive; do not "simplify" it to a conditional render.
- The AI assessment panel calls an external LLM (DeepInfra); it must fail soft when
  the key is absent, so the app still runs without it.
