# scView

**A Docker-based, browser-native workspace for single-cell RNA-seq — with an AI co-pilot and built-in provenance.**

scView pairs interactive, linked visualizations (UMAP/t-SNE/PCA, gene expression, clusters,
markers, gene sets, pathway enrichment) with two things most single-cell tools lack: an **AI
assistant that interprets your data and coaches the next analysis step**, and **built-in
provenance** — "git for the h5ad" — so you can always answer *"what was done to this data, and
why?"* Wet-lab and dry-lab users go from a raw matrix to annotated, reproducible figures without
writing code.

<video src="https://github.com/thirtysix/scView/raw/main/docs/demo/demo.mp4" controls muted loop width="100%"></video>

▶️ **[Watch the demo](docs/demo/demo.mp4)** — *(the inline player appears once the repo is public)* — a
full run-through: load a dataset → AI-assisted assessment → Unified View → cluster colors →
marker-gene expression → 3D embedding → enrichment. The on-screen **real-time clock** shows true
elapsed time, so fast-forwarded sections visibly race ahead.

---

## Why scView

Single-cell analysis forces a split: code-first pipelines (Scanpy/Seurat) are powerful but
inaccessible; point-and-click viewers (cellxgene, Loupe) mostly *display precomputed results*.
Neither **interprets** the data for you nor **records** what was done in a way a collaborator or
reviewer can audit. scView closes both gaps:

- 🤖 **AI co-pilot (grounded & cited).** Ask about your data in plain language. Answers are grounded
  in your dataset's results + provenance **and** a scRNA-seq methods/literature corpus, with
  clickable citations (PubMed + jump-to-cluster). A cheap intent classifier routes each question so
  it only spends what it needs, and it's available even before a dataset is loaded.
- 🧭 **AI-assisted assessment.** A deterministic assessor reports the state of ~15 preprocessing
  steps; an LLM advisor recommends the next steps with reasons and sized parameters — *you* always
  approve before anything runs.
- 🧬 **Provenance — "git for the h5ad".** Every step is recorded into the data as a commit-style,
  replayable recipe with dependency-aware *edit & re-run from here*. Originals stay immutable.
- 📦 **Forgiving multi-format ingestion.** h5ad, 10x MEX/HDF5, loom, zarr, dense CSV, Seurat `.rds`,
  and nf-core/scrnaseq outputs — via a guided "Add Data" flow.
- 🖥️ **Unified, linked, server-backed exploration.** A Kana-style single screen: scatter + tabbed
  Markers/Expression/Gene Sets/Enrichment + violin, scaling to ~200k cells via a FastAPI backend,
  Apache Arrow, and deck.gl.

> Trust by design: the *facts* about your data are computed deterministically and reproducibly; the
> *LLM only advises*, and every action it suggests is approved by you and recorded in provenance.

---

## Screenshots

### Unified View
![Unified View](docs/images/unified-view.png)
*One linked screen: a UMAP colored by cell type, a camera-linked cluster reference map, a summary
card, and a sortable markers table — recoloring, violins, and cluster highlighting all linked.*

### AI co-pilot
![AI co-pilot](docs/images/ai-copilot.png)
*Ask about your data and get a grounded, cited answer — here the cell types and NK-cell markers,
with a "via" route badge, expandable grounding sources, and suggested follow-ups. Citation chips
link to PubMed or jump to the cluster in the app.*

### AI-assisted Data Assessment
![Data Assessment](docs/images/data-assessment.png)
*QC distribution plots plus a preprocessing step list. "Get AI Suggestions" returns recommended
next steps with reasoning and parameters — one click to apply, you decide whether to run.*

### History / provenance
![History](docs/images/history.png)
*Every recorded step as a timeline ("git for the h5ad"), with dependency-aware edit-&-re-run and an
exportable, replayable recipe.*

### Forgiving data import
![Data import](docs/images/data-import.png)
*Drop almost any format — 10x matrix/HDF5, h5ad, loom, CSV, Seurat `.rds`, nf-core outputs — and
reopen or manage previously loaded datasets.*

---

## Installation & usage

scView runs entirely in Docker, so the **same steps work on Linux, macOS, and Windows** — no Python,
Node, or R toolchains to install on the host.

### 1. Prerequisites

- **Docker** with the **Docker Compose v2** plugin:
  - **macOS / Windows** → install **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
    (bundles Compose). On **Windows**, accept the **WSL 2** backend when prompted.
  - **Linux** → install **Docker Engine** + the Compose plugin:
    ```bash
    # Ubuntu/Debian
    sudo apt install docker.io docker-compose-v2
    # Fedora/RHEL
    sudo dnf install docker docker-compose-plugin
    # Arch
    sudo pacman -S docker docker-compose
    # then enable the daemon and allow your user to run Docker:
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"     # log out/in afterward
    ```
- **~8 GB RAM** recommended, **~3 GB disk** for the images, and a modern browser.

Confirm Docker is ready:
```bash
docker --version && docker compose version && docker info
```

### 2. Get the code
```bash
git clone https://github.com/thirtysix/scView.git
cd scView
```

### 3. Start scView

**macOS / Linux** *(and Windows via WSL 2 or Git Bash)* — one guided command:
```bash
./start.sh           # dev stack, hot-reload  → http://localhost:5173  (opens your browser)
./start.sh --prod    # optimized Nginx stack  → http://localhost:3000
./start.sh --stop    # stop all services
./start.sh --clean   # stop and remove containers, volumes, and images
```
`start.sh` checks prerequisites, creates `.env`, builds the images (first run pulls a few GB and
takes a few minutes), waits for health checks, and opens the app.

**Windows (PowerShell, no WSL)** — run Compose directly:
```powershell
copy .env.example .env
docker compose -f docker-compose.dev.yml up --build    # dev  → http://localhost:5173
# …or the production stack:
docker compose up --build -d                            # prod → http://localhost:3000
```

**Any platform with `make`:**
```bash
make dev     # dev stack  → http://localhost:5173
make up      # prod stack → http://localhost:3000
make down    # stop
```

Once it's running:

| Service | URL |
| --- | --- |
| **App** (dev / prod) | http://localhost:5173 · http://localhost:3000 |
| **Backend API + interactive docs** | http://localhost:8080 · http://localhost:8080/docs |
| **R converter** (Seurat `.rds` → h5ad) | http://localhost:8001 |

### 4. First steps
1. Open the app and use **Add Data** (the **Data** tab).
2. Drop in a dataset — `.h5ad`, a 10x matrix (MEX or HDF5), `.loom`, `.zarr`, a dense `.csv`, a
   Seurat `.rds`, or an nf-core/scrnaseq output. *(No data handy? A small `pbmc_1k_v3` sample ships in
   `sample_data/`.)*
3. Open **Data Assessment** for QC plots + AI-suggested next steps, then explore in **Unified View**
   and ask the **AI Co-pilot**.

### 5. Optional — enable the AI features
Copy `.env.example` to `.env` and add a [DeepInfra](https://deepinfra.com/dash/api_keys) API key:
```bash
DEEPINFRA_API_KEY=your-key-here
```
Without a key the assistant degrades gracefully to deterministic, rule-based guidance. The RAG
co-pilot's literature/tutorial corpora additionally require a Postgres/pgvector connection string
(`RAG_DATABASE_URL`) — see [`docs/AI_ASSISTANT.md`](docs/AI_ASSISTANT.md).

---

## Architecture

Three Docker services:

- **backend** — FastAPI (`backend/src/scview`): AnnData I/O with lazy loading + LRU cache, an
  on-demand analysis pipeline (scanpy / harmonypy / gseapy / Scrublet), the AI assessor + advisor +
  RAG co-pilot, and Apache Arrow IPC serialization.
- **frontend** — React 18 + Vite + TypeScript + deck.gl, with Web Workers for Arrow decoding.
- **converter** — an R service (sceasy) that converts Seurat `.rds` → h5ad.

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
AI assistant design: [`docs/AI_ASSISTANT.md`](docs/AI_ASSISTANT.md).

## Development

```bash
make test          # backend test suite (pytest)
ruff check backend/src
( cd frontend && npm run build )   # strict type-check + production build
```

## Status

Active development (Phase 6: polish & testing). The AI co-pilot, dual-corpus RAG, provenance, and
multi-format ingestion are implemented; see [`docs/AI_ASSISTANT.md`](docs/AI_ASSISTANT.md) and
[`docs/FUTURE.md`](docs/FUTURE.md) for the roadmap.

## License

Released under the **[MIT License](LICENSE)** — © 2026 Harlan Barker.
