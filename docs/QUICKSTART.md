# scView Quick Start Guide

## Prerequisites

- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- At least **8 GB RAM** available for Docker
- A modern browser (Chrome, Firefox, Edge)

## 1. Clone and Configure

```bash
git clone <repository-url> scView
cd scView

# Copy the example environment file
cp .env.example .env
```

Edit `.env` if needed:

```dotenv
# Directory where uploaded datasets are stored (inside the container)
DATA_DIR=/data

# Maximum upload size in MB
MAX_UPLOAD_SIZE_MB=2048

# CORS origins (comma-separated; default allows localhost)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Optional: DeepInfra API key for AI-powered analysis suggestions
# Get one at https://deepinfra.com — the app works fine without it
DEEPINFRA_API_KEY=

# R converter service URL (auto-configured in Docker)
CONVERTER_URL=http://converter:8001
```

## 2. Start the Application

### Development Mode (with hot-reload)

```bash
make dev
```

This starts all three services with live reloading:
- **Frontend**: http://localhost:5173 (Vite dev server)
- **Backend API**: http://localhost:8000
- **R Converter**: http://localhost:8001

### Production Mode

```bash
make build   # Build optimized images
make up      # Start services
```

Production serves the frontend via Nginx at http://localhost:3000 with API proxying built in.

## 3. Upload Data

1. Open the app in your browser
2. The **Load Data** panel appears by default
3. Drag and drop a file onto the upload zone, or click to browse:
   - `.h5ad` — Scanpy/AnnData format (loaded directly)
   - `.rds` / `.rdata` — Seurat format (auto-converted to h5ad via the R converter)
4. Wait for processing to complete:
   - h5ad files: usually seconds
   - Seurat files: 30s–2min depending on size (progress shown in UI)

## 4. Explore Your Data

Once a dataset is loaded, the sidebar panels become active:

| Panel | What It Does |
|-------|-------------|
| **Load Data** | Upload and manage datasets |
| **Data Assessment** | Auto-detect preprocessing state; run missing steps; get AI suggestions |
| **Overview** | WebGL scatter plot (UMAP/tSNE/PCA); color by any metadata; lasso selection |
| **Gene Expression** | Search genes, overlay expression on scatter, violin plots |
| **Samples** | Sample composition analysis with stacked bar charts |
| **Clusters** | Cluster composition, click-to-highlight on scatter |
| **Marker Genes** | Sortable/filterable marker gene table with CSV export |
| **Gene Sets** | Score custom gene sets, visualize on scatter |
| **Enrichment** | Pathway enrichment analysis with bar charts |
| **Trajectory** | Pseudotime visualization and gene expression along trajectory |

## 5. Typical Workflow

### For preprocessed data (.h5ad with embeddings)

1. Upload your `.h5ad` file
2. Go to **Overview** — your UMAP/tSNE appears immediately
3. Use the color-by dropdown to explore metadata (clusters, cell types, samples)
4. Switch to **Gene Expression** to query specific genes
5. Check **Marker Genes** for differential expression results
6. Export results via the export menu in each panel

### For raw/unprocessed data

1. Upload your `.h5ad` or `.rds` file
2. Go to **Data Assessment** — the panel shows a checklist of preprocessing steps
3. Green = done, Yellow = partially done, Gray = missing
4. Click **"Run Missing Steps"** to execute the full scanpy pipeline with sensible defaults
5. Or expand individual steps to customize parameters before running
6. Optionally click **"Get AI Suggestions"** (requires DeepInfra API key) for parameter recommendations
7. Once preprocessing completes, all visualization panels are populated

### For Seurat data

1. Upload your `.rds` file — the R converter automatically:
   - Detects Seurat version (v3, v4, or v5)
   - Extracts all assays, reductions, and metadata
   - Converts to `.h5ad` format
2. Once conversion finishes, the dataset loads like any h5ad file

## 6. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `L` | Toggle lasso selection mode (Overview panel) |
| `Escape` | Clear current selection |

## 7. Useful Commands

```bash
# View logs from all services
make logs

# View logs from a specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f converter

# Stop all services
make down

# Clean up containers and volumes
make clean

# Run backend tests
make test
```

## 8. Troubleshooting

### "Cannot connect to backend"
- Check that all services are running: `docker compose ps`
- Verify the backend is healthy: `curl http://localhost:8000/health`

### Upload fails with large files
- Increase `MAX_UPLOAD_SIZE_MB` in `.env`
- Ensure Docker has enough disk space for the data volume

### Seurat conversion fails
- Check converter logs: `docker compose logs converter`
- Ensure the .rds file is a valid Seurat v3/v4/v5 object
- The converter supports objects created with Seurat 3.x, 4.x, and 5.x

### Scatter plot is slow or blank
- Ensure your browser supports WebGL 2.0
- Try reducing point size in Plot Controls
- For datasets >200k cells, performance may vary by GPU

### AI suggestions unavailable
- Set `DEEPINFRA_API_KEY` in `.env` and restart: `make down && make dev`
- The app works fully without AI — rule-based suggestions are always available
