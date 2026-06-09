# Contributing to scView

Thanks for your interest in improving scView! Bug reports, feature ideas, and pull requests are all
welcome.

## Reporting bugs and requesting features

Please open an [issue](https://github.com/thirtysix/scView/issues) using the bug-report or
feature-request template. For bugs, include your OS, how you started scView (`./start.sh`, `make`,
or `docker compose`), the dataset format, and the relevant backend logs
(`docker compose logs backend`).

## Development setup

scView runs in Docker; see [the README](README.md) for prerequisites. For a hot-reloading dev stack:

```bash
git clone https://github.com/thirtysix/scView.git
cd scView
make dev            # frontend on http://localhost:5173, backend on http://localhost:8080
```

The backend source is bind-mounted, so Python changes reload automatically. Adding a Python or npm
dependency requires rebuilding the relevant image (`docker compose build backend` / `frontend`).

## Before opening a pull request

Run the checks locally:

```bash
make test                          # backend test suite (pytest)
docker compose exec backend ruff check src/   # backend lint
( cd frontend && npm run build )   # strict type-check + production build (catches errors `tsc --noEmit` misses)
```

- Add or update tests for any behavior change (the backend suite is the safety net).
- Match the style and structure of the surrounding code.
- Keep commits focused, with clear messages.
- For larger changes, open an issue first to discuss the approach.

## Project layout

- `backend/` — FastAPI service (AnnData I/O, the analysis pipeline, the AI assessor/advisor/RAG
  co-pilot, Arrow serialization). Tests live in `backend/tests/`.
- `frontend/` — React + Vite + TypeScript + deck.gl.
- `converter/` — R service that converts Seurat `.rds` to h5ad.
- `docs/` — architecture, AI-assistant design, provenance, ingestion, and roadmap notes.

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
