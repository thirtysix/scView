FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

COPY src/ src/

# The container runs as root, so bytecode written into a bind-mounted src/ or
# tests/ lands root-owned in the host checkout. Set here, not only in compose,
# so a one-off `docker run` of this image (e.g. to run pytest) is covered too.
ENV PYTHONDONTWRITEBYTECODE=1

# Two glibc malloc arenas instead of 8 per CPU: each request's thread kept its own arena of freed
# memory. Measured v0.78 (2 CPUs, 39,934 x 24,551 file): memory held after 3 pages open Data
# Assessment at once 3.98 -> 3.50 GiB (anon 3.89 -> 3.30), peak 4.40 -> 4.22; Kang run as fast.
ENV MALLOC_ARENA_MAX=2

# Dev target: source mounted as volume, uvicorn with reload
FROM base AS dev
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "scview.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production target
FROM base AS production
ENV PYTHONPATH=/app/src
EXPOSE 8000
# One worker: datasets are cached in process memory, so a second worker keeps
# serving its own stale copy after the pipeline rewrites a dataset.
CMD ["uvicorn", "scview.main:app", "--host", "0.0.0.0", "--port", "8000"]
