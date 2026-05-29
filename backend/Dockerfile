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

# Dev target: source mounted as volume, uvicorn with reload
FROM base AS dev
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "scview.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production target
FROM base AS production
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "scview.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
