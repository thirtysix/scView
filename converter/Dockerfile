FROM rocker/r-ver:4.4.0

LABEL maintainer="scView project"
LABEL description="R-based converter service for Seurat to h5ad conversion"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpng-dev \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python anndata + scipy (required by reticulate manual conversion)
RUN pip3 install --no-cache-dir --break-system-packages anndata scipy || \
    pip3 install --no-cache-dir anndata scipy

# Tell reticulate to use the system Python (not download its own via uv)
ENV RETICULATE_PYTHON=/usr/bin/python3

# Set up working directory
WORKDIR /app

# Copy and run R package installation first (for Docker layer caching)
COPY install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R

# Copy remaining R source files
COPY server.R /app/server.R
COPY convert.R /app/convert.R

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["Rscript", "server.R"]
