# Earth2Studio CAMS Export — lightweight container for data pipeline
# No GPU needed for CAMS data fetch + EXR export
FROM python:3.12-slim

# System deps for eccodes (GRIB/netCDF) and image export
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes-tools \
    libeccodes-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

WORKDIR /app

# Copy project
COPY pyproject.toml uv.lock* ./
COPY earth2studio/ ./earth2studio/
COPY scripts/ ./scripts/

# Install earth2studio with data dependencies (no GPU extras needed)
RUN uv venv && uv sync --extra data --no-dev

# Cache and output dirs
RUN mkdir -p /data/cache /data/outputs
ENV EARTH2STUDIO_CACHE=/data/cache

ENTRYPOINT ["uv", "run", "python"]
CMD ["scripts/export_exr_sequence.py", "--days", "3"]
