FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (for scientific Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata & source first (for dependency install)
# If you have a README.md referenced in pyproject, copy it too.
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies (project + dev extras)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

# Copy the rest of the project (optional / non-package stuff)
COPY scripts ./scripts
COPY models ./models
COPY data ./data

# Default command: run tests
CMD ["pytest", "-q"]
