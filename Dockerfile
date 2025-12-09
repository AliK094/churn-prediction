FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (for scientific Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions
COPY requirements.txt requirements-dev.txt pyproject.toml ./ 

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir .

# Copy project code and artifacts
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY data ./data

# Default command: run tests
CMD ["pytest", "-q"]

