# Freak AI Recommendation System
# Multi-stage Dockerfile for production deployment

# ============================================
# Stage 1: Base image with dependencies
# ============================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash appuser

# Set work directory
WORKDIR /app

# ============================================
# Stage 2: Builder with all dependencies
# ============================================
FROM base as builder

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ============================================
# Stage 3: Production image
# ============================================
FROM base as production

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser . /app

# Create directories for data and models
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embeddings \
    /app/checkpoints /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
CMD ["python", "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 4: Development image
# ============================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --user pytest pytest-asyncio pytest-cov black ruff mypy ipython jupyter

USER appuser

# Override command for development
CMD ["bash"]

# ============================================
# Stage 5: Training image (GPU support)
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as training

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Create app user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies with GPU support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY --chown=appuser:appuser . /app

# Create directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/embeddings \
    /app/checkpoints /app/logs /app/mlruns \
    && chown -R appuser:appuser /app

USER appuser

# Default command: run training
CMD ["python", "scripts/train.py", "--config", "configs/config.yaml"]
