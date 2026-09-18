# Multi-arch compatible Dockerfile for Databricks Steward Agent
# Compatible with Docker and Podman (rootless & rootful)
FROM python:3.11-slim

LABEL maintainer="Databricks Steward Agent Team"
LABEL description="GenAI Data Steward & Semantic Engine for Databricks & Open WebUI"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/steward/.local/bin:${PATH}"

WORKDIR /app

# Install system dependencies (git for GitOps, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for rootless Podman / Docker security
RUN groupadd -g 10001 steward && \
    useradd -u 10001 -g steward -m -s /bin/bash steward && \
    chown -R steward:steward /app

# Switch to non-root user for pip install and running
USER steward

# Copy requirements and install
COPY --chown=steward:steward requirements.txt .
RUN pip install --user "cryptography<=42.0.8" && \
    pip install --user -r requirements.txt

# Copy application source code and configurations
COPY --chown=steward:steward pyproject.toml .
COPY --chown=steward:steward open_webui_pipe.py .
COPY --chown=steward:steward configs/ configs/
COPY --chown=steward:steward src/ src/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution: run FastAPI OpenAI-compatible server
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
