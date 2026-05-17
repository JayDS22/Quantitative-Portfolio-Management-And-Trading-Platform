# Multi-stage build for production-ready trading platform
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models logs

# Expose ports
EXPOSE 8000 8080

# Development command
CMD ["python", "main.py", "run"]

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash trading

# Copy application
COPY --chown=trading:trading . .

# Create directories
RUN mkdir -p data/raw data/processed data/models logs \
    && chown -R trading:trading data logs

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "main:app"]
