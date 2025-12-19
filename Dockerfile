# Dockerfile for AI Connect4 Player Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ---------------------------------------
# Install system dependencies
#    - gcc/build-essential: for compiling pip packages
#    - libpq-dev: for psycopg2 (PostgreSQL driver)
# ---------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire source directory (preserves modular structure)
COPY src/ ./src/


# Create non-root user for security
RUN useradd -ms /bin/bash aiuser && \
    chown -R aiuser:aiuser /app
USER aiuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8002

# Health check to ensure service is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8002/health', timeout=5).raise_for_status()" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]