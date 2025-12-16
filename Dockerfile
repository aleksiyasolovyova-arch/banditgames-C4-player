# Dockerfile for MCTS Repository (Service with RabbitMQ)
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

# Copy MCTS source files from src directory
COPY src/mcts/mcts.py .
COPY src/ConnectState.py .
COPY src/config/meta.py .
COPY src/core/ai_manager.py .

# Copy the listener script
COPY src/in/mcts_listener.py .

# Copy dataset generation files
COPY src/dataset_exporter.py .
COPY src/core/self_play_runner.py .

# Create logs and data directories
RUN mkdir -p /app/logs /app/data/datasets

# Create non-root user for security
RUN useradd -ms /bin/bash mcts_user && \
    chown -R mcts_user:mcts_user /app
USER mcts_user

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check to ensure service is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import mcts; import pika; print('OK')" || exit 1

# Run the MCTS listener service
CMD ["python", "mcts_listener.py"]