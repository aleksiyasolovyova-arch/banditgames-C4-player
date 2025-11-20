# Dockerfile for MCTS Repository
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCTS code
COPY src/*.py ./

# Create non-root user
RUN useradd -ms /bin/bash mcts_user
USER mcts_user

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import mcts; import ai_manager; print('OK')"

# Keep container running (it's a library, not a service)
CMD ["python", "-c", "import time; print('MCTS Service Ready'); time.sleep(86400)"]