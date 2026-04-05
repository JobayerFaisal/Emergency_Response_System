# ============================================================
# Emergency Response System — Unified Backend Dockerfile
# Works for: Agent 1, 2, 3, 4, and Streamlit
# ============================================================

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire backend
COPY . .

# Default port (overridden per-service in docker-compose)
EXPOSE 8000

# Default command (overridden per-service in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]