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
    gdal-bin \
    libgdal-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Install Google Cloud SDK (which includes gcloud)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.gpg > /dev/null && \
    apt-get update && apt-get install -y google-cloud-sdk


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