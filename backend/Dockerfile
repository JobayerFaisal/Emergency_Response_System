# Dockerfile inside backend directory

FROM python:3.12-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
# Set PYTHONPATH to /app
ENV PYTHONPATH=/app


# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code and other files
COPY ./src/ ./src/
COPY ./models/ ./models/
COPY ./scripts/ ./scripts/

# Copy .env file from the root directory (if required)
COPY ../.env .env

# Expose the required port
EXPOSE 8000

# Command to run the FastAPI app
# CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python", "src/agents/agent_1_environmental/main.py"]