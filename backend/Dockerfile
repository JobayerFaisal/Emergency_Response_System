FROM python:3.12-slim

# System dependencies for psycopg2, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8000

# For dev: use --reload if you mounted code as a volume
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
