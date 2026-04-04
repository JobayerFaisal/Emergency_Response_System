FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/agents/agent_3_resource/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY src/__init__.py ./src/__init__.py
COPY src/agents/__init__.py ./src/agents/__init__.py
COPY src/agents/agent_3_resource/ ./src/agents/agent_3_resource/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.agents.agent_3_resource.main:app", "--host", "0.0.0.0", "--port", "8000"]
