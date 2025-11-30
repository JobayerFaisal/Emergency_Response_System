# Emergency_Response_System

Here’s a `README.md` you can drop into your project and share with your friend:

````markdown
# Emergency Response System (Multi-Agent, Dockerized Backend)

This project is a **Multi-AI-Agent based Disaster Response System**.  
The current repository contains the **backend service**, containerized with Docker, using:

- **FastAPI** (Python) – API and agent orchestration
- **PostgreSQL + PostGIS** – main database with geospatial support
- **Redis** – caching and pub/sub for real-time updates and agent communication
- **Docker & docker-compose** – to run everything with one command

> Goal: provide a backend that can ingest incident data, store it with location info, and eventually coordinate multi-agent logic for disaster response (planning, prioritization, communication, etc.).

---

## 🧱 Project Structure

```text
Emergency_Response_System/
  backend/
    app/
      __init__.py
      main.py             # FastAPI entrypoint (ASGI app)
      # (future) core/ db/ models/ schemas/ api/ agents/ ...
    requirements.txt
    Dockerfile
    .env                  # environment variables (not committed)
    .env.example          # example env file (optional, recommended)
  docker-compose.yml
````

* **`backend/app/main.py`**
  Contains the FastAPI application with a simple health endpoint:

  * `GET /api/v1/health` → `{"status": "ok"}`

* **`docker-compose.yml`**
  Orchestrates three services:

  * `db` → PostgreSQL + PostGIS
  * `redis` → Redis server
  * `backend` → FastAPI app running in a Python container

---

## 🛠️ Prerequisites

Your friend needs:

* **Docker Desktop** (Windows/macOS) or Docker Engine (Linux)
* **docker compose** / `docker-compose` available in terminal
* (Optional, for better editor support) **Python 3.10+** installed locally

No local Python/venv is required to *run* the project — Docker handles that.
A local venv is only helpful for **IDE intellisense** and type checking.

---

## 🚀 Getting Started (with Docker)

1. **Clone the repository**

   ```bash
   git clone <repo-url> Emergency_Response_System
   cd Emergency_Response_System
   ```

2. **Create backend `.env` file**

   Inside `backend/`, create a `.env` file:

   ```bash
   cd backend
   cp .env.example .env   # if .env.example exists
   # or create manually:
   ```

   Example `.env`:

   ```env
   DATABASE_URL=postgresql+psycopg2://disaster_user:secretpassword@db:5432/disaster_db
   REDIS_URL=redis://redis:6379/0
   ```

   Then go back to project root:

   ```bash
   cd ..
   ```

3. **Build and run all services**

   From the root (`Emergency_Response_System`):

   ```bash
   docker-compose up --build
   ```

   or (depending on Docker version):

   ```bash
   docker compose up --build
   ```

   This will:

   * Pull **PostGIS** and **Redis** images
   * Build the **backend** image from `backend/Dockerfile`
   * Start containers:

     * `disaster_pg` (Postgres/PostGIS)
     * `disaster_redis` (Redis)
     * `disaster_backend` (FastAPI)

4. **Verify everything is running**

   When it’s ready, logs will show something like:

   ```text
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

   Now open in browser:

   * API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Health check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

   You should see:

   ```json
   {"status": "ok"}
   ```

---

## 📡 Viewing Logs & Print Output

Anything printed in Python (e.g. `print("...")`) and FastAPI/Uvicorn logs show up in **Docker logs**.

From the project root:

```bash
# Stream backend logs live
docker-compose logs -f backend
```

You’ll see:

* Startup logs
* Requests hitting the API
* Any `print()` statements inside your routes/agents

---

## 🔧 Development Workflow

### Editing Code

Because `docker-compose.yml` mounts the backend folder:

```yaml
backend:
  volumes:
    - ./backend:/app
```

Any changes in `backend/` on your machine are reflected inside the container.

If the Docker `CMD` uses `--reload` (recommended for dev):

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

then the FastAPI server will auto-reload when you save Python files.

---

### Optional: Local venv for IDE support (Intellisense)

To remove “Import ‘fastapi’ could not be resolved” warnings and get better autocomplete (VS Code, PyCharm, etc.):

1. Create and activate a venv (inside `backend/`):

   ```bash
   cd backend
   python -m venv .venv

   # Windows (PowerShell)
   . .venv/Scripts/Activate.ps1

   # Linux/macOS
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. In your editor, select the interpreter:

   * `backend/.venv/Scripts/python.exe` (Windows)
   * or `backend/.venv/bin/python` (Linux/macOS)

This does **not** affect Docker; it’s only for local linting/intellisense.

---

## 🗄️ Services Overview

### Backend (FastAPI)

* Location: `backend/app/main.py`
* Exposed at: `http://localhost:8000`
* Example endpoint:

  * `GET /api/v1/health` → Returns `{"status": "ok"}`

As the project grows, you can add:

* `app/db/` – SQLAlchemy models, sessions
* `app/schemas/` – Pydantic schemas
* `app/api/v1/` – routers for incidents/resources/plans
* `app/agents/` – ingestion, assessment, planning, safety, communication agents

---

### Database (PostgreSQL + PostGIS)

* Service name in Docker: `db`
* Host from backend: `db` (not `localhost`)
* Port exposed to host: `5432`
* Default credentials (see `docker-compose.yml`):

  ```yaml
  environment:
    POSTGRES_DB: disaster_db
    POSTGRES_USER: disaster_user
    POSTGRES_PASSWORD: secretpassword
  ```

You can connect with a DB client (e.g. DBeaver, TablePlus, pgAdmin):

* Host: `localhost`
* Port: `5432`
* User: `disaster_user`
* DB: `disaster_db`

PostGIS extensions are automatically enabled in the container.

---

### Redis

* Service name in Docker: `redis`
* Host from backend: `redis`
* Port exposed to host: `6379`

Planned usage:

* Caching frequently accessed data (e.g., dashboard stats)
* Pub/sub channels for agent communication and WebSocket updates

---

## 🤝 How to Contribute (for collaborators)

1. **Clone** the repo and follow the **Docker setup** steps above.

2. **Create a new branch** for your feature:

   ```bash
   git checkout -b feature/<name>
   ```

3. **Add or modify endpoints** in `backend/app/...`.

4. Test locally using:

   ```bash
   docker-compose up --build
   ```

5. Once working, push and create a **pull request**.

Suggested coding areas:

* Add `/api/v1/incidents` (CRUD with PostGIS location)
* Implement agent modules under `app/agents/`
* Add Redis pub/sub for real-time updates
* Add more health/status endpoints for the dashboard

---

## 🧩 Troubleshooting

* **Warning: `the attribute 'version' is obsolete` in docker-compose.yml**

  This is just a warning from newer Compose. You can safely ignore it or remove the `version:` line from `docker-compose.yml`.

* **Backend container exits with `Could not import module "app.main"`**

  Check:

  * `backend/app/main.py` exists

  * `backend/app/__init__.py` exists

  * `main.py` contains:

    ```python
    from fastapi import FastAPI
    app = FastAPI()
    ```

  * `Dockerfile` uses: `CMD ["uvicorn", "app.main:app", ...]`

* **Editor shows `Import "fastapi" could not be resolved`**

  Create a local venv in `backend/` and install requirements (see “Local venv for IDE support”).

---

## 📜 License

Add your license info here if needed (MIT, Apache 2.0, etc.).

```

You can tweak the project name, add your name in a “Authors” section, or add more details once you implement the incident/agent logic.
```
