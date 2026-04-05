# ============================================================
# start_agents.ps1
# Run from the PROJECT ROOT: D:\Emergency_Response_System
# Usage:  .\start_agents.ps1
# ============================================================

$PROJECT_ROOT = "D:\Emergency_Response_System"
$BACKEND_DIR  = "$PROJECT_ROOT\backend"
$VENV_PYTHON  = "$BACKEND_DIR\.venv\Scripts\python.exe"

# FIX: PYTHONPATH needs TWO entries separated by a semicolon:
#
#   1) PROJECT_ROOT  → lets Python find  "backend.src.agents.*"
#                      (because backend\ sits directly inside it)
#
#   2) BACKEND_DIR   → lets Python find  "shared.message_protocol"
#                      (because shared\ sits directly inside backend\)
#
# Without both, agent_1 works (it uses "backend.*" imports) but
# agents 2/3/4 crash with "No module named 'shared'".

$envBlock = @"
`$env:PYTHONPATH         = '$PROJECT_ROOT;$BACKEND_DIR';
`$env:DATABASE_URL       = 'postgresql://postgres:postgres@localhost:5432/disaster_response';
`$env:DATABASE_URL_ASYNC = 'postgresql://postgres:postgres@localhost:5432/disaster_response';
`$env:REDIS_URL          = 'redis://localhost:6379';
`$env:OSRM_URL           = 'http://router.project-osrm.org';
cd '$PROJECT_ROOT';
"@

# ── Agent 1 — Environmental Intelligence (port 8001) ─────────────────
# main.py calls uvicorn.run() internally, so we use python -m
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m backend.src.agents.agent_1_environmental.main
"

# ── Agent 2 — Distress Signal (port 8002) ────────────────────────────
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_2_distress.main:app --host 0.0.0.0 --port 8002 --reload
"

# ── Agent 3 — Resource Management (port 8003) ────────────────────────
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_3_resource.main:app --host 0.0.0.0 --port 8003 --reload
"

# ── Agent 4 — Dispatch Optimization (port 8004) ──────────────────────
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_4_dispatch.main:app --host 0.0.0.0 --port 8004 --reload
"

Write-Host ""
Write-Host "All 4 agents starting in separate windows..." -ForegroundColor Green
Write-Host "  Agent 1 (Environmental): http://localhost:8001/docs"
Write-Host "  Agent 2 (Distress):      http://localhost:8002/docs"
Write-Host "  Agent 3 (Resource):      http://localhost:8003/docs"
Write-Host "  Agent 4 (Dispatch):      http://localhost:8004/docs"
Write-Host ""
Write-Host "To stop all agents: Get-Process python | Stop-Process" -ForegroundColor Yellow