# ============================================================
# start_agents.ps1
# Run from the PROJECT ROOT: D:\Emergency_Response_System
# Usage:  .\start_agents.ps1
# ============================================================

$PROJECT_ROOT = "D:\Emergency_Response_System"
$BACKEND_DIR  = "$PROJECT_ROOT\backend"
$VENV_PYTHON  = "$BACKEND_DIR\.venv\Scripts\python.exe"
$REDIS_DIR    = "C:\Users\redis-windows-8.6.2\redis-windows-8.6.2"

$envBlock = @"
`$env:PYTHONPATH         = '$PROJECT_ROOT;$BACKEND_DIR';
`$env:DATABASE_URL       = 'postgresql://postgres:postgres@localhost:5432/disaster_response';
`$env:DATABASE_URL_ASYNC = 'postgresql://postgres:postgres@localhost:5432/disaster_response';
`$env:REDIS_URL          = 'redis://localhost:6379';
`$env:OSRM_URL           = 'http://router.project-osrm.org';
cd '$PROJECT_ROOT';
"@

# ── Step 1: Start Redis ───────────────────────────────────────────────
Write-Host "Starting Redis..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$REDIS_DIR'; .\redis-server.exe"
Start-Sleep -Seconds 3

# Verify Redis is up
$redisPing = & "$BACKEND_DIR\.venv\Scripts\python.exe" -c "import asyncio; from redis import asyncio as r; asyncio.run(r.from_url('redis://localhost:6379').ping())" 2>&1
if ($redisPing -match "True") {
    Write-Host "Redis is running" -ForegroundColor Green
} else {
    Write-Host "Redis may still be starting - continuing..." -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# ── Step 2: Start Agent 1 — Environmental Intelligence (port 8001) ───
Write-Host "Starting Agent 1..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m backend.src.agents.agent_1_environmental.main
"
Start-Sleep -Seconds 3

# ── Step 3: Start Agent 2 — Distress Signal (port 8002) ──────────────
Write-Host "Starting Agent 2..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_2_distress.main:app --host 0.0.0.0 --port 8002 --reload
"
Start-Sleep -Seconds 3

# ── Step 4: Start Agent 3 — Resource Management (port 8003) ──────────
Write-Host "Starting Agent 3..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_3_resource.main:app --host 0.0.0.0 --port 8003 --reload
"
Start-Sleep -Seconds 3

# ── Step 5: Start Agent 4 — Dispatch Optimization (port 8004) ────────
Write-Host "Starting Agent 4..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    $envBlock
    & '$VENV_PYTHON' -m uvicorn backend.src.agents.agent_4_dispatch.main:app --host 0.0.0.0 --port 8004 --reload
"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " All services starting in separate windows!" -ForegroundColor Green
Write-Host "============================================================"
Write-Host "  Redis:                   localhost:6379"
Write-Host "  Agent 1 (Environmental): http://localhost:8001/docs"
Write-Host "  Agent 2 (Distress):      http://localhost:8002/docs"
Write-Host "  Agent 3 (Resource):      http://localhost:8003/docs"
Write-Host "  Agent 4 (Dispatch):      http://localhost:8004/docs"
Write-Host ""
Write-Host "Wait ~15 seconds for all agents to fully start." -ForegroundColor Yellow
Write-Host ""
Write-Host "Then run Sylhet 2022 simulation:" -ForegroundColor Cyan
Write-Host '  Invoke-RestMethod -Method POST -Uri "http://localhost:8002/trigger/distress-queue" -ContentType "application/json" -Body (Get-Content D:\Emergency_Response_System\simulation_payload.json -Raw)'
Write-Host ""
Write-Host "To stop everything:" -ForegroundColor Red
Write-Host "  Get-Process python,uvicorn | Stop-Process"