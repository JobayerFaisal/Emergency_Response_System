# Emergency Response System - Start Agents 2, 3, 4
# Agent 1 skipped - using Sylhet 2022 simulation instead

$env:PYTHONPATH         = "D:\Emergency_Response_System\backend"
$env:DATABASE_URL_ASYNC = "postgresql://postgres:postgres@localhost:5432/disaster_response"
$env:REDIS_URL          = "redis://localhost:6379"
$env:OSRM_URL           = "http://router.project-osrm.org"

Write-Host "Starting Agent 2..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; `$env:PYTHONPATH='D:\Emergency_Response_System\backend'; `$env:DATABASE_URL_ASYNC='postgresql://postgres:postgres@localhost:5432/disaster_response'; `$env:REDIS_URL='redis://localhost:6379'; uvicorn src.agents.agent_2_distress.main:app --host 0.0.0.0 --port 8002"

Start-Sleep -Seconds 4

Write-Host "Starting Agent 3..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; `$env:PYTHONPATH='D:\Emergency_Response_System\backend'; `$env:DATABASE_URL_ASYNC='postgresql://postgres:postgres@localhost:5432/disaster_response'; `$env:REDIS_URL='redis://localhost:6379'; uvicorn src.agents.agent_3_resource.main:app --host 0.0.0.0 --port 8003"

Start-Sleep -Seconds 4

Write-Host "Starting Agent 4..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; `$env:PYTHONPATH='D:\Emergency_Response_System\backend'; `$env:DATABASE_URL_ASYNC='postgresql://postgres:postgres@localhost:5432/disaster_response'; `$env:REDIS_URL='redis://localhost:6379'; `$env:OSRM_URL='http://router.project-osrm.org'; uvicorn src.agents.agent_4_dispatch.main:app --host 0.0.0.0 --port 8004"

Start-Sleep -Seconds 6

Write-Host ""
Write-Host "All agents started!" -ForegroundColor Cyan
Write-Host "Agent 2: http://localhost:8002/docs"
Write-Host "Agent 3: http://localhost:8003/docs"
Write-Host "Agent 4: http://localhost:8004/docs"
Write-Host ""
Write-Host "Wait 10 seconds then run simulation:" -ForegroundColor Yellow
Write-Host 'Invoke-RestMethod -Method POST -Uri "http://localhost:8002/trigger/distress-queue" -ContentType "application/json" -Body (Get-Content D:\Emergency_Response_System\simulation_payload.json -Raw)'