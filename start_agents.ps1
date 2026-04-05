# Set environment variables
$env:PYTHONPATH    = "D:\Emergency_Response_System\backend"
$env:DATABASE_URL  = "postgresql://postgres:postgres@localhost:5432/disaster_response"
$env:DATABASE_URL_ASYNC = "postgresql://postgres:postgres@localhost:5432/disaster_response"
$env:REDIS_URL     = "redis://localhost:6379"
$env:OSRM_URL      = "http://router.project-osrm.org"

# Start all 4 agents in separate windows
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; python -m backend.src.agents.agent_1_environmental.main"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; uvicorn src.agents.agent_2_distress.main:app --host 0.0.0.0 --port 8002"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; uvicorn src.agents.agent_3_resource.main:app --host 0.0.0.0 --port 8003"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Emergency_Response_System\backend; uvicorn src.agents.agent_4_dispatch.main:app --host 0.0.0.0 --port 8004"

Write-Host "✅ All 4 agents starting in separate windows..."
Write-Host "Agent 1: http://localhost:8001"
Write-Host "Agent 2: http://localhost:8002"
Write-Host "Agent 3: http://localhost:8003"
Write-Host "Agent 4: http://localhost:8004"