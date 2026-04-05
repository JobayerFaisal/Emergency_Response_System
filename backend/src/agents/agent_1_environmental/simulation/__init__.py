"""
HOW TO INTEGRATE SIMULATION INTO AGENT 1
=========================================

1. Copy the simulation/ folder to:
   backend/src/agents/agent_1_environmental/simulation/
   
   Files:
     simulation/__init__.py      ← this file
     simulation/sylhet_2022.py   ← historical data
     simulation/runner.py        ← Redis publisher

2. In backend/src/agents/agent_1_environmental/main.py, add these imports
   near the top (after existing imports):

   from .simulation.runner import SimulationRunner
   from .simulation.sylhet_2022 import SCENARIOS, get_scenario_summary

3. Add these API endpoints to main.py (paste before the last `if __name__` block):

─────────────────────────────────────────────────────────────────────────────

@app.post("/simulate")
async def run_simulation(request: dict = {}):
    \"\"\"
    🌊 Run Sylhet 2022 Flood Simulation.
    
    Injects real historical flood data into the live pipeline.
    Agent 2 → Agent 3 → Agent 4 will respond automatically.
    
    Body (all optional):
    {
        "scenario": "peak",        // "peak" | "early" | "single_zone"
        "delay_seconds": 1.0       // pause between zone alerts (for demo effect)
    }
    \"\"\"
    scenario      = request.get("scenario", "peak")
    delay_seconds = float(request.get("delay_seconds", 1.0))
    
    runner = SimulationRunner(redis_client=agent.redis_client)
    result = await runner.run(scenario=scenario, delay_seconds=delay_seconds)
    return result


@app.get("/simulate/scenarios")
async def list_scenarios():
    \"\"\"List all available simulation scenarios.\"\"\"
    return {
        "scenarios": {
            name: get_scenario_summary(name)
            for name in SCENARIOS.keys()
        },
        "usage": "POST /simulate with body: {\"scenario\": \"peak\"}"
    }


─────────────────────────────────────────────────────────────────────────────

4. That's it! Test with:

   # PowerShell — run peak flood simulation
   Invoke-RestMethod -Method POST -Uri "http://localhost:8001/simulate" \\
     -ContentType "application/json" \\
     -Body '{\"scenario\": \"peak\", \"delay_seconds\": 1.0}'

   # Or open http://localhost:8001/docs and use Swagger UI
   # Click /simulate → Try it out → Execute
"""
