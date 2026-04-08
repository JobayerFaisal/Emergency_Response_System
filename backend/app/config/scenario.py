import os
import json
from typing import Optional

_SCENARIO_STATE = {
    "mode": os.getenv("SCENARIO_MODE", "LIVE"),
    "scenario_date": os.getenv("SCENARIO_DATE", "2022-06-17T09:00:00Z"),
}

_redis_client = None

def set_redis_client(client):
    global _redis_client
    _redis_client = client

def _load_from_redis():
    if not _redis_client:
        return None
    try:
        data = _redis_client.get("scenario_state")
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None

def _save_to_redis(state):
    if not _redis_client:
        return
    try:
        _redis_client.set("scenario_state", json.dumps(state))
    except Exception:
        pass

def set_mode(mode: str, scenario_date: Optional[str] = None):
    global _SCENARIO_STATE
    _SCENARIO_STATE["mode"] = mode
    if scenario_date is not None:
        _SCENARIO_STATE["scenario_date"] = scenario_date
    _save_to_redis(_SCENARIO_STATE)

def get_mode() -> str:
    state = _load_from_redis() or _SCENARIO_STATE
    return state.get("mode", "LIVE")

def is_replay() -> bool:
    return get_mode().startswith("REPLAY")

def get_scenario_date() -> str:
    state = _load_from_redis() or _SCENARIO_STATE
    return state.get("scenario_date", "2022-06-17T09:00:00Z")