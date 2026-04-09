import os
import json
from typing import Optional

# SCENARIO_MODE env var defines *which* replay scenario is available,
# but the app always boots into LIVE mode. Replay is only activated
# when the user clicks "Historical Replay" in the UI.
_REPLAY_SCENARIO_NAME = os.getenv("SCENARIO_MODE", "REPLAY_HISTORICAL")
_REPLAY_SCENARIO_DATE = os.getenv("SCENARIO_DATE", "2022-06-17T09:00:00Z")

_SCENARIO_STATE = {
    "mode": "LIVE",  # always start LIVE — never read mode from env
    "scenario_date": _REPLAY_SCENARIO_DATE,
    "replay_scenario_name": _REPLAY_SCENARIO_NAME,
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
    # In-memory state is authoritative. Redis is only used to
    # sync across workers — never to override an explicit set_mode() call.
    return _SCENARIO_STATE.get("mode", "LIVE")

def is_replay() -> bool:
    return get_mode().startswith("REPLAY")

def get_scenario_date() -> str:
    return _SCENARIO_STATE.get("scenario_date", _REPLAY_SCENARIO_DATE)

def get_replay_scenario_name() -> str:
    """Returns the configured replay scenario name from the env var."""
    return _SCENARIO_STATE.get("replay_scenario_name", _REPLAY_SCENARIO_NAME)