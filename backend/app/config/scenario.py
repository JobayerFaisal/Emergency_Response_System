import os

SCENARIO_MODE = os.getenv("SCENARIO_MODE", "LIVE")
SCENARIO_DATE = os.getenv("SCENARIO_DATE", "2022-06-17T09:00:00Z")

IS_REPLAY = SCENARIO_MODE.startswith("REPLAY")

def is_replay():
    return IS_REPLAY

def get_scenario_date():
    return SCENARIO_DATE