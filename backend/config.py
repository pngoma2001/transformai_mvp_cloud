import os

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

API_KEY = os.getenv("API_KEY", "dev-key-123")

FF_GRID_RUNTIME = _env_bool("FF_GRID_RUNTIME", False)
FF_MODULES = _env_bool("FF_MODULES", False)
FF_MEMO = _env_bool("FF_MEMO", False)
FF_UI_GRID = _env_bool("FF_UI_GRID", False)

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
