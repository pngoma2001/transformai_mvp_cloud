from fastapi import Header, HTTPException, status
from .. import config
from .limits import rate_limit

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or x_api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    rate_limit(x_api_key, config.RATE_LIMIT_RPM)
