import time, threading
from collections import deque
from typing import Deque, Dict

_lock = threading.Lock()
_hits: Dict[str, Deque[float]] = {}

def rate_limit(key: str, rpm: int) -> None:
    now = time.monotonic()
    with _lock:
        q = _hits.setdefault(key, deque())
        while q and (now - q[0]) > 60.0:
            q.popleft()
        if len(q) >= rpm:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
        q.append(now)
