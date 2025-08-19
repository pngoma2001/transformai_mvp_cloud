from fastapi import HTTPException, status

def feature_flag(enabled: bool, name: str):
    def _check():
        if not enabled:
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                                detail=f"{name} is disabled by feature flag.")
    return _check
