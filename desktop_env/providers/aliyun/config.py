import os


MIN_TTL_MINUTES: int = 30

_ttl_env_str = os.getenv("DEFAULT_TTL_MINUTES", "60")
try:
    _ttl_env_val = int(_ttl_env_str)
except Exception:
    _ttl_env_val = 60

if _ttl_env_val > 0 and _ttl_env_val < MIN_TTL_MINUTES:
    DEFAULT_TTL_MINUTES: int = MIN_TTL_MINUTES
else:
    DEFAULT_TTL_MINUTES: int = _ttl_env_val

ENABLE_TTL: bool = os.getenv("ENABLE_TTL", "true").lower() == "true"


def compute_ttl_seconds(ttl_minutes: int) -> int:
    try:
        return max(0, int(ttl_minutes) * 60)
    except Exception:
        return 0


