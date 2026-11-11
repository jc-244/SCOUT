import os


DEFAULT_TTL_MINUTES: int = int(os.getenv("DEFAULT_TTL_MINUTES", "180"))

ENABLE_TTL: bool = os.getenv("ENABLE_TTL", "true").lower() == "true"

AWS_SCHEDULER_ROLE_ARN: str = os.getenv("AWS_SCHEDULER_ROLE_ARN", "").strip()


def compute_ttl_seconds(ttl_minutes: int) -> int:
    try:
        return max(0, int(ttl_minutes) * 60)
    except Exception:
        return 0


