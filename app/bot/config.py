import os

# Telegram Bot
BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]

# Admin user IDs — comma-separated in env
_raw_admin_ids = os.environ.get("ADMIN_TELEGRAM_IDS", "")
ADMIN_IDS: set[int] = {
    int(uid.strip())
    for uid in _raw_admin_ids.split(",")
    if uid.strip().isdigit()
}

# Storage directories
ORIGINALS_DIR: str = os.environ.get("ORIGINALS_DIR", "/data/originals")
CROPS_DIR: str = os.environ.get("CROPS_DIR", "/data/crops")

# Limits
MAX_FILE_SIZE_MB: int = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))
RATE_LIMIT_SEARCH_PER_MINUTE: int = int(
    os.environ.get("RATE_LIMIT_SEARCH_PER_MINUTE", "20")
)
OCR_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.7")
)
