import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    POSTGRES_URL = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 10))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
