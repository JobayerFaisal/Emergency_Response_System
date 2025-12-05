import os
import asyncpg
import pandas as pd
import asyncio


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/disaster_db"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()
