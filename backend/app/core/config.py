"""
Configuration management for the disaster response system.
Loads settings from environment variables with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # ============================================
    # API KEYS
    # ============================================
    openweather_api_key: Optional[str] = Field(default=None)
    twitter_bearer_token: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    
    # ============================================
    # DATABASE
    # ============================================

    # POSTGRES_USER=postgres
    # POSTGRES_PASSWORD=postgres
    # POSTGRES_DB=disaster_db
    # POSTGRES_HOST=localhost
    # POSTGRES_PORT=5432



    postgres_user: str = Field(default="postgres")
    postgres_password: Optional[str] = Field(default="postgres")
    postgres_db: str = Field(default="disaster_db")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # ============================================
    # REDIS
    # ============================================
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: Optional[str] = Field(default=None)
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"
    
    # ============================================
    # AGENT CONFIGURATION
    # ============================================
    agent1_polling_interval: int = Field(default=300)
    agent1_alert_threshold: float = Field(default=10.0)
    agent1_grid_size: int = Field(default=500)
    
    # ============================================
    # GEOGRAPHIC COVERAGE
    # ============================================
    monitoring_area_name: str = Field(default="Dhaka Metropolitan")
    bbox_south: float = Field(default=23.7)
    bbox_west: float = Field(default=90.35)
    bbox_north: float = Field(default=23.9)
    bbox_east: float = Field(default=90.45)
    
    @property
    def monitoring_bbox(self) -> dict:
        """Get bounding box as dict."""
        return {
            "south": self.bbox_south,
            "west": self.bbox_west,
            "north": self.bbox_north,
            "east": self.bbox_east
        }
    
    # ============================================
    # RUNTIME MODE
    # ============================================
    runtime_mode: Literal["simulation", "production"] = Field(
        default="simulation"
    )
    
    # ============================================
    # ============================================
    # LOGGING
    # ============================================
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/agent_1.log")
    # ============================================
    # ============================================
    # ADVANCED SETTINGS
    # ============================================
    max_api_retries: int = Field(default=3)
    api_timeout: int = Field(default=30)
    data_retention_days: int = Field(default=30)
    
    enable_image_analysis: bool = Field(default=True)
    enable_predictions: bool = Field(default=True)
    enable_twitter: bool = Field(default=False)
    # ============================================
    # PATHS
    # ============================================
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        path = self.project_root / "logs"
        path.mkdir(exist_ok=True)
        return path
    
    # ============================================
    # VALIDATORS
    # ============================================
    @validator("bbox_south", "bbox_north")
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator("bbox_west", "bbox_east")
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v
    
    @validator("agent1_polling_interval")
    def validate_polling_interval(cls, v):
        if v < 10:
            raise ValueError("Polling interval must be at least 10 seconds")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Convenience function to check if we're in simulation mode
def is_simulation_mode() -> bool:
    """Check if system is running in simulation mode."""
    return settings.runtime_mode == "simulation"


# Sentinel zones for adaptive polling (Dhaka's most flood-prone areas)
SENTINEL_ZONES = [
    {"name": "Mirpur_10", "lat": 23.8103, "lon": 90.4125},
    {"name": "Mohammadpur", "lat": 23.7650, "lon": 90.3700},
    {"name": "Kamrangirchar", "lat": 23.7161, "lon": 90.3700},
    {"name": "Basundhara", "lat": 23.8223, "lon": 90.4242},
    {"name": "Uttara_Sector_7", "lat": 23.8759, "lon": 90.3795},
]

# Full monitoring grid (all flood-prone zones in Dhaka)
MONITORING_GRID = SENTINEL_ZONES + [
    {"name": "Dhanmondi", "lat": 23.7461, "lon": 90.3742},
    {"name": "Gulshan", "lat": 23.7925, "lon": 90.4078},
    {"name": "Banani", "lat": 23.7937, "lon": 90.4066},
    {"name": "Tejgaon", "lat": 23.7544, "lon": 90.3912},
    {"name": "Motijheel", "lat": 23.7334, "lon": 90.4171},
    {"name": "Jatrabari", "lat": 23.7100, "lon": 90.4333},
    {"name": "Demra", "lat": 23.7547, "lon": 90.5053},
    {"name": "Khilgaon", "lat": 23.7562, "lon": 90.4292},
    {"name": "Rampura", "lat": 23.7584, "lon": 90.4253},
    {"name": "Badda", "lat": 23.7805, "lon": 90.4267},
]