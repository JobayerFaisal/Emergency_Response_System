"""
Environmental Intelligence Agent (WEATHER ONLY + Dummy Flood Output)
====================================================================

This version:
- Collects weather data only
- Processes & normalizes weather
- Runs spatial analysis (weather-based)
- ALWAYS outputs a dummy flood prediction (winter-mode)
- No LLM, No social media
- 100% compatible with your uploaded modules

Author: Environmental Intelligence Team
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, cast
from datetime import datetime, timezone




import asyncpg
from redis import asyncio as aioredis

# Internal imports
from app.agents.agent_1_environmental.models import (
    AgentOutput,
    SentinelZone,
    GeoPoint,
    SeverityLevel,
    MonitoringStatus,
)
from .data_collectors import WeatherAPICollector, DataCollectionOrchestrator
from .data_processors import WeatherDataNormalizer, DataProcessingOrchestrator
from .spatial_analyzer import PostGISSpatialAnalyzer
from .predictor import FloodRiskPredictor, AlertGenerator, PredictionOrchestrator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ===================================================================
# CONFIGURATION
# ===================================================================
class AgentConfig:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()

        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.openweather_api_key:
            raise ValueError("Missing OPENWEATHER_API_KEY")

        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@db:5432/disaster_db",
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379")

        self.monitoring_interval = int(os.getenv("MONITORING_INTERVAL", 300))


# ===================================================================
# MAIN AGENT (WEATHER ONLY)
# ===================================================================
class EnvironmentalIntelligenceAgent:
    def __init__(self, config: AgentConfig):
        self.config = config

        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None

        self.sentinel_zones: List[SentinelZone] = []
        self.latest_output: Optional[AgentOutput] = None
        self.last_update: Optional[datetime] = None

        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Environmental Agent initialized (weather-only mode).")


    async def shutdown(self):
        """Gracefully stop environmental agent."""
        self.running = False

        # Close DB pool
        if self.db_pool:
            await self.db_pool.close()
            self.db_pool = None

        # Close Redis
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

        logging.info("Environmental Agent shut down successfully.")









    # --------------------------------------------------------------
    async def startup(self):
        logger.info("Starting Environmental Agent...")

        # DB
        self.db_pool = await asyncpg.create_pool(self.config.database_url)
        logger.info("PostgreSQL connected.")

        # Redis
        self.redis_client = await aioredis.from_url(
            self.config.redis_url, decode_responses=True
        )
        logger.info("Redis connected.")

        # Weather Collector
        weather_collector = WeatherAPICollector(
            api_key=cast(str, self.config.openweather_api_key),
            cache_client=self.redis_client,
        )

        # WEATHER-ONLY orchestrator
        self.collection_orchestrator = DataCollectionOrchestrator(
            weather_collector=weather_collector
        )

        # WEATHER-ONLY processor
        weather_normalizer = WeatherDataNormalizer()
        self.processing_orchestrator = DataProcessingOrchestrator(
            llm_processor=None,
            weather_normalizer=weather_normalizer,
            social_analyzer=None,
        )

        # Spatial analyzer
        self.spatial_analyzer = PostGISSpatialAnalyzer(db_pool=self.db_pool)
        await self.spatial_analyzer.initialize_schema()

        # Prediction engines
        predictor = FloodRiskPredictor()
        alert_gen = AlertGenerator()
        self.prediction_orchestrator = PredictionOrchestrator(predictor, alert_gen)

        # Load zones
        await self.load_sentinel_zones()

        logger.info("Environmental Agent startup complete.")

    # --------------------------------------------------------------
    async def load_sentinel_zones(self):
        """Load zones from DB or create defaults."""
        
        if self.db_pool is None:
            raise RuntimeError("Database pool not initialized. Call startup() first.")

        # Tell type checker that db_pool is now definitely asyncpg.Pool
        db_pool = self.db_pool
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    id, name, radius_km, population_density, elevation,
                    drainage_capacity, risk_level,
                    ST_Y(center::geometry) AS lat,
                    ST_X(center::geometry) AS lon,
                    created_at, last_monitored
                FROM sentinel_zones;
            """)

        if rows:
            for r in rows:
                self.sentinel_zones.append(
                    SentinelZone(
                        id=r["id"],
                        name=r["name"],
                        center=GeoPoint(latitude=r["lat"], longitude=r["lon"]),
                        radius_km=r["radius_km"],
                        risk_level=SeverityLevel(r["risk_level"]),
                        population_density=r["population_density"],
                        elevation=r["elevation"],
                        drainage_capacity=r["drainage_capacity"],
                        created_at=r["created_at"],
                        last_monitored=r["last_monitored"],
                    )
                )
            return

        # Defaults (Dhaka)
        defaults = [
            ("Dhaka Central", 23.8103, 90.4125, 5.0, SeverityLevel.MODERATE, 45000, 6.0, "poor"),
            ("Mirpur",        23.8223, 90.3654, 4.0, SeverityLevel.HIGH,     52000, 4.0, "poor"),
            ("Gulshan",       23.7806, 90.4175, 3.0, SeverityLevel.LOW,      35000, 8.0, "moderate"),
            ("Mohammadpur",   23.7697, 90.3611, 4.0, SeverityLevel.MODERATE, 48000, 5.0, "poor"),
            ("Uttara",        23.8759, 90.3795, 4.5, SeverityLevel.MODERATE, 42000, 7.0, "moderate"),
        ]

        for name, lat, lon, r, risk, pop, elev, drainage in defaults:
            zone = SentinelZone(
                name=name,
                center=GeoPoint(latitude=lat, longitude=lon),
                radius_km=r,
                risk_level=risk,
                population_density=pop,
                elevation=elev,
                drainage_capacity=drainage,
            )
            await self.spatial_analyzer.store_sentinel_zone(zone)
            self.sentinel_zones.append(zone)

    # --------------------------------------------------------------
    async def run_monitoring_cycle(self) -> AgentOutput:
        logger.info("Starting monitoring cycle (weather-only)...")
        start = datetime.utcnow()

        # 1️⃣ Collect weather for all zones
        collected = await self.collection_orchestrator.collect_all_zones(self.sentinel_zones)

        # 2️⃣ Normalize weather
        processed = await self.processing_orchestrator.process_all_zones(collected)

        # 3️⃣ Spatial analysis (weather only)
        for item in processed:
            zone = item["zone"]
            weather = item.get("weather")

            # Store weather
            if weather:
                await self.spatial_analyzer.store_weather_data(weather, str(zone.id))

            spatial = await self.spatial_analyzer.analyze_zone_spatial_patterns(zone)
            item["spatial_analysis"] = spatial

        # 4️⃣ Historical risk
        historical_risks = {
            str(z.id): await self.spatial_analyzer.get_historical_risk_score(z)
            for z in self.sentinel_zones
        }

        # 5️⃣ Predict (based only on weather)
        predictions, alerts = await self.prediction_orchestrator.predict_all_zones(
            processed, historical_risks
        )

        # 6️⃣ Store prediction records
        if self.db_pool is None:
            raise RuntimeError("Database pool not initialized. Call startup() first.")
        db_pool = self.db_pool
        async with self.db_pool.acquire() as conn:
            for pred in predictions:
                await conn.execute(
                    """
                    INSERT INTO flood_predictions
                    (id, zone_id, timestamp, risk_score, severity_level,
                     confidence, time_to_impact_hours, affected_area_km2,
                     risk_factors, recommended_actions)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    pred.id,
                    pred.zone.id,
                    pred.timestamp,
                    pred.risk_score,
                    pred.severity_level.value,
                    pred.confidence,
                    pred.time_to_impact_hours,
                    pred.affected_area_km2,
                    pred.risk_factors.model_dump_json(),
                    json.dumps(pred.recommended_actions),
                )

        # Update zone timestamps
        now = datetime.utcnow()
        for zone in self.sentinel_zones:
            zone.last_monitored = now
            await self.spatial_analyzer.store_sentinel_zone(zone)

        duration = (datetime.utcnow() - start).total_seconds()

        output = AgentOutput(
            agent_id="agent_1_environmental",
            timestamp=datetime.utcnow(),
            predictions=predictions,
            alerts=alerts,
            monitored_zones=self.sentinel_zones,
            data_sources_status={"weather": "operational"},
            processing_time_seconds=duration,
            next_update_in_seconds=self.config.monitoring_interval,
        )

        self.latest_output = output
        self.last_update = datetime.utcnow()

        logger.info("Monitoring cycle completed successfully.")
        return output

    # --------------------------------------------------------------
    async def start_monitoring(self):
        self.running = True
        logger.info("Weather-only monitoring loop started…")

        while self.running:
            try:
                out = await self.run_monitoring_cycle()
                await asyncio.sleep(out.next_update_in_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                await asyncio.sleep(60)

    # --------------------------------------------------------------

    def get_status(self) -> MonitoringStatus:
        """Return monitoring agent status (safe even before first cycle)."""

        now = datetime.now(timezone.utc)

        # No monitoring cycle has completed yet
        if self.latest_output is None or self.last_update is None:
            return MonitoringStatus(
                active_zones=len(self.sentinel_zones),
                total_predictions=0,
                critical_alerts=0,
                last_update=now,
                next_update=now,
                data_freshness_seconds=999999,
            )

        # Convert stored timestamp to timezone-aware UTC
        last_update = (
            self.last_update
            if self.last_update.tzinfo is not None
            else self.last_update.replace(tzinfo=timezone.utc)
        )

        freshness = (now - last_update).total_seconds()

        return MonitoringStatus(
            active_zones=len(self.sentinel_zones),
            total_predictions=len(self.latest_output.predictions),
            critical_alerts=len(self.latest_output.critical_alerts),
            last_update=last_update,
            next_update=last_update + timedelta(seconds=self.latest_output.next_update_in_seconds),
            data_freshness_seconds=freshness,
        )