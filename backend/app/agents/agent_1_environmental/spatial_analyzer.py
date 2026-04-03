"""
Spatial Analyzer for Environmental Intelligence Agent (Weather-Only)
====================================================================
Performs only weather-based analysis + satellite integration.

Author: Environmental Intelligence Team
Version: 2.0.3 (Weather Only, DB schema fix + JSON fix + SQL fix)
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from asyncpg import Pool

from app.agents.agent_1_environmental.models import (
    GeoPoint, BoundingBox, SentinelZone,
    WeatherData, SpatialAnalysisResult, SeverityLevel
)

logger = logging.getLogger(__name__)


class PostGISSpatialAnalyzer:
    """
    Performs geospatial analysis using PostGIS.
    """

    def __init__(self, db_pool: Pool):
        self.db_pool = db_pool
        logger.info("PostGISSpatialAnalyzer initialized (WEATHER ONLY)")

    # =====================================================================
    # SCHEMA
    # =====================================================================

    async def initialize_schema(self) -> None:
        async with self.db_pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sentinel_zones (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255),
                    center GEOMETRY(Point, 4326),
                    radius_km FLOAT,
                    risk_level VARCHAR(50),
                    population_density INTEGER,
                    elevation FLOAT,
                    drainage_capacity VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_monitored TIMESTAMP
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    id UUID PRIMARY KEY,
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    location GEOMETRY(Point, 4326) NOT NULL,
                    temperature FLOAT,
                    humidity FLOAT,
                    pressure FLOAT,
                    wind_speed FLOAT,
                    precipitation_1h FLOAT,
                    precipitation_3h FLOAT,
                    precipitation_24h FLOAT,
                    condition VARCHAR(50),
                    raw_data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS flood_predictions (
                    id UUID PRIMARY KEY,
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    risk_score FLOAT,
                    severity_level VARCHAR(50),
                    confidence FLOAT,
                    time_to_impact_hours FLOAT,
                    affected_area_km2 FLOAT,
                    risk_factors JSONB,
                    recommended_actions JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS satellite_flood_maps (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    flood_probability FLOAT,
                    estimated_depth FLOAT,
                    raw_features JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sat_zone ON satellite_flood_maps(zone_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sat_time ON satellite_flood_maps(timestamp DESC);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_weather_loc ON weather_data USING GIST(location);"
            )

            logger.info("Weather-only database schema ready.")

    # =====================================================================
    # STORE WEATHER (fixed to use model_dump)
    # =====================================================================

    async def store_weather_data(self, weather: WeatherData, zone_id: Optional[str]):
        """
        Stores *complete* WeatherData JSON structure (not raw OpenWeather API).
        This ensures we can fully reconstruct WeatherData later.
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO weather_data (
                    id, zone_id, timestamp, location,
                    temperature, humidity, pressure, wind_speed,
                    precipitation_1h, precipitation_3h, precipitation_24h,
                    condition, raw_data
                ) VALUES (
                    $1, $2, $3, ST_GeomFromText($4, 4326),
                    $5, $6, $7, $8, $9, $10, $11,
                    $12, $13
                )
                ON CONFLICT (id) DO NOTHING;
            """,
                weather.id,
                zone_id,
                weather.timestamp,
                weather.location.to_wkt(),
                weather.metrics.temperature,
                weather.metrics.humidity,
                weather.metrics.pressure,
                weather.metrics.wind_speed,
                weather.precipitation.rain_1h,
                weather.precipitation.rain_3h,
                weather.precipitation.rain_24h,
                weather.condition.value,
                json.dumps(weather.model_dump()),   # FULL WeatherData JSON
            )

    # =====================================================================
    # SATELLITE
    # =====================================================================

    async def get_latest_satellite_data(self, zone_id: str):
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT flood_probability, estimated_depth, raw_features
                FROM satellite_flood_maps
                WHERE zone_id = $1
                ORDER BY timestamp DESC
                LIMIT 1;
            """, zone_id)

        if not row:
            return None

        return {
            "flood_probability": row["flood_probability"],
            "estimated_depth": row["estimated_depth"],
            "raw_features": row["raw_features"],
        }

    # =====================================================================
    # MAIN SPATIAL ANALYSIS
    # =====================================================================

    async def analyze_zone_spatial_patterns(self, zone: SentinelZone) -> SpatialAnalysisResult:

        logger.info(f"Spatial analysis (weather-only) for: {zone.name}")

        # -----------------------------------------------------------------
        # Satellite
        # -----------------------------------------------------------------
        sat = await self.get_latest_satellite_data(str(zone.id))
        sat_prob = sat["flood_probability"] if sat else 0.0
        sat_depth = sat["estimated_depth"] if sat else 0.0

        # -----------------------------------------------------------------
        # Latest weather (FIXED SQL + JSON parsing)
        # -----------------------------------------------------------------
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT raw_data
                FROM weather_data
                WHERE zone_id = $1
                ORDER BY timestamp DESC     -- FIXED
                LIMIT 1;
            """, zone.id)

        latest_weather = None

        if row and row["raw_data"]:
            raw = row["raw_data"]

            try:
                # asyncpg JSONB is returned as string unless codecs installed
                if isinstance(raw, str):
                    raw = json.loads(raw)

                if isinstance(raw, dict):
                    latest_weather = WeatherData(**raw)
                else:
                    logger.error(f"[WeatherData] raw_data is not dict: {raw}")

            except Exception as e:
                logger.error(f"[WeatherData] Reconstruction failed: {e}")

        # -----------------------------------------------------------------
        # Affected area (rainfall)
        # -----------------------------------------------------------------
        affected_area = await self.estimate_affected_area_weather(zone, latest_weather)

        # -----------------------------------------------------------------
        # Population
        # -----------------------------------------------------------------
        affected_pop = await self.estimate_population(zone, affected_area)

        # -----------------------------------------------------------------
        # Severity score
        # -----------------------------------------------------------------
        weather_severity = 0.0
        if latest_weather:
            from data_processors import WeatherDataNormalizer
            normalized = WeatherDataNormalizer().normalize_weather_data(latest_weather)
            weather_severity = normalized.get("weather_severity", 0.0)

        combined_severity = min(1.0, weather_severity * 0.6 + sat_prob * 0.4)

        # -----------------------------------------------------------------
        # Infra at risk
        # -----------------------------------------------------------------
        critical = await self._detect_infra(zone, affected_area)

        # -----------------------------------------------------------------
        # Final result
        # -----------------------------------------------------------------
        return SpatialAnalysisResult(
            zone=zone,
            timestamp=datetime.utcnow(),
            affected_area_km2=affected_area,
            nearby_reports_count=0,
            average_severity=combined_severity,
            risk_clusters=[],
            affected_population_estimate=affected_pop,
            critical_infrastructure_at_risk=critical,
            satellite_flood_probability=sat_prob,
            satellite_estimated_depth=sat_depth
        )

    # =====================================================================
    # HELPERS
    # =====================================================================

    async def estimate_affected_area_weather(self, zone: SentinelZone, weather: Optional[WeatherData]) -> float:
        if not weather:
            return 0.0

        intensity = weather.precipitation.rain_1h or 0
        total = weather.precipitation.total_rain

        base_area = zone.radius_km ** 2 * 0.05
        intensity_factor = min(1.0, intensity / 40.0)
        total_factor = min(1.0, total / 150.0)

        return min(
            base_area * (0.5 * intensity_factor + 0.5 * total_factor + 1),
            zone.radius_km ** 2 * 3.14159
        )

    async def estimate_population(self, zone: SentinelZone, area_km2: float):
        if not zone.population_density or area_km2 <= 0:
            return None

        estimate = int(zone.population_density * area_km2)
        max_pop = int(zone.population_density * zone.radius_km ** 2 * 3.14159)

        return min(estimate, max_pop)

    async def _detect_infra(self, zone: SentinelZone, area: float) -> List[str]:
        infra = []
        if area > 1.0:
            infra.append("roads_primary")
        if area > 3.0:
            infra.append("commercial_areas")
        if area > 5.0:
            infra.extend(["schools", "hospitals"])
        return infra

    async def get_historical_risk_score(self, zone: SentinelZone, lookback_days=30) -> float:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT AVG(risk_score) AS avg, MAX(risk_score) AS mx, COUNT(*) AS c
                FROM flood_predictions
                WHERE zone_id=$1 AND timestamp >= NOW() - INTERVAL '{lookback_days} days';
            """, zone.id)

        if not row or row["c"] == 0:
            return 0.0

        return min(1.0, float(row["avg"]) * 0.7 + float(row["mx"]) * 0.3)

    async def cleanup_old_data(self, days_to_keep=30):
        async with self.db_pool.acquire() as conn:
            cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
            await conn.execute("DELETE FROM weather_data WHERE timestamp < $1;", cutoff)

            pred_cutoff = datetime.utcnow() - timedelta(days=90)
            await conn.execute("DELETE FROM flood_predictions WHERE timestamp < $1;", pred_cutoff)

            logger.info("Old data cleaned.")


    async def store_sentinel_zone(self, zone: SentinelZone) -> None:
        """Insert or update sentinel zone."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sentinel_zones (
                    id, name, center, radius_km, risk_level,
                    population_density, elevation, drainage_capacity,
                    created_at, last_monitored
                ) VALUES (
                    $1, $2, ST_GeomFromText($3, 4326), $4, $5,
                    $6, $7, $8, $9, $10
                )
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    center = EXCLUDED.center,
                    radius_km = EXCLUDED.radius_km,
                    risk_level = EXCLUDED.risk_level,
                    population_density = EXCLUDED.population_density,
                    elevation = EXCLUDED.elevation,
                    drainage_capacity = EXCLUDED.drainage_capacity,
                    last_monitored = EXCLUDED.last_monitored;
            """,
            zone.id,
            zone.name,
            zone.center.to_wkt(),
            zone.radius_km,
            zone.risk_level.value,
            zone.population_density,
            zone.elevation,
            zone.drainage_capacity,
            zone.created_at,
            zone.last_monitored
        )
