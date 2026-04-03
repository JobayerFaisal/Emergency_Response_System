"""
Data Collectors for Environmental Intelligence Agent (Weather-Only)
===================================================================
Asynchronous data collection from OpenWeatherMap API.

Author: Environmental Intelligence Team
Version: 2.0.3 (Weather Only, UUID safe + no circular reference)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from redis import asyncio as aioredis
import json
from uuid import UUID

from app.agents.agent_1_environmental.models import (
    WeatherData, WeatherMetrics, PrecipitationData, WeatherCondition,
    GeoPoint, SentinelZone, DataSource
)

logger = logging.getLogger(__name__)


# =====================================================================
# JSON HELPERS
# =====================================================================

def encode_json(obj):
    """UUID → str support for Redis."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# =====================================================================
# WEATHER API COLLECTOR
# =====================================================================

class WeatherAPICollector:
    """
    Collects weather data from OpenWeatherMap API.
    Handles:
        - Safe caching
        - Rate limiting
        - Retry + fallback
    """

    def __init__(
        self,
        api_key: str,
        cache_client: Optional[aioredis.Redis] = None,
        cache_ttl: int = 600
    ):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl
        self.timeout = ClientTimeout(total=30)

        # Local rate limit tracking
        self.max_calls_per_minute = 60
        self.call_timestamps: List[datetime] = []

        logger.info("WeatherAPICollector initialized")

    # =====================================================================
    # Rate Limiting
    # =====================================================================

    async def _check_rate_limit(self):
        now = datetime.utcnow()
        self.call_timestamps = [
            ts for ts in self.call_timestamps
            if (now - ts).total_seconds() < 60
        ]

        if len(self.call_timestamps) >= self.max_calls_per_minute:
            wait = 60 - (now - self.call_timestamps[0]).total_seconds()
            logger.warning(f"API rate limit hit — waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        self.call_timestamps.append(now)

    # =====================================================================
    # Cache GET
    # =====================================================================

    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_client:
            return None
        try:
            data = await self.cache_client.get(key)
            if data:
                try:
                    return json.loads(data)
                except Exception:
                    logger.error(f"[Redis] Corrupted JSON: {data}")
                    return None
        except Exception as e:
            logger.error(f"[Redis] Get error: {e}")
        return None

    # =====================================================================
    # Cache SET (NO circular references)
    # =====================================================================

    async def _set_cache(self, key: str, weather: WeatherData):
        """
        Store only the *safe* parts of WeatherData.
        Do NOT store raw_data into cache — it creates circular references.
        """
        if not self.cache_client:
            return

        try:
            cache_payload = {
                "id": str(weather.id),
                "timestamp": weather.timestamp.isoformat(),
                "location": weather.location.model_dump(),
                "condition": weather.condition.value,
                "metrics": weather.metrics.model_dump(),
                "precipitation": weather.precipitation.model_dump(),
                "description": weather.description,
                "source": weather.source.value,
            }

            await self.cache_client.setex(
                key,
                self.cache_ttl,
                json.dumps(cache_payload, default=encode_json)
            )

        except Exception as e:
            logger.error(f"[Redis] Cache error: {e}")

    # =====================================================================
    # Weather Condition Mapping
    # =====================================================================

    def _map_condition(self, weather_id: int, main: str) -> WeatherCondition:
        if 200 <= weather_id < 300:
            return WeatherCondition.THUNDERSTORM
        elif 300 <= weather_id < 400:
            return WeatherCondition.DRIZZLE
        elif 500 <= weather_id < 600:
            return WeatherCondition.HEAVY_RAIN if weather_id >= 502 else WeatherCondition.RAIN
        elif 600 <= weather_id < 700:
            return WeatherCondition.SNOW
        elif 700 <= weather_id < 800:
            if "fog" in main.lower():
                return WeatherCondition.FOG
            return WeatherCondition.MIST
        elif weather_id == 800:
            return WeatherCondition.CLEAR
        return WeatherCondition.CLOUDS

    # =====================================================================
    # Fetch Current Weather
    # =====================================================================

    async def fetch_current_weather(
        self,
        location: GeoPoint,
        zone_id: Optional[str] = None
    ) -> Optional[WeatherData]:

        cache_key = f"weather:{location.latitude}:{location.longitude}"
        cached = await self._get_from_cache(cache_key)

        if cached:
            try:
                return WeatherData(**cached)
            except Exception as e:
                logger.error(f"Failed to reconstruct cached WeatherData: {e}")

        # Make API call with rate limit control
        await self._check_rate_limit()

        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            async with ClientSession(timeout=self.timeout) as s:
                async with s.get(f"{self.base_url}/weather", params=params) as r:

                    if r.status != 200:
                        logger.error(f"Weather API error: HTTP {r.status}")
                        return None

                    data = await r.json()
                    weather = self._parse_weather_response(data, location)

                    # Cache FIXED (safe minimal payload)
                    await self._set_cache(cache_key, weather)

                    return weather

        except Exception as e:
            logger.error(f"Weather API fetch error: {e}")
            return None

    # =====================================================================
    # Parse OpenWeather API JSON → WeatherData
    # =====================================================================

    def _parse_weather_response(self, data: Dict[str, Any], location: GeoPoint) -> WeatherData:

        main = data.get("main", {})
        weather_raw = data.get("weather", [{}])[0]

        condition = self._map_condition(
            weather_raw.get("id", 800),
            weather_raw.get("main", "Clear")
        )

        metrics = WeatherMetrics(
            temperature=main.get("temp", 0.0),
            feels_like=main.get("feels_like", 0.0),
            humidity=main.get("humidity", 0.0),
            pressure=main.get("pressure", 0.0),
            wind_speed=data.get("wind", {}).get("speed", 0.0),
            wind_direction=data.get("wind", {}).get("deg"),
            visibility=data.get("visibility"),
            cloud_coverage=data.get("clouds", {}).get("all", 0.0)
        )

        rain = data.get("rain", {})
        snow = data.get("snow", {})

        precipitation = PrecipitationData(
            rain_1h=rain.get("1h"),
            rain_3h=rain.get("3h"),
            rain_24h=rain.get("24h"),
            snow_1h=snow.get("1h"),
            snow_3h=snow.get("3h"),
            intensity=(rain.get("1h") or 0.0)
        )

        return WeatherData(
            location=location,
            timestamp=datetime.utcfromtimestamp(
                data.get("dt", datetime.utcnow().timestamp())
            ),
            condition=condition,
            metrics=metrics,
            precipitation=precipitation,
            description=weather_raw.get("description", "Unknown"),
            source=DataSource.OPENWEATHERMAP,
            raw_data=data    # Stored in DB, NOT cached
        )

    # =====================================================================
    # Optional Forecast
    # =====================================================================

    async def fetch_forecast(self, location: GeoPoint, hours: int = 48) -> List[WeatherData]:
        """Forecast disabled for weather-only mode."""
        return []


# =====================================================================
# ORCHESTRATOR (Weather-only)
# =====================================================================

class DataCollectionOrchestrator:
    def __init__(self, weather_collector: WeatherAPICollector):
        self.weather_collector = weather_collector

        self.polling_intervals = {
            "minimal": 1800,
            "low": 900,
            "moderate": 300,
            "high": 180,
            "critical": 60,
        }

        logger.info("DataCollectionOrchestrator initialized (WEATHER ONLY)")

    def get_polling_interval(self, zone: SentinelZone) -> int:
        return self.polling_intervals.get(zone.risk_level.value, 300)

    async def collect_zone_data(self, zone: SentinelZone) -> Dict[str, Any]:
        weather = await self.weather_collector.fetch_current_weather(zone.center)

        return {
            "zone": zone,
            "weather": weather,
            "forecast": [],
            "social_posts": [],
            "collected_at": datetime.utcnow(),
        }

    async def collect_all_zones(self, zones: List[SentinelZone]) -> List[Dict[str, Any]]:
        tasks = [self.collect_zone_data(z) for z in zones]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final = [r for r in results if isinstance(r, dict)]
        logger.info(f"Weather-only data collection done: {len(final)}/{len(zones)} zones")

        return final
