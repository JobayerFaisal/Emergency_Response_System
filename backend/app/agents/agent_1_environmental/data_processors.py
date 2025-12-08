"""
Data Processors for Environmental Intelligence Agent (Weather-Only)
===================================================================
Processes raw weather data into normalized metrics for analysis.

Author: Environmental Intelligence Team
Version: 2.0.0 (Weather Only)
"""



import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, cast

from app.agents.agent_1_environmental.models import (
    WeatherData, PrecipitationData, WeatherCondition
    )


logger = logging.getLogger(__name__)


# =====================================================================
# 🌧️ WEATHER DATA NORMALIZER  (UNCHANGED)
# =====================================================================

class WeatherDataNormalizer:
    """
    Normalizes weather data for consistent analysis.
    Converts measurements to normalized scales (0–1).
    """

    def __init__(self):
        self.reference_values = {
            "temperature": {"min": 15, "max": 40},
            "humidity": {"min": 40, "max": 100},
            "pressure": {"min": 980, "max": 1020},
            "wind_speed": {"min": 0, "max": 20},
            "rainfall_1h": {"min": 0, "max": 50},
            "rainfall_3h": {"min": 0, "max": 100},
            "rainfall_24h": {"min": 0, "max": 200},
            "cloud_coverage": {"min": 0, "max": 100},
        }

        logger.info("WeatherDataNormalizer initialized")

    def normalize_value(self, value: float, metric: str) -> float:
        if metric not in self.reference_values:
            return value

        ref = self.reference_values[metric]
        min_val, max_val = ref["min"], ref["max"]
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def calculate_rainfall_intensity(self, precip: PrecipitationData) -> float:
        if precip.rain_1h:
            return self.normalize_value(precip.rain_1h, "rainfall_1h")
        if precip.rain_3h:
            return self.normalize_value(precip.rain_3h / 3, "rainfall_1h")
        if precip.rain_24h:
            return self.normalize_value(precip.rain_24h / 24, "rainfall_1h")
        return 0.0

    def calculate_accumulated_rainfall(self, precip: PrecipitationData) -> float:
        total = precip.total_rain
        return self.normalize_value(total, "rainfall_24h")

    def calculate_weather_severity(self, weather: WeatherData) -> float:
        condition_weights = {
            WeatherCondition.CLEAR: 0.0,
            WeatherCondition.CLOUDS: 0.1,
            WeatherCondition.MIST: 0.2,
            WeatherCondition.FOG: 0.2,
            WeatherCondition.DRIZZLE: 0.3,
            WeatherCondition.RAIN: 0.5,
            WeatherCondition.HEAVY_RAIN: 0.8,
            WeatherCondition.THUNDERSTORM: 0.9,
            WeatherCondition.SNOW: 0.4,
        }

        cond_score = condition_weights.get(weather.condition, 0.5)
        rain_score = self.calculate_rainfall_intensity(weather.precipitation)
        wind_score = self.normalize_value(weather.metrics.wind_speed, "wind_speed")
        humidity_score = self.normalize_value(weather.metrics.humidity, "humidity")

        return min(
            1.0,
            cond_score * 0.4 +
            rain_score * 0.4 +
            wind_score * 0.1 +
            humidity_score * 0.1
        )

    def normalize_weather_data(self, weather: WeatherData) -> Dict[str, float]:
        return {
            "temperature": self.normalize_value(weather.metrics.temperature, "temperature"),
            "humidity": self.normalize_value(weather.metrics.humidity, "humidity"),
            "pressure": self.normalize_value(weather.metrics.pressure, "pressure"),
            "wind_speed": self.normalize_value(weather.metrics.wind_speed, "wind_speed"),
            "cloud_coverage": self.normalize_value(weather.metrics.cloud_coverage, "cloud_coverage"),
            "rainfall_intensity": self.calculate_rainfall_intensity(weather.precipitation),
            "accumulated_rainfall": self.calculate_accumulated_rainfall(weather.precipitation),
            "weather_severity": self.calculate_weather_severity(weather),
        }


# =====================================================================
# ❌ REMOVED BLOCKS:
#   - LLMEnrichmentProcessor
#   - SocialMediaAnalyzer
#   - EnrichedPost processing
# Everything below this line has been replaced with a WEATHER-ONLY orchestrator.
# =====================================================================


# =====================================================================
# 🌤️ WEATHER-ONLY DATA PROCESSING ORCHESTRATOR
# =====================================================================

class DataProcessingOrchestrator:
    """
    WEATHER-ONLY orchestrator.
    No social media enrichment, no LLM, no sentiment analysis.
    """

    def __init__(
        self,
        llm_processor: Optional[Any],   # Ignored
        weather_normalizer: WeatherDataNormalizer,
        social_analyzer: Optional[Any]  # Ignored
    ):
        self.weather_normalizer = weather_normalizer
        logger.info("DataProcessingOrchestrator initialized (WEATHER ONLY)")

    async def process_zone_data(
        self,
        zone_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        zone = zone_data["zone"]
        weather = zone_data.get("weather")

        logger.info(f"Processing WEATHER data for zone: {zone.name}")

        normalized = None
        if weather:
            normalized = self.weather_normalizer.normalize_weather_data(weather)

        return {
            "zone": zone,
            "weather": weather,
            "normalized_weather": normalized,
            "enriched_posts": [],       # always empty
            "social_analysis": {},      # removed
            "processed_at": datetime.utcnow(),
        }

    async def process_all_zones(
        self,
        collected_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        logger.info(f"Processing weather data for {len(collected_data)} zones")

        tasks = [
            self.process_zone_data(zone_data)
            for zone_data in collected_data
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, dict)]

        logger.info(f"Weather processing done for {len(valid)} zones")

        return valid
