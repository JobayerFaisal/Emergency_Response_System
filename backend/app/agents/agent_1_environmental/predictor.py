"""
Flood Predictor (WEATHER ONLY)
==============================

Weather + spatial + historical risk model.
Social media has been fully removed.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple

from app.agents.agent_1_environmental.models import (
    FloodPrediction, FloodRiskFactors, SentinelZone,
    SeverityLevel, AlertType, EnvironmentalAlert,
    WeatherData, SpatialAnalysisResult
)

logger = logging.getLogger(__name__)


# =====================================================================
# WEATHER-ONLY FLOOD RISK PREDICTOR
# =====================================================================

class FloodRiskPredictor:
    """
    Predicts flood risk using:
    - rainfall intensity
    - accumulated rainfall
    - weather severity
    - elevation
    - drainage capacity
    - satellite flood probability
    - historical risk
    - spatial analysis (affected area, severity)
    """

    def __init__(self):
        self.weights = {
            "rainfall_intensity": 0.30,
            "accumulated_rainfall": 0.25,
            "weather_severity": 0.20,
            "historical_risk": 0.10,
            "drainage_factor": 0.10,
            "elevation_factor": 0.05,
        }

        logger.info("FloodRiskPredictor (weather-only) initialized")

    # ---------------- WEATHER FACTORS ----------------

    def calculate_rainfall_intensity_factor(self, weather: Dict[str, float]) -> float:
        return weather.get("rainfall_intensity", 0.0)

    def calculate_accumulated_rainfall_factor(self, weather: Dict[str, float]) -> float:
        return weather.get("accumulated_rainfall", 0.0)

    def calculate_weather_severity_factor(self, weather: Dict[str, float]) -> float:
        return weather.get("weather_severity", 0.0)

    # ---------------- ZONE FACTORS ----------------

    def calculate_drainage_factor(self, zone: SentinelZone) -> float:
        mapping = {"excellent": 0.1, "good": 0.3, "moderate": 0.5, "poor": 0.8, "very_poor": 1.0}
        return mapping.get(str(zone.drainage_capacity).lower(), 0.5)

    def calculate_elevation_factor(self, zone: SentinelZone) -> float:
        if zone.elevation is None:
            return 0.5

        if zone.elevation < 5: return 1.0
        if zone.elevation < 10: return 0.7
        if zone.elevation < 20: return 0.4
        if zone.elevation < 30: return 0.2
        return 0.1

    # ---------------- MASTER RISK FACTOR BUILDER ----------------

    def calculate_risk_factors(
        self,
        zone: SentinelZone,
        normalized_weather: Optional[Dict[str, float]],
        historical_risk: float
    ) -> FloodRiskFactors:

        if not normalized_weather:
            normalized_weather = {
                "rainfall_intensity": 0.0,
                "accumulated_rainfall": 0.0,
                "weather_severity": 0.0,
            }

        return FloodRiskFactors(
            rainfall_intensity=self.calculate_rainfall_intensity_factor(normalized_weather),
            accumulated_rainfall=self.calculate_accumulated_rainfall_factor(normalized_weather),
            weather_severity=self.calculate_weather_severity_factor(normalized_weather),
            social_reports_density=0.0,  # Removed social completely
            historical_risk=historical_risk,
            drainage_factor=self.calculate_drainage_factor(zone),
            elevation_factor=self.calculate_elevation_factor(zone)
        )

    # ---------------- CONFIDENCE ----------------

    def calculate_confidence(self, has_weather: bool, weather_age_hours: float) -> float:
        if not has_weather:
            return 0.3

        if weather_age_hours < 1:
            return 1.0
        if weather_age_hours < 3:
            return 0.9
        if weather_age_hours < 6:
            return 0.7
        return 0.5

    # ---------------- IMPACT TIME ----------------

    def estimate_time_to_impact(self, severity: SeverityLevel, rainfall_intensity: float) -> Optional[float]:
        base = {
            SeverityLevel.MINIMAL: None,
            SeverityLevel.LOW: 12,
            SeverityLevel.MODERATE: 6,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 1
        }[severity]

        if base is None:
            return None

        if rainfall_intensity > 0.8:
            base *= 0.5
        elif rainfall_intensity > 0.6:
            base *= 0.75

        return max(base, 0.25)

    # ---------------- ACTIONS ----------------

    def generate_actions(
        self,
        severity: SeverityLevel,
        time_to_impact: Optional[float],
        affected_area_km2: float,
        infra: List[str]
    ) -> List[str]:

        actions = []

        if severity == SeverityLevel.CRITICAL:
            actions += [
                "Activate emergency response teams",
                "Issue evacuation orders",
                "Deploy boats & rescue units",
                "Open temporary shelters"
            ]
        elif severity == SeverityLevel.HIGH:
            actions += [
                "Warn residents in low-lying areas",
                "Keep evacuation routes ready",
                "Deploy response units to standby"
            ]
        elif severity == SeverityLevel.MODERATE:
            actions += [
                "Issue flood watch advisory",
                "Monitor drainage systems",
                "Prepare emergency assets"
            ]

        if time_to_impact and time_to_impact < 2:
            actions.insert(0, f"URGENT: Flood expected in {time_to_impact:.1f} hours")

        if "hospitals" in infra:
            actions.append("Alert hospitals for possible evacuation or surge")

        if affected_area_km2 > 8:
            actions.append("Coordinate district-wide mitigation efforts")

        return actions

    # ---------------- PREDICTION ----------------

    def predict_flood_risk(
        self,
        zone: SentinelZone,
        weather_data: Optional[WeatherData],
        normalized_weather: Optional[Dict[str, float]],
        spatial: SpatialAnalysisResult,
        historical_risk: float
    ) -> FloodPrediction:

        logger.info(f"Predicting weather-only flood risk for zone: {zone.name}")

        risk_factors = self.calculate_risk_factors(
            zone=zone,
            normalized_weather=normalized_weather,
            historical_risk=historical_risk
        )

        risk_score = risk_factors.weighted_score
        severity = FloodPrediction._risk_to_severity(risk_score)

        weather_age_hours = (
            (datetime.utcnow() - weather_data.timestamp).total_seconds() / 3600
            if weather_data else 999
        )

        confidence = self.calculate_confidence(
            has_weather=weather_data is not None,
            weather_age_hours=weather_age_hours
        )

        time_to_impact = self.estimate_time_to_impact(
            severity,
            risk_factors.rainfall_intensity
        )

        actions = self.generate_actions(
            severity,
            time_to_impact,
            spatial.affected_area_km2,
            spatial.critical_infrastructure_at_risk
        )

        prediction = FloodPrediction(
            zone=zone,
            timestamp=datetime.utcnow(),
            risk_score=risk_score,
            severity_level=severity,
            confidence=confidence,
            risk_factors=risk_factors,
            time_to_impact_hours=time_to_impact,
            affected_area_km2=spatial.affected_area_km2,
            estimated_affected_population=spatial.affected_population_estimate,
            recommended_actions=actions,
            alert_level=AlertType.FLOOD_RISK if severity >= SeverityLevel.MODERATE else AlertType.ALL_CLEAR
        )

        return prediction
    

# =====================================================================
# WEATHER-ONLY ALERT GENERATOR
# =====================================================================

class AlertGenerator:
    """Generates simple alerts for weather-only predictions."""

    TEMPLATES = {
        SeverityLevel.CRITICAL:
            "🚨 CRITICAL FLOOD RISK in {zone}! Severe weather and high water accumulation expected.",
        SeverityLevel.HIGH:
            "⚠️ HIGH FLOOD RISK in {zone}! Heavy rainfall and poor drainage detected.",
        SeverityLevel.MODERATE:
            "⚡ MODERATE FLOOD RISK in {zone}. Monitor water levels and weather updates.",
        SeverityLevel.LOW:
            "ℹ️ LOW flood risk in {zone}. Minor issues possible.",
        SeverityLevel.MINIMAL:
            "✅ MINIMAL risk in {zone}. No flood concern.",
    }

    def generate_alert(self, prediction: FloodPrediction) -> EnvironmentalAlert:

        message = self.TEMPLATES[prediction.severity_level].format(
            zone=prediction.zone.name
        )

        return EnvironmentalAlert(
            alert_type=prediction.alert_level,
            severity=prediction.severity_level,
            zone=prediction.zone,
            prediction=prediction,
            message=message,
            priority={
                SeverityLevel.CRITICAL: 5,
                SeverityLevel.HIGH: 4,
                SeverityLevel.MODERATE: 3,
                SeverityLevel.LOW: 2,
                SeverityLevel.MINIMAL: 1,
            }[prediction.severity_level],
        )


# =====================================================================
# WEATHER-ONLY PREDICTION ORCHESTRATOR
# =====================================================================

class PredictionOrchestrator:
    """Runs predictions for each zone, generates alerts."""

    def __init__(self, predictor: FloodRiskPredictor, alert_gen: AlertGenerator):
        self.predictor = predictor
        self.alert_gen = alert_gen

    async def predict_for_zone(
        self,
        processed: Dict[str, Any],
        historical_risk: float
    ) -> Tuple[FloodPrediction, Optional[EnvironmentalAlert]]:

        zone = processed["zone"]
        weather = processed.get("weather")
        normalized = processed.get("normalized_weather")
        spatial = processed["spatial_analysis"]

        prediction = self.predictor.predict_flood_risk(
            zone=zone,
            weather_data=weather,
            normalized_weather=normalized,
            spatial=spatial,
            historical_risk=historical_risk
        )

        alert = None
        if prediction.severity_level != SeverityLevel.MINIMAL:
            alert = self.alert_gen.generate_alert(prediction)

        return prediction, alert

    async def predict_all_zones(
        self,
        processed_list: List[Dict[str, Any]],
        historical_risk_map: Dict[str, float]
    ) -> Tuple[List[FloodPrediction], List[EnvironmentalAlert]]:

        predictions = []
        alerts = []

        for data in processed_list:
            zone_id = str(data["zone"].id)
            hist = historical_risk_map.get(zone_id, 0.0)

            pred, alert = await self.predict_for_zone(data, hist)
            predictions.append(pred)
            if alert:
                alerts.append(alert)

        return predictions, alerts
