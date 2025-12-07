"""
Flood Predictor for Environmental Intelligence Agent
====================================================
Multi-factor flood risk prediction and alert generation.
Combines weather data, social media reports, and spatial analysis.

Author: Environmental Intelligence Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from models import (
    FloodPrediction, FloodRiskFactors, SentinelZone,
    SeverityLevel, AlertType, EnvironmentalAlert,
    WeatherData, EnrichedSocialPost, SpatialAnalysisResult
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================================================
# FLOOD RISK PREDICTOR
# =====================================================================

class FloodRiskPredictor:
    """
    Predicts flood risk using multi-factor analysis.
    Combines weather patterns, social media signals, and spatial data.
    """
    
    def __init__(self):
        """Initialize flood risk predictor"""
        # Risk factor weights (tuned for Bangladesh urban flooding)
        self.weights = {
            'rainfall_intensity': 0.25,
            'accumulated_rainfall': 0.20,
            'weather_severity': 0.15,
            'social_reports_density': 0.15,
            'historical_risk': 0.10,
            'drainage_factor': 0.10,
            'elevation_factor': 0.05
        }
        
        # Confidence adjustment factors
        self.confidence_factors = {
            'data_completeness': 0.3,
            'data_freshness': 0.2,
            'social_credibility': 0.2,
            'historical_accuracy': 0.15,
            'spatial_coherence': 0.15
        }
        
        # Time-to-impact estimation (hours)
        self.impact_time_estimates = {
            SeverityLevel.MINIMAL: None,
            SeverityLevel.LOW: 12.0,
            SeverityLevel.MODERATE: 6.0,
            SeverityLevel.HIGH: 3.0,
            SeverityLevel.CRITICAL: 1.0
        }
        
        logger.info("FloodRiskPredictor initialized")
    
    def calculate_rainfall_intensity_factor(
        self,
        normalized_weather: Dict[str, float]
    ) -> float:
        """
        Calculate rainfall intensity risk factor.
        
        Args:
            normalized_weather: Normalized weather metrics
            
        Returns:
            Risk factor (0-1)
        """
        return normalized_weather.get('rainfall_intensity', 0.0)
    
    def calculate_accumulated_rainfall_factor(
        self,
        normalized_weather: Dict[str, float]
    ) -> float:
        """
        Calculate accumulated rainfall risk factor.
        
        Args:
            normalized_weather: Normalized weather metrics
            
        Returns:
            Risk factor (0-1)
        """
        return normalized_weather.get('accumulated_rainfall', 0.0)
    
    def calculate_weather_severity_factor(
        self,
        normalized_weather: Dict[str, float]
    ) -> float:
        """
        Calculate overall weather severity factor.
        
        Args:
            normalized_weather: Normalized weather metrics
            
        Returns:
            Risk factor (0-1)
        """
        return normalized_weather.get('weather_severity', 0.0)
    
    def calculate_social_reports_factor(
        self,
        social_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate social media reports density factor.
        
        Args:
            social_analysis: Social media analysis results
            
        Returns:
            Risk factor (0-1)
        """
        report_density = social_analysis.get('report_density', 0.0)
        urgency_score = social_analysis.get('urgency_score', 0.0)
        
        # Combine density and urgency
        factor = (report_density * 0.6 + urgency_score * 0.4)
        
        return min(factor, 1.0)
    
    def calculate_drainage_factor(self, zone: SentinelZone) -> float:
        """
        Calculate drainage capacity factor (1=poor, 0=excellent).
        
        Args:
            zone: Sentinel zone
            
        Returns:
            Risk factor (0-1)
        """
        if not zone.drainage_capacity:
            return 0.5  # Unknown = moderate risk
        
        drainage_scores = {
            'excellent': 0.1,
            'good': 0.3,
            'moderate': 0.5,
            'poor': 0.8,
            'very_poor': 1.0
        }
        
        return drainage_scores.get(
            zone.drainage_capacity.lower(),
            0.5
        )
    
    def calculate_elevation_factor(self, zone: SentinelZone) -> float:
        """
        Calculate elevation risk factor (1=low elevation, 0=high).
        
        Args:
            zone: Sentinel zone
            
        Returns:
            Risk factor (0-1)
        """
        if zone.elevation is None:
            return 0.5  # Unknown = moderate risk
        
        # Bangladesh typical elevation range: 0-50m
        # Higher elevation = lower flood risk
        if zone.elevation < 5:  # Very low
            return 1.0
        elif zone.elevation < 10:  # Low
            return 0.7
        elif zone.elevation < 20:  # Moderate
            return 0.4
        elif zone.elevation < 30:  # Higher
            return 0.2
        else:  # High ground
            return 0.1
    
    def calculate_risk_factors(
        self,
        zone: SentinelZone,
        normalized_weather: Optional[Dict[str, float]],
        social_analysis: Dict[str, Any],
        historical_risk: float
    ) -> FloodRiskFactors:
        """
        Calculate all risk factors.
        
        Args:
            zone: Sentinel zone
            normalized_weather: Normalized weather metrics
            social_analysis: Social media analysis
            historical_risk: Historical risk score
            
        Returns:
            Complete risk factors
        """
        # Handle missing weather data
        if not normalized_weather:
            normalized_weather = {
                'rainfall_intensity': 0.0,
                'accumulated_rainfall': 0.0,
                'weather_severity': 0.0
            }
        
        return FloodRiskFactors(
            rainfall_intensity=self.calculate_rainfall_intensity_factor(
                normalized_weather
            ),
            accumulated_rainfall=self.calculate_accumulated_rainfall_factor(
                normalized_weather
            ),
            weather_severity=self.calculate_weather_severity_factor(
                normalized_weather
            ),
            social_reports_density=self.calculate_social_reports_factor(
                social_analysis
            ),
            historical_risk=historical_risk,
            drainage_factor=self.calculate_drainage_factor(zone),
            elevation_factor=self.calculate_elevation_factor(zone)
        )
    
    def calculate_confidence(
        self,
        has_weather: bool,
        weather_age_hours: float,
        social_credibility: float,
        social_posts_count: int,
        spatial_coherence: float
    ) -> float:
        """
        Calculate prediction confidence.
        
        Args:
            has_weather: Whether weather data is available
            weather_age_hours: Age of weather data in hours
            social_credibility: Average credibility of social posts
            social_posts_count: Number of social posts analyzed
            spatial_coherence: Spatial analysis coherence score
            
        Returns:
            Confidence score (0-1)
        """
        # Data completeness
        completeness = 1.0 if has_weather else 0.3
        if social_posts_count > 0:
            completeness = min(completeness + 0.2, 1.0)
        
        # Data freshness (weather)
        freshness = 1.0
        if has_weather:
            if weather_age_hours < 1:
                freshness = 1.0
            elif weather_age_hours < 3:
                freshness = 0.9
            elif weather_age_hours < 6:
                freshness = 0.7
            else:
                freshness = 0.5
        else:
            freshness = 0.3
        
        # Social credibility
        social_conf = social_credibility if social_posts_count > 0 else 0.5
        
        # Historical accuracy (placeholder - would track actual accuracy)
        historical_accuracy = 0.75
        
        # Weighted combination
        confidence = (
            completeness * self.confidence_factors['data_completeness'] +
            freshness * self.confidence_factors['data_freshness'] +
            social_conf * self.confidence_factors['social_credibility'] +
            historical_accuracy * self.confidence_factors['historical_accuracy'] +
            spatial_coherence * self.confidence_factors['spatial_coherence']
        )
        
        return min(confidence, 1.0)
    
    def estimate_time_to_impact(
        self,
        severity: SeverityLevel,
        rainfall_intensity: float,
        social_urgency: float
    ) -> Optional[float]:
        """
        Estimate time until flood impact.
        
        Args:
            severity: Predicted severity level
            rainfall_intensity: Normalized rainfall intensity
            social_urgency: Social media urgency score
            
        Returns:
            Hours until impact or None
        """
        base_time = self.impact_time_estimates.get(severity)
        
        if base_time is None:
            return None
        
        # Adjust based on intensity
        if rainfall_intensity > 0.8:
            base_time *= 0.5  # Halve time for extreme rainfall
        elif rainfall_intensity > 0.6:
            base_time *= 0.75
        
        # Adjust if social media shows flooding already occurring
        if social_urgency > 0.7:
            base_time = min(base_time, 0.5)  # Already happening or imminent
        
        return max(base_time, 0.25)  # Minimum 15 minutes
    
    def generate_recommended_actions(
        self,
        severity: SeverityLevel,
        time_to_impact: Optional[float],
        affected_area_km2: float,
        critical_infrastructure: List[str]
    ) -> List[str]:
        """
        Generate recommended actions based on prediction.
        
        Args:
            severity: Severity level
            time_to_impact: Hours until impact
            affected_area_km2: Affected area
            critical_infrastructure: Infrastructure at risk
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # Severity-based actions
        if severity == SeverityLevel.CRITICAL:
            actions.extend([
                "IMMEDIATE: Activate emergency response teams",
                "IMMEDIATE: Issue evacuation orders for affected areas",
                "Deploy rescue boats and emergency equipment",
                "Establish emergency shelters",
                "Alert hospitals and emergency services"
            ])
        
        elif severity == SeverityLevel.HIGH:
            actions.extend([
                "Alert emergency response teams to standby",
                "Prepare evacuation routes and shelters",
                "Issue public warnings via all channels",
                "Pre-position rescue equipment",
                "Monitor situation closely (every 30 minutes)"
            ])
        
        elif severity == SeverityLevel.MODERATE:
            actions.extend([
                "Issue flood watch advisory",
                "Alert relevant authorities",
                "Prepare emergency response resources",
                "Monitor situation (every hour)",
                "Inform potentially affected residents"
            ])
        
        elif severity == SeverityLevel.LOW:
            actions.extend([
                "Monitor weather conditions",
                "Inform local authorities",
                "Keep emergency services on standby"
            ])
        
        # Time-sensitive actions
        if time_to_impact and time_to_impact < 2:
            actions.insert(0, f"URGENT: Impact expected in {time_to_impact:.1f} hours")
        
        # Area-specific actions
        if affected_area_km2 > 10:
            actions.append("Large area affected: coordinate multi-district response")
        
        # Infrastructure-specific actions
        if "hospitals" in critical_infrastructure:
            actions.append("Alert hospitals: prepare for patient influx/evacuation")
        
        if "schools" in critical_infrastructure:
            actions.append("Consider school closures in affected areas")
        
        if "emergency_services" in critical_infrastructure:
            actions.append("Relocate emergency services if necessary")
        
        return actions
    
    def predict_flood_risk(
        self,
        zone: SentinelZone,
        weather_data: Optional[WeatherData],
        normalized_weather: Optional[Dict[str, float]],
        social_analysis: Dict[str, Any],
        spatial_analysis: SpatialAnalysisResult,
        historical_risk: float
    ) -> FloodPrediction:
        """
        Generate comprehensive flood risk prediction.
        
        Args:
            zone: Sentinel zone
            weather_data: Raw weather data
            normalized_weather: Normalized weather metrics
            social_analysis: Social media analysis
            spatial_analysis: Spatial analysis results
            historical_risk: Historical risk score
            
        Returns:
            Complete flood prediction
        """
        logger.info(f"Generating prediction for zone: {zone.name}")
        
        # Calculate risk factors
        risk_factors = self.calculate_risk_factors(
            zone,
            normalized_weather,
            social_analysis,
            historical_risk
        )
        
        # Calculate overall risk score
        risk_score = risk_factors.weighted_score
        
        # Determine severity level
        severity = FloodPrediction._risk_to_severity(risk_score)
        
        # Calculate confidence
        weather_age = 0.0
        if weather_data:
            weather_age = (
                datetime.utcnow() - weather_data.timestamp
            ).total_seconds() / 3600
        
        confidence = self.calculate_confidence(
            has_weather=weather_data is not None,
            weather_age_hours=weather_age,
            social_credibility=social_analysis.get('average_credibility', 0.5),
            social_posts_count=social_analysis.get('relevant_posts', 0),
            spatial_coherence=spatial_analysis.average_severity
        )
        
        # Estimate time to impact
        time_to_impact = self.estimate_time_to_impact(
            severity,
            risk_factors.rainfall_intensity,
            social_analysis.get('urgency_score', 0.0)
        )
        
        # Generate recommended actions
        recommended_actions = self.generate_recommended_actions(
            severity,
            time_to_impact,
            spatial_analysis.affected_area_km2,
            spatial_analysis.critical_infrastructure_at_risk
        )
        
        # Determine alert type
        if severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            alert_level = AlertType.FLOOD_RISK
        elif severity == SeverityLevel.MODERATE:
            alert_level = AlertType.WEATHER_WARNING
        else:
            alert_level = AlertType.ALL_CLEAR
        
        prediction = FloodPrediction(
            zone=zone,
            timestamp=datetime.utcnow(),
            risk_score=risk_score,
            severity_level=severity,
            confidence=confidence,
            risk_factors=risk_factors,
            time_to_impact_hours=time_to_impact,
            affected_area_km2=spatial_analysis.affected_area_km2,
            estimated_affected_population=spatial_analysis.affected_population_estimate,
            recommended_actions=recommended_actions,
            alert_level=alert_level
        )
        
        logger.info(
            f"Prediction complete: {zone.name} - "
            f"Risk={risk_score:.2f}, Severity={severity.value}, "
            f"Confidence={confidence:.2f}"
        )
        
        return prediction


# =====================================================================
# ALERT GENERATOR
# =====================================================================

class AlertGenerator:
    """
    Generates environmental alerts from predictions.
    Formats alerts for consumption by other agents.
    """
    
    def __init__(self):
        """Initialize alert generator"""
        self.alert_templates = {
            SeverityLevel.CRITICAL: (
                "🚨 CRITICAL FLOOD ALERT: {zone_name}\n"
                "Immediate flooding expected in {time} hours.\n"
                "Risk Score: {risk:.0%} | Confidence: {confidence:.0%}\n"
                "Affected Area: {area:.1f} km²\n"
                "Estimated Population: {population:,}\n"
                "IMMEDIATE ACTION REQUIRED"
            ),
            SeverityLevel.HIGH: (
                "⚠️ HIGH FLOOD RISK: {zone_name}\n"
                "Significant flooding likely in {time} hours.\n"
                "Risk Score: {risk:.0%} | Confidence: {confidence:.0%}\n"
                "Affected Area: {area:.1f} km²\n"
                "Prepare emergency response"
            ),
            SeverityLevel.MODERATE: (
                "⚡ MODERATE FLOOD RISK: {zone_name}\n"
                "Flooding possible in {time} hours.\n"
                "Risk Score: {risk:.0%} | Confidence: {confidence:.0%}\n"
                "Monitor situation closely"
            ),
            SeverityLevel.LOW: (
                "ℹ️ LOW FLOOD RISK: {zone_name}\n"
                "Minor flooding possible.\n"
                "Risk Score: {risk:.0%} | Confidence: {confidence:.0%}\n"
                "Continue monitoring"
            ),
            SeverityLevel.MINIMAL: (
                "✅ MINIMAL RISK: {zone_name}\n"
                "No significant flooding expected.\n"
                "Risk Score: {risk:.0%}\n"
                "Normal conditions"
            )
        }
        
        logger.info("AlertGenerator initialized")
    
    def generate_alert(self, prediction: FloodPrediction) -> EnvironmentalAlert:
        """
        Generate an environmental alert from a prediction.
        
        Args:
            prediction: Flood prediction
            
        Returns:
            Environmental alert
        """
        # Format time to impact
        time_str = "unknown"
        if prediction.time_to_impact_hours:
            if prediction.time_to_impact_hours < 1:
                time_str = f"{prediction.time_to_impact_hours * 60:.0f} minutes"
            else:
                time_str = f"{prediction.time_to_impact_hours:.1f}"
        
        # Format population
        population_str = (
            f"{prediction.estimated_affected_population:,}"
            if prediction.estimated_affected_population
            else "unknown"
        )
        
        # Format message
        template = self.alert_templates[prediction.severity_level]
        message = template.format(
            zone_name=prediction.zone.name,
            time=time_str,
            risk=prediction.risk_score,
            confidence=prediction.confidence,
            area=prediction.affected_area_km2,
            population=population_str
        )
        
        return EnvironmentalAlert(
            alert_type=prediction.alert_level,
            severity=prediction.severity_level,
            zone=prediction.zone,
            prediction=prediction,
            message=message,
            priority=self._calculate_priority(prediction)
        )
    
    def _calculate_priority(self, prediction: FloodPrediction) -> int:
        """Calculate alert priority (1-5)"""
        base_priority = {
            SeverityLevel.MINIMAL: 1,
            SeverityLevel.LOW: 2,
            SeverityLevel.MODERATE: 3,
            SeverityLevel.HIGH: 4,
            SeverityLevel.CRITICAL: 5
        }[prediction.severity_level]
        
        # Boost priority if impact is imminent
        if (prediction.time_to_impact_hours and 
            prediction.time_to_impact_hours < 2):
            base_priority = 5
        
        return base_priority
    
    def generate_alerts_batch(
        self,
        predictions: List[FloodPrediction]
    ) -> List[EnvironmentalAlert]:
        """
        Generate alerts for multiple predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            List of alerts
        """
        alerts = [
            self.generate_alert(pred)
            for pred in predictions
            if pred.severity_level != SeverityLevel.MINIMAL
        ]
        
        # Sort by priority (highest first)
        alerts.sort(key=lambda a: a.priority, reverse=True)
        
        logger.info(f"Generated {len(alerts)} alerts from {len(predictions)} predictions")
        
        return alerts


# =====================================================================
# PREDICTION ORCHESTRATOR
# =====================================================================

class PredictionOrchestrator:
    """
    Orchestrates the prediction process for all zones.
    Coordinates predictor and alert generator.
    """
    
    def __init__(
        self,
        predictor: FloodRiskPredictor,
        alert_generator: AlertGenerator
    ):
        """
        Initialize orchestrator.
        
        Args:
            predictor: Flood risk predictor
            alert_generator: Alert generator
        """
        self.predictor = predictor
        self.alert_generator = alert_generator
        
        logger.info("PredictionOrchestrator initialized")
    
    async def predict_for_zone(
        self,
        processed_data: Dict[str, Any],
        historical_risk: float = 0.0
    ) -> Tuple[FloodPrediction, Optional[EnvironmentalAlert]]:
        """
        Generate prediction and alert for a zone.
        
        Args:
            processed_data: Processed zone data
            historical_risk: Historical risk score
            
        Returns:
            Tuple of (prediction, alert)
        """
        zone = processed_data['zone']
        weather = processed_data.get('weather')
        normalized_weather = processed_data.get('normalized_weather')
        social_analysis = processed_data.get('social_analysis', {})
        
        # Create minimal spatial analysis if not provided
        # (In full implementation, this comes from spatial_analyzer)
        spatial_analysis = SpatialAnalysisResult(
            zone=zone,
            timestamp=datetime.utcnow(),
            affected_area_km2=social_analysis.get('report_density', 0) * zone.radius_km ** 2,
            nearby_reports_count=social_analysis.get('flood_reports', 0),
            average_severity=social_analysis.get('urgency_score', 0.0),
            risk_clusters=[],
            affected_population_estimate=None,
            critical_infrastructure_at_risk=[]
        )
        
        # Generate prediction
        prediction = self.predictor.predict_flood_risk(
            zone=zone,
            weather_data=weather,
            normalized_weather=normalized_weather,
            social_analysis=social_analysis,
            spatial_analysis=spatial_analysis,
            historical_risk=historical_risk
        )
        
        # Generate alert if severity warrants it
        alert = None
        if prediction.severity_level != SeverityLevel.MINIMAL:
            alert = self.alert_generator.generate_alert(prediction)
        
        return prediction, alert
    
    async def predict_all_zones(
        self,
        processed_data_list: List[Dict[str, Any]],
        historical_risks: Optional[Dict[str, float]] = None
    ) -> Tuple[List[FloodPrediction], List[EnvironmentalAlert]]:
        """
        Generate predictions and alerts for all zones.
        
        Args:
            processed_data_list: List of processed zone data
            historical_risks: Dictionary mapping zone IDs to historical risk scores
            
        Returns:
            Tuple of (predictions, alerts)
        """
        if historical_risks is None:
            historical_risks = {}
        
        predictions = []
        alerts = []
        
        for processed_data in processed_data_list:
            zone_id = str(processed_data['zone'].id)
            historical_risk = historical_risks.get(zone_id, 0.0)
            
            prediction, alert = await self.predict_for_zone(
                processed_data,
                historical_risk
            )
            
            predictions.append(prediction)
            if alert:
                alerts.append(alert)
        
        logger.info(
            f"Generated {len(predictions)} predictions, "
            f"{len(alerts)} alerts for {len(processed_data_list)} zones"
        )
        
        return predictions, alerts