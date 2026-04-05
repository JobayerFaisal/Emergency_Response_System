# backend/src/agents/agent_1_environmental/predictor.py



"""
Flood Predictor for Environmental Intelligence Agent
====================================================
Multi-factor flood risk prediction and alert generation.

8-Factor Risk Model:
  1. Rainfall intensity        — weather API (forecast signal)
  2. Accumulated rainfall      — weather API (forecast signal)
  3. Weather severity          — weather API (forecast signal)
  4. Satellite flood detection  — GEE SAR change detection (ground truth — highest weight)
  5. Flood depth estimate       — depth estimation CNN (severity amplifier)
  6. Drainage capacity          — zone metadata
  7. Elevation                  — zone metadata
  8. Social media reports       — OPTIONAL (included if available, zero-weight if not)

Override Logic:
  When satellite confirms active flooding (flood_percentage > threshold),
  the risk score is floored at HIGH severity regardless of weather data.
  Weather becomes "will it get worse?", satellite is "what's happening now".

Author: Environmental Intelligence Team
Version: 2.0.0
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.src.agents.agent_1_environmental.models import (
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
    
    Combines satellite observations, weather forecasts, zone characteristics,
    and optionally social media signals into a unified risk score.
    
    Satellite data (when available) carries the highest weight as it represents
    ground truth observations rather than predictions.
    """
    
    # --- Satellite flood detection thresholds ---
    # These map flood_percentage from SAR change detection to risk scores
    SATELLITE_FLOOD_THRESHOLDS = {
        'confirmed_flooding': 5.0,   # >= 5% flood = confirmed active flood
        'severe_flooding': 25.0,     # >= 25% = severe
        'critical_flooding': 50.0,   # >= 50% = critical
    }
    
    # --- Flood depth thresholds (meters) ---
    DEPTH_THRESHOLDS = {
        'nuisance': 0.3,       # < 0.3m = ankle-level, nuisance
        'moderate': 0.5,       # 0.3-0.5m = knee-level, movement impaired
        'dangerous': 1.0,      # 0.5-1.0m = waist-level, dangerous
        'life_threatening': 2.0 # > 2.0m = life-threatening
    }
    
    def __init__(self):
        """Initialize flood risk predictor with 8-factor model"""
        
        # Confidence adjustment factors (updated for satellite era)
        self.confidence_factors = {
            'data_completeness': 0.25,
            'data_freshness': 0.20,
            'satellite_confidence': 0.25,  # NEW — satellite data quality
            'social_credibility': 0.10,    # Reduced — optional source
            'historical_accuracy': 0.10,
            'spatial_coherence': 0.10
        }
        
        # Time-to-impact estimation (hours)
        self.impact_time_estimates = {
            SeverityLevel.MINIMAL: None,
            SeverityLevel.LOW: 12.0,
            SeverityLevel.MODERATE: 6.0,
            SeverityLevel.HIGH: 3.0,
            SeverityLevel.CRITICAL: 1.0
        }
        
        logger.info(
            "FloodRiskPredictor initialized — "
            "8-factor model with satellite override"
        )
    
    # -----------------------------------------------------------------
    # FACTOR CALCULATORS (each returns 0.0 – 1.0)
    # -----------------------------------------------------------------
    
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
    
    def calculate_satellite_flood_factor(
        self,
        satellite_flood_pct: float,
        satellite_risk: str
    ) -> float:
        """
        Calculate satellite flood detection risk factor.
        
        This is the strongest signal in the model. It converts the 
        flood_percentage from GEE SAR change detection into a 0-1 risk score.
        
        Mapping logic:
          0%   flood -> 0.0 (no flooding observed)
          5%   flood -> 0.4 (confirmed minor flooding)
          15%  flood -> 0.6 (significant flooding)
          25%  flood -> 0.8 (severe flooding)
          50%+ flood -> 1.0 (critical, catastrophic)
        
        Also factors in the risk_level string from satellite_service as a
        secondary cross-check.
        
        Args:
            satellite_flood_pct: Percentage of zone area detected as flooded (0-100)
            satellite_risk: Risk level string from satellite service 
                           (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)
            
        Returns:
            Risk factor (0-1)
        """
        pct = satellite_flood_pct
        
        # Piecewise linear mapping of flood percentage to risk score
        if pct <= 0.0:
            pct_score = 0.0
        elif pct < 5.0:
            # 0-5% -> 0.0-0.4 (early signs)
            pct_score = (pct / 5.0) * 0.4
        elif pct < 15.0:
            # 5-15% -> 0.4-0.6 (confirmed flooding, escalating)
            pct_score = 0.4 + ((pct - 5.0) / 10.0) * 0.2
        elif pct < 25.0:
            # 15-25% -> 0.6-0.8 (significant to severe)
            pct_score = 0.6 + ((pct - 15.0) / 10.0) * 0.2
        elif pct < 50.0:
            # 25-50% -> 0.8-0.95 (severe)
            pct_score = 0.8 + ((pct - 25.0) / 25.0) * 0.15
        else:
            # 50%+ -> 0.95-1.0 (catastrophic)
            pct_score = min(0.95 + ((pct - 50.0) / 50.0) * 0.05, 1.0)
        
        # Cross-check with risk_level string as secondary signal
        risk_level_scores = {
            'MINIMAL': 0.0,
            'LOW': 0.2,
            'MEDIUM': 0.5,
            'HIGH': 0.75,
            'CRITICAL': 1.0
        }
        risk_str_score = risk_level_scores.get(
            satellite_risk.upper() if satellite_risk else 'MINIMAL',
            0.0
        )
        
        # Primary weight on flood percentage (objective), secondary on risk string
        factor = pct_score * 0.8 + risk_str_score * 0.2
        
        return min(factor, 1.0)
    
    def calculate_flood_depth_factor(
        self,
        depth_data: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate flood depth risk factor from depth estimation CNN.
        
        Converts mean/max depth in meters to a 0-1 severity score.
        
        Depth mapping:
          0.0m    -> 0.0 (dry)
          0.3m    -> 0.3 (ankle-level, nuisance)
          0.5m    -> 0.5 (knee-level, movement impaired)
          1.0m    -> 0.7 (waist-level, dangerous)
          2.0m    -> 0.9 (life-threatening)
          3.0m+   -> 1.0 (catastrophic)
        
        Uses max_depth as primary signal (worst case) and mean_depth as 
        secondary (average conditions across flooded area).
        
        Args:
            depth_data: Dict with keys from DepthPredictor.analyze():
                - statistics.mean_depth_m
                - statistics.max_depth_m
                - statistics.flood_area_percent
                Or None if depth estimation unavailable/not run.
            
        Returns:
            Risk factor (0-1)
        """
        if not depth_data:
            return 0.0
        
        stats = depth_data.get('statistics', {})
        max_depth = stats.get('max_depth_m', 0.0)
        mean_depth = stats.get('mean_depth_m', 0.0)
        
        # Piecewise linear on max_depth (worst case scenario)
        if max_depth <= 0.1:
            max_score = 0.0
        elif max_depth < 0.3:
            max_score = (max_depth / 0.3) * 0.3
        elif max_depth < 0.5:
            max_score = 0.3 + ((max_depth - 0.3) / 0.2) * 0.2
        elif max_depth < 1.0:
            max_score = 0.5 + ((max_depth - 0.5) / 0.5) * 0.2
        elif max_depth < 2.0:
            max_score = 0.7 + ((max_depth - 1.0) / 1.0) * 0.2
        else:
            max_score = min(0.9 + ((max_depth - 2.0) / 1.0) * 0.1, 1.0)
        
        # Mean depth as secondary signal
        if mean_depth <= 0.1:
            mean_score = 0.0
        elif mean_depth < 0.5:
            mean_score = (mean_depth / 0.5) * 0.4
        elif mean_depth < 1.0:
            mean_score = 0.4 + ((mean_depth - 0.5) / 0.5) * 0.3
        else:
            mean_score = min(0.7 + ((mean_depth - 1.0) / 1.0) * 0.3, 1.0)
        
        # Max depth carries 70% weight (worst case), mean carries 30%
        factor = max_score * 0.7 + mean_score * 0.3
        
        return min(factor, 1.0)
    
    def calculate_social_reports_factor(
        self,
        social_analysis: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """
        Calculate social media reports density factor.
        
        Returns both the factor score AND a boolean indicating whether
        social data was actually available. If no social posts were found
        or social media collection failed, has_data=False and the weight
        for this factor will be redistributed to other factors.
        
        Args:
            social_analysis: Social media analysis results
                Expected keys: report_density, urgency_score, relevant_posts
            
        Returns:
            Tuple of (risk_factor 0-1, has_data bool)
        """
        relevant_posts = social_analysis.get('relevant_posts', 0)
        
        # If no social data at all, signal that this factor is absent
        if relevant_posts == 0 and social_analysis.get('report_density', 0.0) == 0.0:
            return 0.0, False
        
        report_density = social_analysis.get('report_density', 0.0)
        urgency_score = social_analysis.get('urgency_score', 0.0)
        
        # Combine density and urgency
        factor = (report_density * 0.6 + urgency_score * 0.4)
        
        return min(factor, 1.0), True
    
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
    
    # -----------------------------------------------------------------
    # RISK FACTOR AGGREGATION
    # -----------------------------------------------------------------
    
    def calculate_risk_factors(
        self,
        zone: SentinelZone,
        normalized_weather: Optional[Dict[str, float]],
        social_analysis: Dict[str, Any],
        historical_risk: float,
        satellite_data: Optional[Dict[str, Any]] = None,
        depth_data: Optional[Dict[str, Any]] = None
    ) -> FloodRiskFactors:
        """
        Calculate all 8 risk factors and return a FloodRiskFactors object
        with dynamic weighting metadata.
        
        Args:
            zone: Sentinel zone
            normalized_weather: Normalized weather metrics
            social_analysis: Social media analysis (can be empty dict)
            historical_risk: Historical risk score (0-1)
            satellite_data: Dict with satellite_risk, satellite_flood_pct,
                           satellite_flood_area_km2 keys. None if unavailable.
            depth_data: Dict from DepthPredictor.analyze() with statistics
                       sub-dict. None if unavailable.
            
        Returns:
            Complete risk factors with metadata flags
        """
        # Handle missing weather data
        if not normalized_weather:
            normalized_weather = {
                'rainfall_intensity': 0.0,
                'accumulated_rainfall': 0.0,
                'weather_severity': 0.0
            }
        
        # --- Calculate each factor ---
        
        # Weather factors
        rainfall = self.calculate_rainfall_intensity_factor(normalized_weather)
        accumulated = self.calculate_accumulated_rainfall_factor(normalized_weather)
        weather_sev = self.calculate_weather_severity_factor(normalized_weather)
        
        # Satellite factors
        has_satellite = (
            satellite_data is not None
            and satellite_data.get('satellite_flood_pct') is not None
        )
        satellite_confirmed = False
        
        if has_satellite:
            sat_flood_pct = (satellite_data or {}).get('satellite_flood_pct', 0.0)
            sat_risk = (satellite_data or {}).get('satellite_risk', 'MINIMAL')
            satellite_factor = self.calculate_satellite_flood_factor(
                sat_flood_pct, sat_risk
            )
            
            # Check if satellite confirms active flooding
            satellite_confirmed = (
                sat_flood_pct
                >= self.SATELLITE_FLOOD_THRESHOLDS['confirmed_flooding']
            )
            
            logger.info(
                f"  Satellite factor: {satellite_factor:.3f} "
                f"(flood_pct={sat_flood_pct:.1f}%, risk={sat_risk}, "
                f"confirmed={'YES' if satellite_confirmed else 'NO'})"
            )
        else:
            satellite_factor = 0.0
            logger.info(
                "  Satellite data: not available — using weather-only mode"
            )
        
        # Depth factor
        depth_factor = self.calculate_flood_depth_factor(depth_data)
        if depth_data:
            stats = depth_data.get('statistics', {})
            logger.info(
                f"  Depth factor: {depth_factor:.3f} "
                f"(max={stats.get('max_depth_m', 0):.2f}m, "
                f"mean={stats.get('mean_depth_m', 0):.2f}m)"
            )
        
        # Social media factor (optional)
        social_factor, has_social = self.calculate_social_reports_factor(
            social_analysis
        )
        if has_social:
            logger.info(
                f"  Social factor: {social_factor:.3f} (data available)"
            )
        else:
            logger.debug(
                "  Social media data: not available — weight redistributed"
            )
        
        # Zone factors
        drainage = self.calculate_drainage_factor(zone)
        elevation = self.calculate_elevation_factor(zone)
        
        return FloodRiskFactors(
            # Weather
            rainfall_intensity=rainfall,
            accumulated_rainfall=accumulated,
            weather_severity=weather_sev,
            # Satellite
            satellite_flood_detection=satellite_factor,
            flood_depth_estimate=depth_factor,
            # Zone
            drainage_factor=drainage,
            elevation_factor=elevation,
            # Optional
            social_reports_density=social_factor,
            historical_risk=historical_risk,
            # Metadata for dynamic weighting
            has_satellite_data=has_satellite,
            has_social_data=has_social,
            satellite_confirmed_flooding=satellite_confirmed,
        )
    
    # -----------------------------------------------------------------
    # CONFIDENCE CALCULATION
    # -----------------------------------------------------------------
    
    def calculate_confidence(
        self,
        has_weather: bool,
        weather_age_hours: float,
        has_satellite: bool,
        satellite_confidence: float,
        social_credibility: float,
        social_posts_count: int,
        spatial_coherence: float
    ) -> float:
        """
        Calculate prediction confidence.
        
        Satellite data significantly boosts confidence (ground truth).
        Social media is helpful but not required.
        
        Args:
            has_weather: Whether weather data is available
            weather_age_hours: Age of weather data in hours
            has_satellite: Whether satellite data is available
            satellite_confidence: Confidence from satellite service (0-1)
            social_credibility: Average credibility of social posts
            social_posts_count: Number of social posts analyzed
            spatial_coherence: Spatial analysis coherence score
            
        Returns:
            Confidence score (0-1)
        """
        # Data completeness — satellite adds a big boost
        completeness = 0.2  # Base
        if has_weather:
            completeness += 0.3
        if has_satellite:
            completeness += 0.4  # Satellite = strongest data
        if social_posts_count > 0:
            completeness += 0.1
        completeness = min(completeness, 1.0)
        
        # Data freshness (weather)
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
        
        # Satellite confidence
        sat_conf = satellite_confidence if has_satellite else 0.3
        
        # Social credibility (graceful when absent)
        social_conf = social_credibility if social_posts_count > 0 else 0.5
        
        # Historical accuracy (placeholder — would track actual accuracy)
        historical_accuracy = 0.75
        
        # Weighted combination
        confidence = (
            completeness * self.confidence_factors['data_completeness']
            + freshness * self.confidence_factors['data_freshness']
            + sat_conf * self.confidence_factors['satellite_confidence']
            + social_conf * self.confidence_factors['social_credibility']
            + historical_accuracy * self.confidence_factors['historical_accuracy']
            + spatial_coherence * self.confidence_factors['spatial_coherence']
        )
        
        return min(confidence, 1.0)
    
    # -----------------------------------------------------------------
    # TIME TO IMPACT
    # -----------------------------------------------------------------
    
    def estimate_time_to_impact(
        self,
        severity: SeverityLevel,
        rainfall_intensity: float,
        satellite_confirmed: bool,
        flood_depth_factor: float,
        social_urgency: float = 0.0
    ) -> Optional[float]:
        """
        Estimate time until flood impact.
        
        If satellite confirms active flooding, impact is NOW (immediate).
        Otherwise uses weather intensity + optional social urgency.
        
        Args:
            severity: Predicted severity level
            rainfall_intensity: Normalized rainfall intensity
            satellite_confirmed: Whether satellite detected active flooding
            flood_depth_factor: Depth factor (0-1), higher = deeper
            social_urgency: Social media urgency score (optional, default 0)
            
        Returns:
            Hours until impact or None
        """
        base_time = self.impact_time_estimates.get(severity)
        
        if base_time is None:
            return None
        
        # SATELLITE OVERRIDE: If satellite sees flooding, it's happening NOW
        if satellite_confirmed:
            if flood_depth_factor > 0.7:
                return 0.0  # Already deep flooding
            else:
                return 0.25  # 15 minutes — imminent/active
        
        # Adjust based on rainfall intensity
        if rainfall_intensity > 0.8:
            base_time *= 0.5  # Halve time for extreme rainfall
        elif rainfall_intensity > 0.6:
            base_time *= 0.75
        
        # Social media can signal flooding already occurring (if available)
        if social_urgency > 0.7:
            base_time = min(base_time, 0.5)  # Already happening or imminent
        
        return max(base_time, 0.25)  # Minimum 15 minutes
    
    # -----------------------------------------------------------------
    # RECOMMENDED ACTIONS
    # -----------------------------------------------------------------
    
    def generate_recommended_actions(
        self,
        severity: SeverityLevel,
        time_to_impact: Optional[float],
        affected_area_km2: float,
        critical_infrastructure: List[str],
        satellite_confirmed: bool = False,
        flood_depth_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate recommended actions based on prediction.
        
        Satellite-confirmed flooding generates more urgent and specific
        actions than weather-only predictions.
        
        Args:
            severity: Severity level
            time_to_impact: Hours until impact
            affected_area_km2: Affected area
            critical_infrastructure: Infrastructure at risk
            satellite_confirmed: Whether satellite detected active flooding
            flood_depth_data: Depth statistics (if available)
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # --- Satellite-confirmed flooding actions (most urgent) ---
        if satellite_confirmed:
            actions.append(
                "SATELLITE CONFIRMED: Active flooding detected in this zone"
            )
            
            # Add depth-specific actions if depth data is available
            if flood_depth_data:
                stats = flood_depth_data.get('statistics', {})
                max_depth = stats.get('max_depth_m', 0)
                
                if max_depth >= 2.0:
                    actions.append(
                        f"CRITICAL DEPTH: Max {max_depth:.1f}m detected — "
                        "life-threatening. Deploy rescue boats immediately"
                    )
                elif max_depth >= 1.0:
                    actions.append(
                        f"DANGEROUS DEPTH: Max {max_depth:.1f}m detected — "
                        "evacuate ground-floor residents"
                    )
                elif max_depth >= 0.5:
                    actions.append(
                        f"MODERATE DEPTH: Max {max_depth:.1f}m detected — "
                        "restrict vehicle movement, assist elderly/disabled"
                    )
        
        # --- Severity-based actions ---
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
        
        # --- Time-sensitive actions ---
        if time_to_impact is not None and time_to_impact == 0.0:
            actions.insert(
                0,
                "FLOODING IS ACTIVE NOW — immediate response required"
            )
        elif time_to_impact is not None and time_to_impact < 2:
            actions.insert(
                0,
                f"URGENT: Impact expected in {time_to_impact:.1f} hours"
            )
        
        # --- Area-specific actions ---
        if affected_area_km2 > 10:
            actions.append(
                "Large area affected: coordinate multi-district response"
            )
        
        # --- Infrastructure-specific actions ---
        if "hospitals" in critical_infrastructure:
            actions.append(
                "Alert hospitals: prepare for patient influx/evacuation"
            )
        
        if "schools" in critical_infrastructure:
            actions.append("Consider school closures in affected areas")
        
        if "emergency_services" in critical_infrastructure:
            actions.append("Relocate emergency services if necessary")
        
        return actions
    
    # -----------------------------------------------------------------
    # MAIN PREDICTION METHOD
    # -----------------------------------------------------------------
    
    def predict_flood_risk(
        self,
        zone: SentinelZone,
        weather_data: Optional[WeatherData],
        normalized_weather: Optional[Dict[str, float]],
        social_analysis: Dict[str, Any],
        spatial_analysis: SpatialAnalysisResult,
        historical_risk: float,
        satellite_data: Optional[Dict[str, Any]] = None,
        depth_data: Optional[Dict[str, Any]] = None
    ) -> FloodPrediction:
        """
        Generate comprehensive flood risk prediction using 8-factor model.
        
        Args:
            zone: Sentinel zone
            weather_data: Raw weather data (can be None)
            normalized_weather: Normalized weather metrics (can be None)
            social_analysis: Social media analysis (can be empty {})
            spatial_analysis: Spatial analysis results
            historical_risk: Historical risk score
            satellite_data: Dict with satellite_risk, satellite_flood_pct,
                           satellite_flood_area_km2. None if unavailable.
            depth_data: Dict from DepthPredictor.analyze(). None if unavailable.
            
        Returns:
            Complete flood prediction
        """
        logger.info(f"Generating prediction for zone: {zone.name}")
        
        # --- Step 1: Calculate all risk factors ---
        risk_factors = self.calculate_risk_factors(
            zone=zone,
            normalized_weather=normalized_weather,
            social_analysis=social_analysis,
            historical_risk=historical_risk,
            satellite_data=satellite_data,
            depth_data=depth_data
        )
        
        # --- Step 2: Get overall risk score (dynamic weighting in model) ---
        risk_score = risk_factors.weighted_score
        
        # --- Step 3: Determine severity level ---
        severity = FloodPrediction._risk_to_severity(risk_score)
        
        # --- Step 4: Calculate confidence ---
        weather_age = 0.0
        if weather_data:
            weather_age = (
                datetime.now(timezone.utc) - weather_data.timestamp
            ).total_seconds() / 3600
        
        # Extract satellite confidence if available
        sat_confidence = 0.0
        if satellite_data:
            sat_confidence = satellite_data.get(
                'satellite_confidence', 0.5
            )
        
        confidence = self.calculate_confidence(
            has_weather=weather_data is not None,
            weather_age_hours=weather_age,
            has_satellite=risk_factors.has_satellite_data,
            satellite_confidence=sat_confidence,
            social_credibility=social_analysis.get(
                'average_credibility', 0.5
            ),
            social_posts_count=social_analysis.get('relevant_posts', 0),
            spatial_coherence=spatial_analysis.average_severity
        )
        
        # --- Step 5: Estimate time to impact ---
        time_to_impact = self.estimate_time_to_impact(
            severity=severity,
            rainfall_intensity=risk_factors.rainfall_intensity,
            satellite_confirmed=risk_factors.satellite_confirmed_flooding,
            flood_depth_factor=risk_factors.flood_depth_estimate,
            social_urgency=social_analysis.get('urgency_score', 0.0)
        )
        
        # --- Step 6: Calculate affected area ---
        # If satellite has flood_area_km2, use it (ground truth)
        # Otherwise fall back to spatial analysis estimate
        affected_area = spatial_analysis.affected_area_km2
        if (satellite_data
                and satellite_data.get('satellite_flood_area_km2', 0) > 0):
            affected_area = max(
                affected_area,
                satellite_data['satellite_flood_area_km2']
            )
        
        # --- Step 7: Generate recommended actions ---
        recommended_actions = self.generate_recommended_actions(
            severity=severity,
            time_to_impact=time_to_impact,
            affected_area_km2=affected_area,
            critical_infrastructure=(
                spatial_analysis.critical_infrastructure_at_risk
            ),
            satellite_confirmed=risk_factors.satellite_confirmed_flooding,
            flood_depth_data=depth_data
        )
        
        # --- Step 8: Determine alert type ---
        if risk_factors.satellite_confirmed_flooding:
            # Satellite-confirmed flooding = always FLOOD_RISK alert
            alert_level = AlertType.FLOOD_RISK
        elif severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            alert_level = AlertType.FLOOD_RISK
        elif severity == SeverityLevel.MODERATE:
            alert_level = AlertType.WEATHER_WARNING
        else:
            alert_level = AlertType.ALL_CLEAR
        
        # Handle time_to_impact for FloodPrediction model validation
        # (model requires gt=0 if set, so 0.0 becomes None)
        prediction_tti = (
            time_to_impact
            if time_to_impact is not None and time_to_impact > 0
            else None
        )
        
        prediction = FloodPrediction(
            zone=zone,
            timestamp=datetime.now(timezone.utc),
            risk_score=risk_score,
            severity_level=severity,
            confidence=confidence,
            risk_factors=risk_factors,
            time_to_impact_hours=prediction_tti,
            affected_area_km2=affected_area,
            estimated_affected_population=(
                spatial_analysis.affected_population_estimate
            ),
            recommended_actions=recommended_actions,
            alert_level=alert_level
        )
        
        # --- Logging ---
        data_sources = []
        if weather_data:
            data_sources.append("weather")
        if risk_factors.has_satellite_data:
            data_sources.append("satellite")
            if risk_factors.satellite_confirmed_flooding:
                data_sources.append("SAR-CONFIRMED")
        if depth_data:
            data_sources.append("depth")
        if risk_factors.has_social_data:
            data_sources.append("social")
        
        logger.info(
            f"Prediction complete: {zone.name} — "
            f"Risk={risk_score:.2f}, Severity={severity.value}, "
            f"Confidence={confidence:.2f}, "
            f"Sources=[{', '.join(data_sources)}]"
        )

        # --- Weather summary log line ---
        if weather_data:
            rain_1h  = weather_data.precipitation.rain_1h  or 0.0
            rain_3h  = weather_data.precipitation.rain_3h  or 0.0
            rain_24h = weather_data.precipitation.rain_24h or 0.0
            logger.info(
                f"  Weather: {weather_data.condition.value} | "
                f"Temp={weather_data.metrics.temperature:.1f}C | "
                f"Humidity={weather_data.metrics.humidity:.0f}% | "
                f"Wind={weather_data.metrics.wind_speed:.1f}m/s | "
                f"Rain(1h/3h/24h)={rain_1h:.1f}/{rain_3h:.1f}/{rain_24h:.1f}mm | "
                f"Pressure={weather_data.metrics.pressure:.0f}hPa"
            )
        else:
            logger.info("  Weather: no data available")
        
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
                time_str = (
                    f"{prediction.time_to_impact_hours * 60:.0f} minutes"
                )
            else:
                time_str = f"{prediction.time_to_impact_hours:.1f}"
        elif prediction.risk_factors.satellite_confirmed_flooding:
            time_str = "NOW (satellite confirmed)"
        
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
        
        # Boost priority if satellite confirms flooding
        if prediction.risk_factors.satellite_confirmed_flooding:
            base_priority = max(base_priority, 4)
        
        # Boost priority if impact is imminent
        if (prediction.time_to_impact_hours
                and prediction.time_to_impact_hours < 2):
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
        
        logger.info(
            f"Generated {len(alerts)} alerts "
            f"from {len(predictions)} predictions"
        )
        
        return alerts


# =====================================================================
# PREDICTION ORCHESTRATOR
# =====================================================================

class PredictionOrchestrator:
    """
    Orchestrates the prediction process for all zones.
    Coordinates predictor and alert generator.
    
    Extracts satellite_data and depth_data from the processed_data dict
    (added by main.py's monitoring cycle) and passes them to the predictor.
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
        
        Extracts weather, satellite, depth, and social data from the
        processed_data dict and passes them to the 8-factor predictor.
        
        Args:
            processed_data: Processed zone data from monitoring pipeline.
                Expected keys:
                  - zone: SentinelZone
                  - weather: Optional[WeatherData]
                  - normalized_weather: Optional[Dict]
                  - social_analysis: Optional[Dict]
                  - satellite_risk: Optional[str]        (from main.py)
                  - satellite_flood_pct: Optional[float]  (from main.py)
                  - satellite_flood_area_km2: Optional[float] (from main.py)
                  - depth_analysis: Optional[Dict]        (from depth estimator)
            historical_risk: Historical risk score
            
        Returns:
            Tuple of (prediction, alert)
        """
        zone = processed_data['zone']
        weather = processed_data.get('weather')
        normalized_weather = processed_data.get('normalized_weather')
        social_analysis = processed_data.get('social_analysis', {})
        
        # --- Extract satellite data (if main.py attached it) ---
        satellite_data = None
        if processed_data.get('satellite_flood_pct') is not None:
            satellite_data = {
                'satellite_risk': processed_data.get(
                    'satellite_risk', 'MINIMAL'
                ),
                'satellite_flood_pct': processed_data.get(
                    'satellite_flood_pct', 0.0
                ),
                'satellite_flood_area_km2': processed_data.get(
                    'satellite_flood_area_km2', 0.0
                ),
                'satellite_confidence': processed_data.get(
                    'satellite_confidence', 0.5
                ),
            }
        
        # --- Extract depth data (if available) ---
        depth_data = processed_data.get('depth_analysis', None)
        
        # --- Create spatial analysis ---
        # Use satellite flood area if available, otherwise estimate
        if (satellite_data
                and satellite_data.get('satellite_flood_area_km2', 0) > 0):
            estimated_area = satellite_data['satellite_flood_area_km2']
        else:
            estimated_area = (
                social_analysis.get('report_density', 0)
                * zone.radius_km ** 2
            )
        
        spatial_analysis = SpatialAnalysisResult(
            zone=zone,
            timestamp=datetime.now(timezone.utc),
            affected_area_km2=estimated_area,
            nearby_reports_count=social_analysis.get('flood_reports', 0),
            average_severity=social_analysis.get('urgency_score', 0.0),
            risk_clusters=[],
            affected_population_estimate=None,
            critical_infrastructure_at_risk=[]
        )
        
        # --- Generate prediction using 8-factor model ---
        prediction = self.predictor.predict_flood_risk(
            zone=zone,
            weather_data=weather,
            normalized_weather=normalized_weather,
            social_analysis=social_analysis,
            spatial_analysis=spatial_analysis,
            historical_risk=historical_risk,
            satellite_data=satellite_data,
            depth_data=depth_data
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
            historical_risks: Dict mapping zone IDs to historical risk scores
            
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