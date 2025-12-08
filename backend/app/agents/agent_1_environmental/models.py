"""
Data Models for Environmental Intelligence Agent
=================================================
Pydantic models for weather data, social media content, spatial analysis,
and flood predictions.

Author: Environmental Intelligence Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from uuid import UUID, uuid4


# =====================================================================
# ENUMS
# =====================================================================

class SeverityLevel(str, Enum):
    """Flood severity classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherCondition(str, Enum):
    """Weather condition types"""
    CLEAR = "clear"
    CLOUDS = "clouds"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    DRIZZLE = "drizzle"
    SNOW = "snow"
    MIST = "mist"
    FOG = "fog"


class DataSource(str, Enum):
    """Source of environmental data"""
    OPENWEATHERMAP = "openweathermap"
    TWITTER = "twitter"
    SENSOR = "sensor"
    MANUAL = "manual"
    PREDICTION = "prediction"


class AlertType(str, Enum):
    """Types of alerts"""
    WEATHER_WARNING = "weather_warning"
    FLOOD_RISK = "flood_risk"
    EVACUATION = "evacuation"
    ALL_CLEAR = "all_clear"


# =====================================================================
# GEOSPATIAL MODELS
# =====================================================================

class GeoPoint(BaseModel):
    """Geographic point with coordinates"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "latitude": 23.8103,
            "longitude": 90.4125
        }
    })

    def to_wkt(self) -> str:
        """Convert to Well-Known Text format for PostGIS"""
        return f"POINT({self.longitude} {self.latitude})"
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON format"""
        return {
            "type": "Point",
            "coordinates": [self.longitude, self.latitude]
        }


class BoundingBox(BaseModel):
    """Geographic bounding box"""
    north: float = Field(..., ge=-90, le=90)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    west: float = Field(..., ge=-180, le=180)
    
    @validator('south')
    def south_less_than_north(cls, v, values):
        if 'north' in values and v >= values['north']:
            raise ValueError('South must be less than north')
        return v
    
    @validator('west')
    def west_less_than_east(cls, v, values):
        if 'east' in values and v >= values['east']:
            raise ValueError('West must be less than east')
        return v

    def to_wkt(self) -> str:
        """Convert to Well-Known Text polygon format"""
        return (f"POLYGON(("
                f"{self.west} {self.south}, "
                f"{self.east} {self.south}, "
                f"{self.east} {self.north}, "
                f"{self.west} {self.north}, "
                f"{self.west} {self.south}))")


# =====================================================================
# WEATHER DATA MODELS
# =====================================================================

class WeatherMetrics(BaseModel):
    """Core weather measurements"""
    temperature: float = Field(..., description="Temperature in Celsius")
    feels_like: float = Field(..., description="Perceived temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_direction: Optional[float] = Field(None, ge=0, le=360, description="Wind direction in degrees")
    visibility: Optional[float] = Field(None, ge=0, description="Visibility in meters")
    cloud_coverage: float = Field(..., ge=0, le=100, description="Cloud coverage percentage")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "temperature": 28.5,
            "feels_like": 32.0,
            "humidity": 85,
            "pressure": 1010,
            "wind_speed": 3.5,
            "wind_direction": 180,
            "visibility": 8000,
            "cloud_coverage": 75
        }
    })


class PrecipitationData(BaseModel):
    """Precipitation measurements"""
    rain_1h: Optional[float] = Field(None, ge=0, description="Rain volume for last hour in mm")
    rain_3h: Optional[float] = Field(None, ge=0, description="Rain volume for last 3 hours in mm")
    rain_24h: Optional[float] = Field(None, ge=0, description="Rain volume for last 24 hours in mm")
    snow_1h: Optional[float] = Field(None, ge=0, description="Snow volume for last hour in mm")
    snow_3h: Optional[float] = Field(None, ge=0, description="Snow volume for last 3 hours in mm")
    intensity: Optional[float] = Field(None, ge=0, description="Current precipitation intensity")
    
    @property
    def total_rain(self) -> float:
        """Calculate total rain in last 24 hours"""
        return self.rain_24h or self.rain_3h or self.rain_1h or 0.0
    
    @property
    def is_heavy(self) -> bool:
        """Check if precipitation is heavy (>7.5mm/hr)"""
        if self.rain_1h and self.rain_1h > 7.5:
            return True
        if self.rain_3h and (self.rain_3h / 3) > 7.5:
            return True
        return False


class WeatherData(BaseModel):
    """Complete weather observation data"""
    id: UUID = Field(default_factory=uuid4)
    location: GeoPoint
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    condition: WeatherCondition
    metrics: WeatherMetrics
    precipitation: PrecipitationData
    description: str = Field(..., description="Human-readable weather description")
    source: DataSource = DataSource.OPENWEATHERMAP
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Original API response")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "location": {"latitude": 23.8103, "longitude": 90.4125},
            "condition": "heavy_rain",
            "metrics": {
                "temperature": 28.5,
                "feels_like": 32.0,
                "humidity": 85,
                "pressure": 1010,
                "wind_speed": 3.5,
                "cloud_coverage": 75
            },
            "precipitation": {
                "rain_1h": 15.5,
                "rain_3h": 35.0
            },
            "description": "Heavy rain with thunderstorms"
        }
    })


# =====================================================================
# SOCIAL MEDIA MODELS
# =====================================================================

class SocialMediaPost(BaseModel):
    """Social media post with environmental information"""
    id: UUID = Field(default_factory=uuid4)
    platform_id: str = Field(..., description="Original post ID from platform")
    platform: str = Field(default="twitter", description="Social media platform")
    content: str = Field(..., description="Post content/text")
    author: str = Field(..., description="Author username")
    timestamp: datetime
    location: Optional[GeoPoint] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    media_urls: List[str] = Field(default_factory=list)
    engagement: Dict[str, int] = Field(default_factory=dict, description="Likes, retweets, etc.")
    source: DataSource = DataSource.TWITTER
    raw_data: Optional[Dict[str, Any]] = None


class EnrichedSocialPost(SocialMediaPost):
    """Social media post enriched with LLM analysis"""
    enriched_at: datetime = Field(default_factory=datetime.utcnow)
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance to flooding (0-1)")
    sentiment: str = Field(..., description="Sentiment: positive, negative, neutral, urgent")
    extracted_locations: List[str] = Field(default_factory=list)
    severity_indicators: List[str] = Field(default_factory=list)
    flood_keywords: List[str] = Field(default_factory=list)
    llm_summary: str = Field(..., description="LLM-generated summary")
    contains_flood_report: bool = Field(default=False)
    credibility_score: float = Field(..., ge=0, le=1, description="Estimated credibility (0-1)")


# =====================================================================
# SPATIAL ANALYSIS MODELS
# =====================================================================

class SentinelZone(BaseModel):
    """High-risk monitoring zone"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    center: GeoPoint
    radius_km: float = Field(..., gt=0, description="Monitoring radius in kilometers")
    risk_level: SeverityLevel
    population_density: Optional[int] = Field(None, description="People per sq km")
    elevation: Optional[float] = Field(None, description="Average elevation in meters")
    drainage_capacity: Optional[str] = Field(None, description="Poor, moderate, good, excellent")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_monitored: Optional[datetime] = None
    
    def get_bounding_box(self) -> BoundingBox:
        """Calculate bounding box for the zone"""
        # Approximate: 1 degree ≈ 111 km
        lat_offset = self.radius_km / 111.0
        lon_offset = self.radius_km / (111.0 * abs(self.center.latitude))
        
        return BoundingBox(
            north=self.center.latitude + lat_offset,
            south=self.center.latitude - lat_offset,
            east=self.center.longitude + lon_offset,
            west=self.center.longitude - lon_offset
        )


class SpatialAnalysisResult(BaseModel):
    """Result of geospatial analysis"""
    id: UUID = Field(default_factory=uuid4)
    zone: SentinelZone
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    affected_area_km2: float = Field(..., ge=0)
    nearby_reports_count: int = Field(..., ge=0)
    average_severity: float = Field(..., ge=0, le=1)
    risk_clusters: List[GeoPoint] = Field(default_factory=list)
    affected_population_estimate: Optional[int] = None
    critical_infrastructure_at_risk: List[str] = Field(default_factory=list)
    satellite_flood_probability: Optional[float] = 0.0
    satellite_estimated_depth: Optional[float] = 0.0


# =====================================================================
# PREDICTION MODELS
# =====================================================================

class FloodRiskFactors(BaseModel):
    """Factors contributing to flood risk"""
    rainfall_intensity: float = Field(..., ge=0, le=1, description="Normalized 0-1")
    accumulated_rainfall: float = Field(..., ge=0, le=1, description="Normalized 0-1")
    weather_severity: float = Field(..., ge=0, le=1, description="Normalized 0-1")
    social_reports_density: float = Field(..., ge=0, le=1, description="Normalized 0-1")
    historical_risk: float = Field(..., ge=0, le=1, description="Based on zone history")
    drainage_factor: float = Field(..., ge=0, le=1, description="1=poor, 0=excellent")
    elevation_factor: float = Field(..., ge=0, le=1, description="1=low, 0=high")
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted risk score"""
        weights = {
            'rainfall_intensity': 0.25,
            'accumulated_rainfall': 0.20,
            'weather_severity': 0.15,
            'social_reports_density': 0.15,
            'historical_risk': 0.10,
            'drainage_factor': 0.10,
            'elevation_factor': 0.05
        }
        
        return (
            self.rainfall_intensity * weights['rainfall_intensity'] +
            self.accumulated_rainfall * weights['accumulated_rainfall'] +
            self.weather_severity * weights['weather_severity'] +
            self.social_reports_density * weights['social_reports_density'] +
            self.historical_risk * weights['historical_risk'] +
            self.drainage_factor * weights['drainage_factor'] +
            self.elevation_factor * weights['elevation_factor']
        )


class FloodPrediction(BaseModel):
    """Flood risk prediction for a zone"""
    id: UUID = Field(default_factory=uuid4)
    zone: SentinelZone
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk score (0-1)")
    severity_level: SeverityLevel
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    risk_factors: FloodRiskFactors
    time_to_impact_hours: Optional[float] = Field(None, gt=0, description="Estimated hours until flooding")
    affected_area_km2: float = Field(..., ge=0)
    estimated_affected_population: Optional[int] = None
    recommended_actions: List[str] = Field(default_factory=list)
    alert_level: AlertType
    
    @validator('severity_level')
    def severity_matches_risk(cls, v, values):
        """Ensure severity level matches risk score"""
        if 'risk_score' in values:
            risk = values['risk_score']
            expected_severity = cls._risk_to_severity(risk)
            if v != expected_severity:
                # Auto-correct severity based on risk score
                return expected_severity
        return v
    
    @staticmethod
    def _risk_to_severity(risk_score: float) -> SeverityLevel:
        """Convert risk score to severity level"""
        if risk_score >= 0.8:
            return SeverityLevel.CRITICAL
        elif risk_score >= 0.6:
            return SeverityLevel.HIGH
        elif risk_score >= 0.4:
            return SeverityLevel.MODERATE
        elif risk_score >= 0.2:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.MINIMAL


# =====================================================================
# AGENT OUTPUT MODELS
# =====================================================================

class EnvironmentalAlert(BaseModel):
    """Alert message for other agents"""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    alert_type: AlertType
    severity: SeverityLevel
    zone: SentinelZone
    prediction: FloodPrediction
    message: str
    priority: int = Field(..., ge=1, le=5, description="1=lowest, 5=highest")
    
    @validator('priority')
    def priority_matches_severity(cls, v, values):
        """Auto-set priority based on severity"""
        if 'severity' in values:
            severity_map = {
                SeverityLevel.MINIMAL: 1,
                SeverityLevel.LOW: 2,
                SeverityLevel.MODERATE: 3,
                SeverityLevel.HIGH: 4,
                SeverityLevel.CRITICAL: 5
            }
            return severity_map.get(values['severity'], 3)
        return v


class AgentOutput(BaseModel):
    """Complete output from Environmental Intelligence Agent"""
    agent_id: str = "agent_1_environmental"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predictions: List[FloodPrediction]
    alerts: List[EnvironmentalAlert]
    monitored_zones: List[SentinelZone]
    data_sources_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of each data source"
    )
    processing_time_seconds: float
    next_update_in_seconds: float
    
    @property
    def critical_alerts(self) -> List[EnvironmentalAlert]:
        """Get only critical alerts"""
        return [a for a in self.alerts if a.severity == SeverityLevel.CRITICAL]
    
    @property
    def high_risk_zones(self) -> List[SentinelZone]:
        """Get zones with high or critical risk"""
        high_risk = set()
        for pred in self.predictions:
            if pred.severity_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                high_risk.add(pred.zone.id)
        return [z for z in self.monitored_zones if z.id in high_risk]


# =====================================================================
# DATABASE MODELS
# =====================================================================

class WeatherRecord(BaseModel):
    """Database record for weather data"""
    id: UUID = Field(default_factory=uuid4)
    zone_id: UUID
    timestamp: datetime
    location_wkt: str = Field(..., description="PostGIS POINT geometry")
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    precipitation_1h: Optional[float] = None
    precipitation_3h: Optional[float] = None
    condition: str
    raw_data_json: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SocialMediaRecord(BaseModel):
    """Database record for social media posts"""
    id: UUID = Field(default_factory=uuid4)
    platform_id: str
    zone_id: Optional[UUID] = None
    timestamp: datetime
    content: str
    author: str
    location_wkt: Optional[str] = None
    relevance_score: Optional[float] = None
    sentiment: Optional[str] = None
    contains_flood_report: bool = False
    enriched: bool = False
    raw_data_json: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionRecord(BaseModel):
    """Database record for predictions"""
    id: UUID = Field(default_factory=uuid4)
    zone_id: UUID
    timestamp: datetime
    risk_score: float
    severity_level: str
    confidence: float
    time_to_impact_hours: Optional[float] = None
    affected_area_km2: float
    risk_factors_json: Dict[str, Any]
    recommended_actions: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =====================================================================
# API RESPONSE MODELS
# =====================================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = "agent_1_environmental"
    version: str = "1.0.0"
    data_sources: Dict[str, bool]
    database_connected: bool
    cache_connected: bool


class MonitoringStatus(BaseModel):
    """Current monitoring status"""
    active_zones: int
    total_predictions: int
    critical_alerts: int
    last_update: datetime
    next_update: datetime
    data_freshness_seconds: float