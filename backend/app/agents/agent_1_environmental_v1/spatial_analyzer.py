"""
Spatial Analyzer for Environmental Intelligence Agent
=====================================================
PostGIS-powered geospatial analysis for flood risk assessment.
Handles proximity queries, spatial clustering, and affected area calculations.

Author: Environmental Intelligence Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncpg
from asyncpg import Pool, Connection
import json

from models import (
    GeoPoint, BoundingBox, SentinelZone, EnrichedSocialPost,
    WeatherData, SpatialAnalysisResult, SeverityLevel
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================================================
# POSTGIS SPATIAL ANALYZER
# =====================================================================

class PostGISSpatialAnalyzer:
    """
    Performs geospatial analysis using PostGIS.
    Analyzes flood reports, weather patterns, and affected areas.
    """
    
    def __init__(self, db_pool: Pool):
        """
        Initialize spatial analyzer.
        
        Args:
            db_pool: AsyncPG connection pool with PostGIS-enabled database
        """
        self.db_pool = db_pool
        logger.info("PostGISSpatialAnalyzer initialized")
    
    async def initialize_schema(self) -> None:
        """Create necessary database tables and indexes"""
        async with self.db_pool.acquire() as conn:
            # Enable PostGIS extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            
            # Sentinel zones table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sentinel_zones (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    center GEOMETRY(Point, 4326) NOT NULL,
                    radius_km FLOAT NOT NULL,
                    risk_level VARCHAR(50) NOT NULL,
                    population_density INTEGER,
                    elevation FLOAT,
                    drainage_capacity VARCHAR(50),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_monitored TIMESTAMP
                );
            """)
            
            # Weather data table
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
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            
            # Social media posts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS social_media_posts (
                    id UUID PRIMARY KEY,
                    platform_id VARCHAR(255) UNIQUE NOT NULL,
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    content TEXT NOT NULL,
                    author VARCHAR(255),
                    location GEOMETRY(Point, 4326),
                    relevance_score FLOAT,
                    sentiment VARCHAR(50),
                    contains_flood_report BOOLEAN DEFAULT FALSE,
                    enriched BOOLEAN DEFAULT FALSE,
                    raw_data JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            
            # Flood predictions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS flood_predictions (
                    id UUID PRIMARY KEY,
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    risk_score FLOAT NOT NULL,
                    severity_level VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    time_to_impact_hours FLOAT,
                    affected_area_km2 FLOAT,
                    risk_factors JSONB,
                    recommended_actions JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)

            # Satellite flood maps table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS satellite_flood_maps (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    zone_id UUID REFERENCES sentinel_zones(id),
                    timestamp TIMESTAMP NOT NULL,
                    flood_probability FLOAT,
                    estimated_depth FLOAT,
                    raw_features JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)

            # Indexes for satellite table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_satellite_zone
                ON satellite_flood_maps(zone_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_satellite_timestamp
                ON satellite_flood_maps(timestamp DESC);
            """)


            # Create spatial indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_zones_center 
                ON sentinel_zones USING GIST(center);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_weather_location 
                ON weather_data USING GIST(location);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_social_location 
                ON social_media_posts USING GIST(location);
            """)
            
            # Create time-based indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_weather_timestamp 
                ON weather_data(timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_social_timestamp 
                ON social_media_posts(timestamp DESC);
            """)
            
            logger.info("Database schema initialized successfully")
    
    async def store_sentinel_zone(self, zone: SentinelZone) -> None:
        """
        Store or update a sentinel zone.
        
        Args:
            zone: Sentinel zone to store
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sentinel_zones (
                    id, name, center, radius_km, risk_level,
                    population_density, elevation, drainage_capacity,
                    created_at, last_monitored
                ) VALUES ($1, $2, ST_GeomFromText($3, 4326), $4, $5, $6, $7, $8, $9, $10)
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
    
    async def store_weather_data(
        self,
        weather: WeatherData,
        zone_id: Optional[str] = None
    ) -> None:
        """
        Store weather data.
        
        Args:
            weather: Weather data to store
            zone_id: Associated zone ID
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO weather_data (
                    id, zone_id, timestamp, location,
                    temperature, humidity, pressure, wind_speed,
                    precipitation_1h, precipitation_3h, precipitation_24h,
                    condition, raw_data
                ) VALUES ($1, $2, $3, ST_GeomFromText($4, 4326), $5, $6, $7, $8, $9, $10, $11, $12, $13)
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
                json.dumps(weather.raw_data) if weather.raw_data else None
            )
    
    async def store_social_post(
        self,
        post: EnrichedSocialPost,
        zone_id: Optional[str] = None
    ) -> None:
        """
        Store enriched social media post.
        
        Args:
            post: Enriched post to store
            zone_id: Associated zone ID
        """
        async with self.db_pool.acquire() as conn:
            location_wkt = post.location.to_wkt() if post.location else None
            
            await conn.execute("""
                INSERT INTO social_media_posts (
                    id, platform_id, zone_id, timestamp, content, author,
                    location, relevance_score, sentiment,
                    contains_flood_report, enriched, raw_data
                ) VALUES ($1, $2, $3, $4, $5, $6, 
                          CASE WHEN $7 IS NOT NULL THEN ST_GeomFromText($7, 4326) ELSE NULL END,
                          $8, $9, $10, $11, $12)
                ON CONFLICT (platform_id) DO UPDATE SET
                    relevance_score = EXCLUDED.relevance_score,
                    sentiment = EXCLUDED.sentiment,
                    contains_flood_report = EXCLUDED.contains_flood_report,
                    enriched = EXCLUDED.enriched;
            """,
                post.id,
                post.platform_id,
                zone_id,
                post.timestamp,
                post.content,
                post.author,
                location_wkt,
                post.relevance_score,
                post.sentiment,
                post.contains_flood_report,
                True,
                json.dumps(post.raw_data) if post.raw_data else None
            )
    
    # =====================================================================
    # SATELLITE DATA STORAGE
    # =====================================================================

    async def store_satellite_data(self, zone_id: str, sat: Dict[str, Any]) -> None:
        """
        Store satellite-based flood detection results.
        
        Args:
            zone_id: Related sentinel zone ID
            sat: Satellite result dictionary:
                {
                    "flood_probability": float,
                    "estimated_water_depth": float,
                    "raw_features": dict
                }
        """
        if not sat:
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO satellite_flood_maps (
                    zone_id, timestamp, flood_probability, estimated_depth, raw_features
                ) VALUES ($1, NOW(), $2, $3, $4)
            """,
            zone_id,
            sat.get("flood_probability"),
            sat.get("estimated_water_depth"),
            json.dumps(sat.get("raw_features"))
            )



    async def find_nearby_flood_reports(
        self,
        center: GeoPoint,
        radius_km: float,
        since_hours: int = 24,
        min_relevance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find flood reports near a location.
        
        Args:
            center: Center point
            radius_km: Search radius in kilometers
            since_hours: Look back this many hours
            min_relevance: Minimum relevance score
            
        Returns:
            List of flood reports with distance
        """
        async with self.db_pool.acquire() as conn:
            since_time = datetime.utcnow() - timedelta(hours=since_hours)
            
            rows = await conn.fetch("""
                SELECT 
                    id,
                    platform_id,
                    content,
                    author,
                    timestamp,
                    ST_X(location::geometry) as longitude,
                    ST_Y(location::geometry) as latitude,
                    ST_Distance(
                        location::geography,
                        ST_GeogFromText($1)
                    ) / 1000 as distance_km,
                    relevance_score,
                    sentiment,
                    contains_flood_report
                FROM social_media_posts
                WHERE 
                    location IS NOT NULL
                    AND timestamp >= $2
                    AND relevance_score >= $3
                    AND contains_flood_report = TRUE
                    AND ST_DWithin(
                        location::geography,
                        ST_GeogFromText($1),
                        $4
                    )
                ORDER BY distance_km ASC;
            """,
                f"POINT({center.longitude} {center.latitude})",
                since_time,
                min_relevance,
                radius_km * 1000  # Convert to meters
            )
            
            return [dict(row) for row in rows]
    
    # =====================================================================
    # SATELLITE DATA RETRIEVAL
    # =====================================================================

    async def get_latest_satellite_data(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest satellite flood prediction for a zone.
        
        Returns:
            Dictionary with flood_probability, estimated_depth, raw_features
        """
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
                "raw_features": row["raw_features"]
            }


    
    async def calculate_affected_area(
        self,
        zone: SentinelZone,
        since_hours: int = 24
    ) -> float:
        """
        Calculate affected area based on flood report clusters.
        
        Args:
            zone: Sentinel zone
            since_hours: Time window for reports
            
        Returns:
            Affected area in square kilometers
        """
        async with self.db_pool.acquire() as conn:
            since_time = datetime.utcnow() - timedelta(hours=since_hours)
            
            # Get convex hull of flood report locations
            result = await conn.fetchrow("""
                WITH flood_points AS (
                    SELECT location
                    FROM social_media_posts
                    WHERE 
                        zone_id = $1
                        AND timestamp >= $2
                        AND contains_flood_report = TRUE
                        AND location IS NOT NULL
                )
                SELECT 
                    COALESCE(
                        ST_Area(
                            ST_ConvexHull(
                                ST_Collect(location::geometry)
                            )::geography
                        ) / 1000000,  -- Convert to km²
                        0
                    ) as area_km2,
                    COUNT(*) as report_count
                FROM flood_points;
            """,
                zone.id,
                since_time
            )
            
            if result and result['report_count'] >= 3:
                # Need at least 3 points for a valid area
                return float(result['area_km2'])
            
            return 0.0
    
    async def find_risk_clusters(
        self,
        zone: SentinelZone,
        since_hours: int = 24,
        cluster_radius_m: float = 500
    ) -> List[GeoPoint]:
        """
        Find clusters of flood reports using DBSCAN-style clustering.
        
        Args:
            zone: Sentinel zone
            since_hours: Time window
            cluster_radius_m: Clustering radius in meters
            
        Returns:
            List of cluster center points
        """
        async with self.db_pool.acquire() as conn:
            since_time = datetime.utcnow() - timedelta(hours=since_hours)
            
            # Use ST_ClusterDBSCAN for spatial clustering
            rows = await conn.fetch("""
                WITH clustered_reports AS (
                    SELECT 
                        location,
                        ST_ClusterDBSCAN(location::geometry, eps := $3, minpoints := 3) 
                            OVER() as cluster_id
                    FROM social_media_posts
                    WHERE 
                        zone_id = $1
                        AND timestamp >= $2
                        AND contains_flood_report = TRUE
                        AND location IS NOT NULL
                )
                SELECT 
                    cluster_id,
                    ST_X(ST_Centroid(ST_Collect(location::geometry))) as longitude,
                    ST_Y(ST_Centroid(ST_Collect(location::geometry))) as latitude,
                    COUNT(*) as report_count
                FROM clustered_reports
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
                HAVING COUNT(*) >= 3
                ORDER BY report_count DESC;
            """,
                zone.id,
                since_time,
                cluster_radius_m
            )
            
            clusters = [
                GeoPoint(latitude=row['latitude'], longitude=row['longitude'])
                for row in rows
            ]
            
            logger.info(f"Found {len(clusters)} risk clusters in zone {zone.name}")
            return clusters
    
    async def estimate_affected_population(
        self,
        zone: SentinelZone,
        affected_area_km2: float
    ) -> Optional[int]:
        """
        Estimate affected population based on area and population density.
        
        Args:
            zone: Sentinel zone
            affected_area_km2: Affected area in square kilometers
            
        Returns:
            Estimated affected population
        """
        if not zone.population_density or affected_area_km2 <= 0:
            return None
        
        # Simple estimation: density * area
        estimated = int(zone.population_density * affected_area_km2)
        
        # Cap at zone's total estimated population
        zone_area = zone.radius_km ** 2 * 3.14159
        max_population = int(zone.population_density * zone_area)
        
        return min(estimated, max_population)
    
    async def analyze_zone_spatial_patterns(
        self,
        zone: SentinelZone,
        since_hours: int = 24
    ) -> SpatialAnalysisResult:
        """
        Comprehensive spatial analysis for a zone.
        Now includes satellite flood probability + estimated water depth.
        """
        logger.info(f"Analyzing spatial patterns for zone: {zone.name}")

        # ------------------------------------------------------------
        # 1️⃣ Retrieve social-media–based flood data
        # ------------------------------------------------------------
        nearby_reports = await self.find_nearby_flood_reports(
            zone.center,
            zone.radius_km,
            since_hours
        )

        affected_area = await self.calculate_affected_area(zone, since_hours)
        risk_clusters = await self.find_risk_clusters(zone, since_hours)

        # Social-based severity score
        avg_severity = 0.0
        if nearby_reports:
            avg_severity = sum(r['relevance_score'] for r in nearby_reports) / len(nearby_reports)

        # ------------------------------------------------------------
        # 2️⃣ Retrieve latest satellite flood detection results
        # ------------------------------------------------------------
        satellite = await self.get_latest_satellite_data(str(zone.id))

        sat_flood_prob = satellite["flood_probability"] if satellite else 0.0
        sat_depth = satellite["estimated_depth"] if satellite else 0.0

        # ------------------------------------------------------------
        # 3️⃣ Blend severity using satellite + social media
        # ------------------------------------------------------------
        # 60% social sentiment + relevance
        # 40% satellite ML flood probability
        avg_severity = (avg_severity * 0.6) + (sat_flood_prob * 0.4)

        # ------------------------------------------------------------
        # 4️⃣ Improve affected area estimate using satellite depth
        # ------------------------------------------------------------
        if sat_depth and sat_depth > 0.1:  # >10cm depth = meaningful surface water
            min_area = zone.radius_km ** 2 * 0.15  # at least 15% of monitored zone
            affected_area = max(affected_area, min_area)

        # ------------------------------------------------------------
        # 5️⃣ Estimate affected population
        # ------------------------------------------------------------
        affected_pop = await self.estimate_affected_population(zone, affected_area)

        # ------------------------------------------------------------
        # 6️⃣ Find infrastructure at risk
        # ------------------------------------------------------------
        critical_infrastructure = await self._identify_critical_infrastructure(
            zone,
            affected_area
        )

        # ------------------------------------------------------------
        # 7️⃣ Return spatial analysis object WITH satellite results
        # ------------------------------------------------------------
        return SpatialAnalysisResult(
            zone=zone,
            timestamp=datetime.utcnow(),
            affected_area_km2=affected_area,
            nearby_reports_count=len(nearby_reports),
            average_severity=avg_severity,
            risk_clusters=risk_clusters,
            affected_population_estimate=affected_pop,
            critical_infrastructure_at_risk=critical_infrastructure,
            satellite_flood_probability=sat_flood_prob,    # NEW
            satellite_estimated_depth=sat_depth            # NEW
        )

    
    async def _identify_critical_infrastructure(
        self,
        zone: SentinelZone,
        affected_area_km2: float
    ) -> List[str]:
        """
        Identify critical infrastructure at risk (placeholder).
        
        In production, this would query a GIS database of infrastructure.
        """
        # Placeholder implementation
        infrastructure = []
        
        # Simple heuristic based on zone and area
        if affected_area_km2 > 1.0:
            infrastructure.append("roads_primary")
        
        if affected_area_km2 > 2.0:
            infrastructure.append("commercial_areas")
        
        if affected_area_km2 > 5.0:
            infrastructure.extend(["hospitals", "schools", "emergency_services"])
        
        # Check for specific high-risk zones
        if "hospital" in zone.name.lower():
            infrastructure.append("medical_facilities")
        
        return infrastructure
    
    async def get_historical_risk_score(
        self,
        zone: SentinelZone,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate historical flood risk based on past predictions.
        
        Args:
            zone: Sentinel zone
            lookback_days: Days to look back
            
        Returns:
            Historical risk score (0-1)
        """
        async with self.db_pool.acquire() as conn:
            since_time = datetime.utcnow() - timedelta(days=lookback_days)
            
            result = await conn.fetchrow("""
                SELECT 
                    AVG(risk_score) as avg_risk,
                    MAX(risk_score) as max_risk,
                    COUNT(*) as prediction_count
                FROM flood_predictions
                WHERE 
                    zone_id = $1
                    AND timestamp >= $2;
            """,
                zone.id,
                since_time
            )
            
            if result and result['prediction_count'] > 0:
                # Weight recent history: 70% average, 30% max
                historical_risk = (
                    float(result['avg_risk']) * 0.7 +
                    float(result['max_risk']) * 0.3
                )
                return min(historical_risk, 1.0)
            
            return 0.0
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """
        Clean up old data from database.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        async with self.db_pool.acquire() as conn:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old weather data
            weather_deleted = await conn.execute("""
                DELETE FROM weather_data WHERE timestamp < $1;
            """, cutoff_date)
            
            # Delete old social media posts
            social_deleted = await conn.execute("""
                DELETE FROM social_media_posts WHERE timestamp < $1;
            """, cutoff_date)
            
            # Keep predictions longer (90 days)
            prediction_cutoff = datetime.utcnow() - timedelta(days=90)
            predictions_deleted = await conn.execute("""
                DELETE FROM flood_predictions WHERE timestamp < $1;
            """, prediction_cutoff)
            
            logger.info(
                f"Cleaned up old data: {weather_deleted} weather records, "
                f"{social_deleted} social posts, {predictions_deleted} predictions"
            )


# =====================================================================
# SPATIAL QUERY HELPERS
# =====================================================================

class SpatialQueryHelper:
    """Helper functions for common spatial queries"""
    
    @staticmethod
    def haversine_distance(point1: GeoPoint, point2: GeoPoint) -> float:
        """
        Calculate haversine distance between two points.
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            Distance in kilometers
        """
        import math
        
        # Radius of Earth in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1 = math.radians(point1.latitude)
        lon1 = math.radians(point1.longitude)
        lat2 = math.radians(point2.latitude)
        lon2 = math.radians(point2.longitude)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    @staticmethod
    def is_point_in_zone(point: GeoPoint, zone: SentinelZone) -> bool:
        """
        Check if a point is within a zone's radius.
        
        Args:
            point: Point to check
            zone: Sentinel zone
            
        Returns:
            True if point is in zone
        """
        distance = SpatialQueryHelper.haversine_distance(point, zone.center)
        return distance <= zone.radius_km
    
    @staticmethod
    def calculate_bbox_area(bbox: BoundingBox) -> float:
        """
        Calculate approximate area of a bounding box.
        
        Args:
            bbox: Bounding box
            
        Returns:
            Area in square kilometers
        """
        # Approximate using haversine
        height = SpatialQueryHelper.haversine_distance(
            GeoPoint(latitude=bbox.south, longitude=bbox.west),
            GeoPoint(latitude=bbox.north, longitude=bbox.west)
        )
        
        width = SpatialQueryHelper.haversine_distance(
            GeoPoint(latitude=bbox.south, longitude=bbox.west),
            GeoPoint(latitude=bbox.south, longitude=bbox.east)
        )
        
        return height * width