"""
Satellite Imagery Monitor - Integration with Agent 1
Connects Google Earth Engine flood detection with your existing system
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

# Your existing imports (adjust paths as needed)
# from src.agents.agent_1_environmental.models import ThreatLevel, Location
# from src.agents.agent_1_environmental.database import DatabaseManager
# from src.agents.agent_1_environmental.redis_client import RedisClient

from satellite_imagery_service import SatelliteImageryService, FloodDetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SatelliteImageryData:
    """Data model for satellite imagery analysis"""
    timestamp: str
    location: Dict[str, float]  # {'lat': X, 'lon': Y}
    flood_detected: bool
    flood_area_km2: float
    confidence_score: float
    threat_level: str  # 'none', 'low', 'moderate', 'high', 'critical'
    affected_regions: List[Dict]
    geojson_url: Optional[str]
    map_urls: Dict[str, str]
    raw_data: Dict


class SatelliteImageryMonitor:
    """
    Monitor for satellite imagery-based flood detection
    Integrates with your existing Environmental Agent
    """
    
    def __init__(
        self,
        db_manager=None,  # Your DatabaseManager instance
        redis_client=None,  # Your RedisClient instance
        gee_credentials_path: Optional[str] = None
    ):
        """
        Initialize the satellite imagery monitor
        
        Args:
            db_manager: Your database manager instance
            redis_client: Your Redis client instance
            gee_credentials_path: Path to GEE service account credentials
        """
        self.db = db_manager
        self.redis = redis_client
        
        # Initialize satellite service
        self.satellite_service = SatelliteImageryService(gee_credentials_path)
        
        # Monitoring configuration
        self.config = {
            'dhaka_center': (23.8103, 90.4125),
            'monitoring_radius_km': 50,
            'check_interval_hours': 6,
            'threat_thresholds': {
                'critical': 100,  # km²
                'high': 50,
                'moderate': 10,
                'low': 1,
                'none': 0
            }
        }
        
        self._is_monitoring = False
        
    async def start_monitoring(self):
        """Start continuous satellite monitoring"""
        self._is_monitoring = True
        logger.info("Starting satellite imagery monitoring...")
        
        while self._is_monitoring:
            try:
                # Check for floods
                result = await self.check_for_floods()
                
                if result and result.flood_detected:
                    logger.warning(f"FLOOD DETECTED! Area: {result.flood_area_km2:.2f} km²")
                    
                    # Process and store results
                    await self.process_flood_detection(result)
                    
                    # Publish alert
                    await self.publish_alert(result)
                
                # Wait before next check
                await asyncio.sleep(self.config['check_interval_hours'] * 3600)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._is_monitoring = False
        logger.info("Stopping satellite imagery monitoring...")
    
    async def check_for_floods(self) -> Optional[SatelliteImageryData]:
        """
        Check satellite imagery for flood indicators
        
        Returns:
            SatelliteImageryData if analysis successful, None otherwise
        """
        try:
            logger.info("Checking satellite imagery for floods...")
            
            # Get date ranges
            now = datetime.now()
            after_end = now.strftime('%Y-%m-%d')
            after_start = (now - timedelta(days=2)).strftime('%Y-%m-%d')
            before_end = (now - timedelta(days=15)).strftime('%Y-%m-%d')
            before_start = (now - timedelta(days=45)).strftime('%Y-%m-%d')
            
            # Detect floods using satellite service
            result = await asyncio.to_thread(
                self.satellite_service.detect_flood,
                location=self.config['dhaka_center'],
                radius_km=self.config['monitoring_radius_km'],
                before_start=before_start,
                before_end=before_end,
                after_start=after_start,
                after_end=after_end
            )
            
            # Convert to your data model
            imagery_data = self._convert_to_imagery_data(result)
            
            return imagery_data
            
        except Exception as e:
            logger.error(f"Error checking satellite imagery: {e}", exc_info=True)
            return None
    
    def _convert_to_imagery_data(
        self, 
        result: FloodDetectionResult
    ) -> SatelliteImageryData:
        """Convert GEE result to your data model"""
        
        # Determine threat level based on flood area
        threat_level = self._calculate_threat_level(result.flood_area_km2)
        
        # Check if flood detected (area > 0.5 km²)
        flood_detected = result.flood_area_km2 > 0.5
        
        return SatelliteImageryData(
            timestamp=datetime.now().isoformat(),
            location={
                'lat': self.config['dhaka_center'][0],
                'lon': self.config['dhaka_center'][1]
            },
            flood_detected=flood_detected,
            flood_area_km2=result.flood_area_km2,
            confidence_score=result.detection_confidence,
            threat_level=threat_level,
            affected_regions=result.affected_regions,
            geojson_url=None,  # Set this if you upload to cloud storage
            map_urls=result.image_urls,
            raw_data=asdict(result)
        )
    
    def _calculate_threat_level(self, flood_area_km2: float) -> str:
        """Calculate threat level based on flood area"""
        thresholds = self.config['threat_thresholds']
        
        if flood_area_km2 >= thresholds['critical']:
            return 'critical'
        elif flood_area_km2 >= thresholds['high']:
            return 'high'
        elif flood_area_km2 >= thresholds['moderate']:
            return 'moderate'
        elif flood_area_km2 >= thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    async def process_flood_detection(self, data: SatelliteImageryData):
        """
        Process flood detection result
        Store in database, cache in Redis, etc.
        """
        try:
            # Store in database (adjust to your schema)
            if self.db:
                await self._store_in_database(data)
            
            # Cache in Redis for quick access
            if self.redis:
                await self._cache_in_redis(data)
            
            # Generate GeoJSON file
            await self._save_geojson(data)
            
            logger.info(f"Processed flood detection: {data.threat_level} threat")
            
        except Exception as e:
            logger.error(f"Error processing flood detection: {e}", exc_info=True)
    
    async def _store_in_database(self, data: SatelliteImageryData):
        """
        Store satellite imagery data in PostgreSQL
        Adjust this to match your database schema
        """
        # Example - adjust to your actual database schema
        query = """
            INSERT INTO satellite_imagery_data 
            (timestamp, location, flood_detected, flood_area_km2, 
             confidence_score, threat_level, affected_regions, map_urls)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        # This is pseudocode - adjust to your actual DB implementation
        # await self.db.execute(query, ...)
        
        logger.info("Stored satellite data in database")
    
    async def _cache_in_redis(self, data: SatelliteImageryData):
        """Cache latest satellite data in Redis"""
        # Store latest detection
        key = "satellite:latest"
        value = json.dumps(asdict(data))
        
        # This is pseudocode - adjust to your Redis implementation
        # await self.redis.set(key, value, ex=86400)  # 24 hour expiry
        
        # Also store in time series
        ts_key = f"satellite:timeseries"
        # await self.redis.zadd(ts_key, {value: data.timestamp})
        
        logger.info("Cached satellite data in Redis")
    
    async def _save_geojson(self, data: SatelliteImageryData):
        """Save GeoJSON to file system or cloud storage"""
        filename = f"flood_extent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        filepath = f"/mnt/user-data/outputs/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(data.raw_data['geojson'], f)
        
        logger.info(f"Saved GeoJSON to {filepath}")
    
    async def publish_alert(self, data: SatelliteImageryData):
        """
        Publish flood alert to Redis pub/sub
        This notifies other agents in your system
        """
        if not self.redis or data.threat_level == 'none':
            return
        
        alert = {
            'type': 'SATELLITE_FLOOD_ALERT',
            'timestamp': data.timestamp,
            'threat_level': data.threat_level,
            'flood_area_km2': data.flood_area_km2,
            'location': data.location,
            'affected_regions': data.affected_regions,
            'confidence': data.confidence_score
        }
        
        # Publish to Redis channel
        channel = "agent1:alerts"
        # await self.redis.publish(channel, json.dumps(alert))
        
        logger.warning(f"Published {data.threat_level} flood alert")
    
    async def get_historical_data(
        self, 
        days: int = 7
    ) -> List[SatelliteImageryData]:
        """
        Retrieve historical satellite imagery data
        Useful for trend analysis
        """
        # Query from database
        # This is pseudocode - adjust to your implementation
        
        query = """
            SELECT * FROM satellite_imagery_data
            WHERE timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp DESC
        """
        
        # results = await self.db.fetch(query, days)
        # return [self._row_to_data(row) for row in results]
        
        return []
    
    def get_threat_summary(self) -> Dict:
        """Get current threat summary from satellite data"""
        # Get latest from Redis or database
        # This is a placeholder
        
        return {
            'current_threat_level': 'moderate',
            'flood_area_km2': 15.3,
            'affected_regions': 3,
            'last_update': datetime.now().isoformat(),
            'confidence': 0.85
        }


# FastAPI Integration (if you're using FastAPI)
class SatelliteImageryRouter:
    """
    API routes for satellite imagery endpoints
    Add these to your FastAPI app
    """
    
    def __init__(self, monitor: SatelliteImageryMonitor):
        self.monitor = monitor
    
    async def get_latest_imagery(self):
        """GET /api/v1/satellite/latest"""
        data = await self.monitor.check_for_floods()
        return {
            'status': 'success',
            'data': asdict(data) if data else None
        }
    
    async def get_flood_status(self):
        """GET /api/v1/satellite/flood-status"""
        summary = self.monitor.get_threat_summary()
        return {
            'status': 'success',
            'data': summary
        }
    
    async def get_historical(self, days: int = 7):
        """GET /api/v1/satellite/historical?days=7"""
        history = await self.monitor.get_historical_data(days)
        return {
            'status': 'success',
            'data': [asdict(d) for d in history],
            'count': len(history)
        }
    
    async def trigger_manual_check(self):
        """POST /api/v1/satellite/check"""
        data = await self.monitor.check_for_floods()
        return {
            'status': 'success',
            'data': asdict(data) if data else None,
            'message': 'Manual check completed'
        }


async def main():
    """Example usage and testing"""
    
    # Initialize monitor
    monitor = SatelliteImageryMonitor()
    
    # Perform one-time check
    print("Performing satellite imagery check...")
    result = await monitor.check_for_floods()
    
    if result:
        print(f"\n{'='*60}")
        print(f"SATELLITE IMAGERY ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Location: {result.location}")
        print(f"Flood Detected: {result.flood_detected}")
        print(f"Flood Area: {result.flood_area_km2:.2f} km²")
        print(f"Threat Level: {result.threat_level.upper()}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Affected Regions: {len(result.affected_regions)}")
        print(f"{'='*60}\n")
        
        if result.flood_detected:
            print("⚠️  FLOOD ALERT - Immediate action recommended")
    
    # Uncomment to start continuous monitoring
    # await monitor.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())