"""
Agent 1: Environmental Intelligence - Main Application
Integrated monitoring of Weather, Social Media, and Satellite Imagery
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import sys

# Your existing monitors
# from weather_monitor import WeatherMonitor
# from social_media_monitor import SocialMediaMonitor

# New satellite monitor
from satellite_monitor import SatelliteImageryMonitor, SatelliteImageryData

# Database and messaging
# from database import DatabaseManager
# from redis_client import RedisClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Agent1Environmental:
    """
    Agent 1: Environmental Intelligence
    
    Monitors three data sources:
    1. Weather API - Real-time weather conditions
    2. Social Media - Flood reports from Twitter/Reddit
    3. Satellite Imagery - Sentinel-1 flood detection
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the Environmental Intelligence Agent"""
        
        self.config = config or self._default_config()
        
        # Initialize database and messaging (use your actual implementations)
        # self.db = DatabaseManager(config['database'])
        # self.redis = RedisClient(config['redis'])
        self.db = None  # Placeholder
        self.redis = None  # Placeholder
        
        # Initialize monitors
        logger.info("Initializing monitors...")
        
        # self.weather_monitor = WeatherMonitor(
        #     db_manager=self.db,
        #     redis_client=self.redis,
        #     api_key=config['openweather_api_key']
        # )
        
        # self.social_monitor = SocialMediaMonitor(
        #     db_manager=self.db,
        #     redis_client=self.redis,
        #     twitter_token=config['twitter_token']
        # )
        
        self.satellite_monitor = SatelliteImageryMonitor(
            db_manager=self.db,
            redis_client=self.redis,
            gee_credentials_path=config.get('gee_credentials_path')
        )
        
        self._monitoring_tasks = []
        self._is_running = False
        
        logger.info("Agent 1 initialized successfully")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'dhaka_location': (23.8103, 90.4125),
            'monitoring_radius_km': 50,
            'update_intervals': {
                'weather': 300,      # 5 minutes
                'social_media': 180,  # 3 minutes
                'satellite': 21600,   # 6 hours
            },
            'alert_thresholds': {
                'weather': {
                    'heavy_rain_mm': 50,
                    'wind_speed_kmh': 60
                },
                'satellite': {
                    'flood_area_km2': 10
                }
            }
        }
    
    async def start(self):
        """Start all monitoring services"""
        logger.info("="*60)
        logger.info("AGENT 1: ENVIRONMENTAL INTELLIGENCE - STARTING")
        logger.info("="*60)
        
        self._is_running = True
        
        # Start each monitor in its own task
        try:
            # Weather monitoring
            # self._monitoring_tasks.append(
            #     asyncio.create_task(self.weather_monitor.start_monitoring())
            # )
            
            # Social media monitoring
            # self._monitoring_tasks.append(
            #     asyncio.create_task(self.social_monitor.start_monitoring())
            # )
            
            # Satellite imagery monitoring
            self._monitoring_tasks.append(
                asyncio.create_task(self.satellite_monitor.start_monitoring())
            )
            
            # Alert processing task
            self._monitoring_tasks.append(
                asyncio.create_task(self._process_alerts())
            )
            
            # Status reporting task
            self._monitoring_tasks.append(
                asyncio.create_task(self._report_status())
            )
            
            logger.info("All monitoring services started")
            
            # Wait for all tasks
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            await self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            await self.stop()
    
    async def stop(self):
        """Stop all monitoring services gracefully"""
        logger.info("Stopping Agent 1...")
        
        self._is_running = False
        
        # Stop individual monitors
        # self.weather_monitor.stop_monitoring()
        # self.social_monitor.stop_monitoring()
        self.satellite_monitor.stop_monitoring()
        
        # Cancel all tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        logger.info("Agent 1 stopped")
    
    async def _process_alerts(self):
        """
        Process and correlate alerts from different sources
        This is where the magic happens - combining multiple data sources
        """
        logger.info("Alert processing system started")
        
        while self._is_running:
            try:
                # Check for correlated threats
                threat_analysis = await self._analyze_threats()
                
                if threat_analysis['threat_detected']:
                    await self._handle_threat(threat_analysis)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _analyze_threats(self) -> Dict:
        """
        Analyze data from all sources to identify correlated threats
        
        Returns threat analysis with combined intelligence from:
        - Weather conditions
        - Social media reports
        - Satellite imagery
        """
        analysis = {
            'threat_detected': False,
            'threat_level': 'none',
            'threat_type': None,
            'confidence': 0.0,
            'sources': [],
            'details': {}
        }
        
        try:
            # Get latest data from each source
            # weather_data = await self._get_latest_weather()
            # social_data = await self._get_latest_social()
            satellite_data = await self._get_latest_satellite()
            
            # Example: Correlate satellite flood with heavy rain
            # if satellite_data and satellite_data.flood_detected:
            #     analysis['threat_detected'] = True
            #     analysis['threat_type'] = 'flood'
            #     analysis['sources'].append('satellite')
            #     
            #     if weather_data and weather_data.rainfall_mm > 50:
            #         analysis['sources'].append('weather')
            #         analysis['confidence'] += 0.3
            #     
            #     if social_data and social_data.flood_mentions > 10:
            #         analysis['sources'].append('social_media')
            #         analysis['confidence'] += 0.2
            
            # For now, use satellite data only
            if satellite_data and satellite_data.flood_detected:
                analysis['threat_detected'] = True
                analysis['threat_type'] = 'flood'
                analysis['threat_level'] = satellite_data.threat_level
                analysis['confidence'] = satellite_data.confidence_score
                analysis['sources'] = ['satellite']
                analysis['details'] = {
                    'flood_area_km2': satellite_data.flood_area_km2,
                    'affected_regions': satellite_data.affected_regions
                }
            
        except Exception as e:
            logger.error(f"Error analyzing threats: {e}", exc_info=True)
        
        return analysis
    
    async def _get_latest_satellite(self) -> SatelliteImageryData:
        """Get latest satellite imagery data"""
        # In production, get from Redis cache or database
        # For now, return the summary
        try:
            summary = self.satellite_monitor.get_threat_summary()
            
            if summary['current_threat_level'] != 'none':
                # Create a SatelliteImageryData object from summary
                from satellite_monitor import SatelliteImageryData
                return SatelliteImageryData(
                    timestamp=summary['last_update'],
                    location=self.config['dhaka_location'],
                    flood_detected=True,
                    flood_area_km2=summary['flood_area_km2'],
                    confidence_score=summary['confidence'],
                    threat_level=summary['current_threat_level'],
                    affected_regions=[],
                    geojson_url=None,
                    map_urls={},
                    raw_data={}
                )
        except Exception as e:
            logger.debug(f"No recent satellite data: {e}")
        
        return None
    
    async def _handle_threat(self, analysis: Dict):
        """
        Handle detected threats
        - Log to database
        - Send alerts to other agents
        - Update dashboard
        """
        logger.warning("="*60)
        logger.warning(f"⚠️  THREAT DETECTED: {analysis['threat_type'].upper()}")
        logger.warning(f"Level: {analysis['threat_level'].upper()}")
        logger.warning(f"Confidence: {analysis['confidence']:.2%}")
        logger.warning(f"Sources: {', '.join(analysis['sources'])}")
        logger.warning("="*60)
        
        # Store threat in database
        # await self._store_threat(analysis)
        
        # Publish alert to Redis for other agents
        # await self._publish_threat_alert(analysis)
        
        # For now, just log
        if 'flood_area_km2' in analysis['details']:
            logger.warning(f"Flood Area: {analysis['details']['flood_area_km2']:.2f} km²")
            logger.warning(f"Affected Regions: {len(analysis['details']['affected_regions'])}")
    
    async def _report_status(self):
        """Periodically report agent status"""
        logger.info("Status reporting system started")
        
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Report every hour
                
                logger.info("="*60)
                logger.info("AGENT 1 STATUS REPORT")
                logger.info("="*60)
                logger.info(f"Time: {datetime.now().isoformat()}")
                logger.info(f"Status: {'RUNNING' if self._is_running else 'STOPPED'}")
                logger.info(f"Active Monitors: {len(self._monitoring_tasks)}")
                
                # Get status from each monitor
                # weather_status = self.weather_monitor.get_status()
                # social_status = self.social_monitor.get_status()
                satellite_status = self.satellite_monitor.get_threat_summary()
                
                logger.info(f"Satellite Monitor: {satellite_status['current_threat_level']}")
                logger.info("="*60)
                
            except Exception as e:
                logger.error(f"Error in status reporting: {e}", exc_info=True)
    
    async def test_satellite_only(self):
        """
        Test satellite monitoring independently
        Useful for debugging and capstone demo
        """
        logger.info("="*60)
        logger.info("TESTING SATELLITE IMAGERY MODULE")
        logger.info("="*60)
        
        result = await self.satellite_monitor.check_for_floods()
        
        if result:
            logger.info(f"\n✅ Satellite Check Complete")
            logger.info(f"Flood Detected: {result.flood_detected}")
            logger.info(f"Flood Area: {result.flood_area_km2:.2f} km²")
            logger.info(f"Threat Level: {result.threat_level.upper()}")
            logger.info(f"Confidence: {result.confidence_score:.2%}")
            
            if result.flood_detected:
                logger.warning(f"\n⚠️  FLOOD ALERT")
                logger.warning(f"Immediate attention required!")
        else:
            logger.warning("❌ Satellite check failed")


async def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'dhaka_location': (23.8103, 90.4125),
        'monitoring_radius_km': 50,
        
        # API Keys (load from environment in production)
        # 'openweather_api_key': os.getenv('OPENWEATHER_API_KEY'),
        # 'twitter_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'gee_credentials_path': None,  # Use interactive auth
        
        # Database
        # 'database': {
        #     'host': 'localhost',
        #     'port': 5432,
        #     'database': 'disaster_response',
        #     'user': 'postgres',
        #     'password': 'password'
        # },
        
        # Redis
        # 'redis': {
        #     'host': 'localhost',
        #     'port': 6379,
        #     'db': 0
        # }
    }
    
    # Initialize agent
    agent = Agent1Environmental(config)
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--test-satellite':
        # Test satellite only
        await agent.test_satellite_only()
    else:
        # Start full monitoring
        await agent.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
