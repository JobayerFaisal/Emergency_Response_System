# backend/src/agents/agent_1_environmental/main.py


"""
Environmental Intelligence Agent - Main Orchestrator
===================================================
Agent 1: Real-time environmental monitoring and flood prediction.
Coordinates data collection, processing, analysis, and prediction.

Author: Environmental Intelligence Team
Version: 1.0.0
"""


import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, cast
import signal
import sys
import json
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncpg
from redis import asyncio as aioredis
import uvicorn


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


# Import all components
from backend.src.agents.agent_1_environmental.models import (
    AgentOutput, SentinelZone, GeoPoint, SeverityLevel,
    HealthCheckResponse, MonitoringStatus, WeatherData
)
from backend.src.agents.agent_1_environmental.data_collectors import (
    WeatherAPICollector,
    SocialMediaCollector,
    DataCollectionOrchestrator,
    RiverLevelCollector
)
from backend.src.agents.agent_1_environmental.services.satellite_service import SatelliteDataCollector
from backend.src.agents.agent_1_environmental.data_processors import (
    LLMEnrichmentProcessor,
    WeatherDataNormalizer,
    SocialMediaAnalyzer,
    DataProcessingOrchestrator
)
from backend.src.agents.agent_1_environmental.spatial_analyzer import PostGISSpatialAnalyzer
from backend.src.agents.agent_1_environmental.predictor import (
    FloodRiskPredictor,
    AlertGenerator,
    PredictionOrchestrator
)

# Configure logging
# FIX 1: Force UTF-8 on the stream handler so emoji characters (✅ etc.)
# don't crash on Windows consoles that default to cp1252 encoding.
_utf8_stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_1_environmental.log', encoding='utf-8'),
        logging.StreamHandler(_utf8_stream),
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

class AgentConfig:
    """Agent configuration from environment variables"""
    
    def __init__(self):
        """Load configuration from environment"""
        import os
        from dotenv import load_dotenv

        # Load .env file
        load_dotenv()

        # API Keys
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        # FIX 2a: Twitter is optional — agent runs without it
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # Database
        # FIX 2b: Correct default to match real DATABASE_URL
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/disaster_response'
        )

        # FIX 2c: Redis default changed from redis://redis:6379 to localhost
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

        # Agent Settings
        self.agent_id = os.getenv('AGENT_ID', 'agent_1_environmental')
        self.monitoring_interval = int(os.getenv('MONITORING_INTERVAL', '300'))
        self.enable_adaptive_polling = os.getenv('ENABLE_ADAPTIVE_POLLING', 'true').lower() == 'true'

        # Validate only truly required config
        self._validate()

        # Tell type checkers these values are non-None after validation
        self.openweather_api_key = cast(str, self.openweather_api_key)
        self.openai_api_key = cast(str, self.openai_api_key)
    
    def _validate(self):
        """Validate required configuration — Twitter and Redis are optional"""
        required = {
            'OPENWEATHER_API_KEY': self.openweather_api_key,
            'OPENAI_API_KEY': self.openai_api_key,
        }

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        # FIX 2d: Warn about optional services instead of crashing
        if not self.twitter_bearer_token:
            logger.warning("TWITTER_BEARER_TOKEN not set — social media collection disabled")


# =====================================================================
# ENVIRONMENTAL INTELLIGENCE AGENT
# =====================================================================

class EnvironmentalIntelligenceAgent:
    """
    Main agent class coordinating all environmental monitoring operations.
    Implements adaptive polling, prediction, and alert generation.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with all components.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Component instances (initialized in startup)
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.weather_collector: Optional[WeatherAPICollector] = None
        self.social_collector: Optional[SocialMediaCollector] = None
        self.collection_orchestrator: Optional[DataCollectionOrchestrator] = None
        self.satellite_collector: Optional[SatelliteDataCollector] = None
        self.river_collector: Optional[RiverLevelCollector] = None
        self.llm_processor: Optional[LLMEnrichmentProcessor] = None
        self.weather_normalizer: Optional[WeatherDataNormalizer] = None
        self.social_analyzer: Optional[SocialMediaAnalyzer] = None
        self.processing_orchestrator: Optional[DataProcessingOrchestrator] = None
        self.spatial_analyzer: Optional[PostGISSpatialAnalyzer] = None
        self.flood_predictor: Optional[FloodRiskPredictor] = None
        self.alert_generator: Optional[AlertGenerator] = None
        self.prediction_orchestrator: Optional[PredictionOrchestrator] = None
        
        # Sentinel zones (loaded from database or config)
        self.sentinel_zones: List[SentinelZone] = []
        
        # Latest output
        self.latest_output: Optional[AgentOutput] = None
        self.last_update: Optional[datetime] = None
        
        logger.info(f"Agent {config.agent_id} initialized")
    
    async def startup(self):
        """Initialize all components and connections"""
        logger.info("Starting up Environmental Intelligence Agent...")
        
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=5,
                max_size=20
            )
            logger.info("Database connection pool created")
            
            # FIX 3: Redis is optional — if it's not running locally we skip caching
            # gracefully instead of crashing. All cache reads/writes already have
            # try/except, so None cache_client simply means no caching.
            try:
                self.redis_client = await aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Ping to verify the connection is actually alive.
                # FIX: Pyright redis stubs declare ping()->bool (sync stub bug).
                # We cast to suppress the false "bool is not awaitable" error.
                # At runtime aioredis.ping() is always a coroutine.
                import inspect as _inspect
                _ping = self.redis_client.ping()
                if _inspect.isawaitable(_ping):
                    await _ping
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}) — running without cache")
                self.redis_client = None
            
            # Initialize data collectors
            self.weather_collector = WeatherAPICollector(
                api_key=cast(str, self.config.openweather_api_key),
                cache_client=self.redis_client
            )
            
            # FIX 4: Only create social collector if a token is configured.
            # Without a valid bearer token Tweepy raises 401 on every call.
            if self.config.twitter_bearer_token:
                self.social_collector = SocialMediaCollector(
                    bearer_token=self.config.twitter_bearer_token,
                    cache_client=self.redis_client
                )
            else:
                logger.warning("Social media collector disabled — no Twitter token")
                self.social_collector = None
            
            # Initialize satellite collector (GEE + CNN flood detection)
            try:
                self.satellite_collector = SatelliteDataCollector(
                    cache_client=self.redis_client
                )
                logger.info("Satellite collector initialized")
            except Exception as e:
                logger.warning(f"Satellite collector unavailable: {e} — continuing without satellite data")
                self.satellite_collector = None
            
            # Initialize river level collector (GloFAS — no API key needed)
            self.river_collector = RiverLevelCollector(
                cache_client=self.redis_client
            )
            logger.info("River level collector initialized (GloFAS v4)")


            self.collection_orchestrator = DataCollectionOrchestrator(
                weather_collector=self.weather_collector,
                social_collector=self.social_collector,
                satellite_collector=self.satellite_collector,
                river_collector=self.river_collector
            )
            logger.info("Data collectors initialized")
            
            # Initialize data processors
            self.llm_processor = LLMEnrichmentProcessor(
                api_key=cast(str, self.config.openai_api_key)
            )
            
            self.weather_normalizer = WeatherDataNormalizer()
            self.social_analyzer = SocialMediaAnalyzer()
            
            self.processing_orchestrator = DataProcessingOrchestrator(
                llm_processor=self.llm_processor,
                weather_normalizer=self.weather_normalizer,
                social_analyzer=self.social_analyzer
            )
            logger.info("Data processors initialized")
            
            # Initialize spatial analyzer
            self.spatial_analyzer = PostGISSpatialAnalyzer(
                db_pool=cast(asyncpg.Pool, self.db_pool)
            )
            await self.spatial_analyzer.initialize_schema()
            logger.info("Spatial analyzer initialized")
            
            # Initialize predictors
            self.flood_predictor = FloodRiskPredictor()
            self.alert_generator = AlertGenerator()
            
            self.prediction_orchestrator = PredictionOrchestrator(
                predictor=self.flood_predictor,
                alert_generator=self.alert_generator
            )
            logger.info("Prediction system initialized")
            
            # Load sentinel zones
            await self.load_sentinel_zones()
            
            logger.info("✅ Agent startup complete")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Cleanup and close all connections"""
        logger.info("Shutting down Environmental Intelligence Agent...")
        
        self.running = False
        
        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database pool closed")
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
        
        logger.info("✅ Agent shutdown complete")
    
    async def load_sentinel_zones(self):
        """Load sentinel zones from database or create default zones"""
        # Try to load from database
        if self.db_pool is None:
            logger.error("Database pool is not initialized.")
            # Create default zones for Dhaka
            self.sentinel_zones = self._create_default_zones()
            # Store in database if possible
            if self.spatial_analyzer is not None:
                for zone in self.sentinel_zones:
                    await cast(PostGISSpatialAnalyzer, self.spatial_analyzer).store_sentinel_zone(zone)
            logger.info(f"Created {len(self.sentinel_zones)} default sentinel zones")
            return

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM sentinel_zones;")
            
            if rows:
                # Load existing zones
                self.sentinel_zones = [
                    SentinelZone(
                        id=row['id'],
                        name=row['name'],
                        center=GeoPoint(
                            latitude=await conn.fetchval(
                                "SELECT ST_Y($1::geometry);", row['center']
                            ),
                            longitude=await conn.fetchval(
                                "SELECT ST_X($1::geometry);", row['center']
                            )
                        ),
                        radius_km=row['radius_km'],
                        risk_level=SeverityLevel(row['risk_level']),
                        population_density=row['population_density'],
                        elevation=row['elevation'],
                        drainage_capacity=row['drainage_capacity'],
                        created_at=row['created_at'],
                        last_monitored=row['last_monitored']
                    )
                    for row in rows
                ]
                logger.info(f"Loaded {len(self.sentinel_zones)} sentinel zones from database")
            else:
                # Create default zones for Dhaka
                self.sentinel_zones = self._create_default_zones()
                
                # Store in database
                for zone in self.sentinel_zones:
                    await cast(PostGISSpatialAnalyzer, self.spatial_analyzer).store_sentinel_zone(zone)
                
                logger.info(f"Created {len(self.sentinel_zones)} default sentinel zones")
    
    def _create_default_zones(self) -> List[SentinelZone]:
        """Create default sentinel zones for Dhaka, Bangladesh"""
        return [
            # SentinelZone(
            #     name="Dhaka Central",
            #     center=GeoPoint(latitude=23.8103, longitude=90.4125),
            #     radius_km=5.0,
            #     risk_level=SeverityLevel.MODERATE,
            #     population_density=45000,
            #     elevation=6.0,
            #     drainage_capacity="poor"
            # ),
            # SentinelZone(
            #     name="Mirpur",
            #     center=GeoPoint(latitude=23.8223, longitude=90.3654),
            #     radius_km=4.0,
            #     risk_level=SeverityLevel.HIGH,
            #     population_density=52000,
            #     elevation=4.0,
            #     drainage_capacity="poor"
            # ),
            # SentinelZone(
            #     name="Gulshan",
            #     center=GeoPoint(latitude=23.7806, longitude=90.4175),
            #     radius_km=3.0,
            #     risk_level=SeverityLevel.LOW,
            #     population_density=35000,
            #     elevation=8.0,
            #     drainage_capacity="moderate"
            # ),
            # SentinelZone(
            #     name="Mohammadpur",
            #     center=GeoPoint(latitude=23.7697, longitude=90.3611),
            #     radius_km=4.0,
            #     risk_level=SeverityLevel.MODERATE,
            #     population_density=48000,
            #     elevation=5.0,
            #     drainage_capacity="poor"
            # ),
            # SentinelZone(
            #     name="Uttara",
            #     center=GeoPoint(latitude=23.8759, longitude=90.3795),
            #     radius_km=4.5,
            #     risk_level=SeverityLevel.MODERATE,
            #     population_density=42000,
            #     elevation=7.0,
            #     drainage_capacity="moderate"
            # )
            SentinelZone(
                name="Sunamganj Sadar",          # Haor region — floods every year
                center=GeoPoint(latitude=24.8660, longitude=91.3990),
                radius_km=6.0,
                risk_level=SeverityLevel.CRITICAL,
                population_density=28000,
                elevation=2.0,                   # Very low — 2m above sea level
                drainage_capacity="poor"
            ),
            SentinelZone(
                name="Sylhet City",              # 2022 worst flood — 84% submerged
                center=GeoPoint(latitude=24.8975, longitude=91.8720),
                radius_km=5.0,
                risk_level=SeverityLevel.HIGH,
                population_density=35000,
                elevation=3.5,
                drainage_capacity="poor"
            ),
            SentinelZone(
                name="Netrokona Sadar",          # Haor district — regular flooding
                center=GeoPoint(latitude=24.8703, longitude=90.7279),
                radius_km=5.0,
                risk_level=SeverityLevel.HIGH,
                population_density=22000,
                elevation=3.0,
                drainage_capacity="poor"
            ),
            SentinelZone(
                name="Sirajganj Sadar",          # Jamuna river corridor — annual floods
                center=GeoPoint(latitude=24.4490, longitude=89.7000),
                radius_km=5.0,
                risk_level=SeverityLevel.HIGH,
                population_density=31000,
                elevation=4.0,
                drainage_capacity="poor"
            ),
            SentinelZone(
                name="Jamalpur Sadar",           # Brahmaputra floodplain — severely flood-prone
                center=GeoPoint(latitude=24.9000, longitude=89.9333),
                radius_km=5.5,
                risk_level=SeverityLevel.HIGH,
                population_density=29000,
                elevation=3.5,
                drainage_capacity="poor"
            )



        ]
    
    async def run_monitoring_cycle(self) -> AgentOutput:
        """
        Execute one complete monitoring cycle.
        
        Returns:
            Agent output with predictions and alerts
        """
        cycle_start = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info("Starting monitoring cycle")
        
        try:
            # Step 1: Collect data from all sources
            logger.info("Step 1: Collecting data...")
            assert self.collection_orchestrator is not None, "DataCollectionOrchestrator not initialized"
            collected_data = await self.collection_orchestrator.collect_all_zones(
                self.sentinel_zones
            )
            
            # Step 2: Process and enrich data
            logger.info("Step 2: Processing and enriching data...")
            assert self.processing_orchestrator is not None, "DataProcessingOrchestrator not initialized"
            processed_data = await self.processing_orchestrator.process_all_zones(
                collected_data
            )
            
            # Step 3: Perform spatial analysis
            logger.info("Step 3: Performing spatial analysis...")
            assert self.spatial_analyzer is not None, "Spatial analyzer not initialized"
            spatial_results = {}
            satellite_summary = {}

            # Re-attach satellite AND river data from collected_data into processed_data.
            # DataProcessingOrchestrator only handles weather/social and passes the rest
            # of the dict through — but 'satellite' and 'river' keys come from the
            # collector output and must be explicitly re-attached here before prediction.
            collected_by_zone = {
                str(c['zone'].id): c.get('satellite')
                for c in collected_data
                if c.get('zone') is not None
            }
            # River data is stored under key 'river' in collector output (not 'river_data')
            river_by_zone = {
                str(c['zone'].id): c.get('river')
                for c in collected_data
                if c.get('zone') is not None
            }

            for data in processed_data:
                zone = data['zone']

                # Re-attach satellite data from collected_data if missing
                if 'satellite' not in data or data.get('satellite') is None:
                    data['satellite'] = collected_by_zone.get(str(zone.id))
                
                # Store weather and social data
                if data.get('weather'):
                    await cast(PostGISSpatialAnalyzer, self.spatial_analyzer).store_weather_data(
                        data['weather'],
                        str(zone.id)
                    )
                
                for post in data.get('enriched_posts', []):
                    await cast(PostGISSpatialAnalyzer, self.spatial_analyzer).store_social_post(
                        post,
                        str(zone.id)
                    )

                # Re-attach river data from collected_data.
                # The collector stores it under 'river', but the predictor expects
                # 'river_data'. processed_data never had this key — attach it now.
                if not data.get('river_data'):
                    data['river_data'] = river_by_zone.get(str(zone.id))
                
                # Carry satellite data forward for prediction
                sat = data.get('satellite')
                if sat and hasattr(sat, 'flood_detection') and sat.flood_detection:
                    fd = sat.flood_detection
                    satellite_summary[str(zone.id)] = {
                        'risk_level': fd.risk_level,
                        'flood_pct': fd.flood_percentage,
                        'flood_area_km2': fd.flood_area_km2,
                        'confidence': fd.confidence,
                        'status': fd.status,
                    }
                    data['satellite_risk'] = fd.risk_level
                    data['satellite_flood_pct'] = fd.flood_percentage
                    data['satellite_flood_area_km2'] = fd.flood_area_km2
                
                # Perform spatial analysis
                spatial_result = await cast(PostGISSpatialAnalyzer, self.spatial_analyzer).analyze_zone_spatial_patterns(
                    zone
                )
                spatial_results[str(zone.id)] = spatial_result
                
                # Add spatial result to processed data
                data['spatial_analysis'] = spatial_result
            
            if satellite_summary:
                logger.info(f"   Satellite data merged for {len(satellite_summary)} zones")
            
            # Step 4: Get historical risk scores
            logger.info("Step 4: Retrieving historical risk scores...")
            historical_risks = {}
            for zone in self.sentinel_zones:
                risk = await self.spatial_analyzer.get_historical_risk_score(zone)
                historical_risks[str(zone.id)] = risk
            
            # Step 5: Generate predictions and alerts
            logger.info("Step 5: Generating predictions and alerts...")
            assert self.prediction_orchestrator is not None, "PredictionOrchestrator not initialized"
            predictions, alerts = await self.prediction_orchestrator.predict_all_zones(
                processed_data,
                historical_risks
            )
            
            # Step 6: Store predictions in database
            logger.info("Step 6: Storing predictions...")
            for prediction in predictions:
                if self.db_pool is None:
                    logger.error("Database pool is not initialized.")
                    continue
                async with self.db_pool.acquire() as conn:
                    try:
                        await conn.execute("""
                            INSERT INTO flood_predictions (
                                id, zone_id, timestamp, risk_score, severity_level,
                                confidence, time_to_impact_hours, affected_area_km2,
                                risk_factors, recommended_actions
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (id) DO UPDATE SET
                                risk_score = EXCLUDED.risk_score,
                                severity_level = EXCLUDED.severity_level,
                                confidence = EXCLUDED.confidence,
                                risk_factors = EXCLUDED.risk_factors;
                        """,
                            prediction.id,
                            prediction.zone.id,
                            prediction.timestamp.replace(tzinfo=None),
                            prediction.risk_score,
                            prediction.severity_level.value,
                            prediction.confidence,
                            prediction.time_to_impact_hours,
                            prediction.affected_area_km2,
                            json.dumps(prediction.risk_factors.model_dump()) if hasattr(prediction.risk_factors, 'model_dump') else json.dumps(prediction.risk_factors),
                            json.dumps(prediction.recommended_actions)
                        )
                        logger.info(f"Stored prediction for zone {prediction.zone.name}: risk={prediction.risk_score:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to store prediction for zone {prediction.zone.name}: {e}")
                        continue
            
            # Update zone monitoring timestamps
            for zone in self.sentinel_zones:
                zone.last_monitored = datetime.now(timezone.utc)
                await self.spatial_analyzer.store_sentinel_zone(zone)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            
            # Determine next update interval (adaptive polling)
            next_update = self._calculate_next_update_interval(predictions)

            # Create agent output
            river_zones_ok = sum(
                1 for d in processed_data
                if d.get('river_data') is not None
            )            
            # Create agent output
            output = AgentOutput(
                agent_id=self.config.agent_id,
                timestamp=datetime.now(timezone.utc),
                predictions=predictions,
                alerts=alerts,
                monitored_zones=self.sentinel_zones,
                data_sources_status={
                    'weather_api': 'operational',
                    'social_media': 'operational',
                    'spatial_db': 'operational',
                    'satellite_gee': 'operational' if satellite_summary else 'unavailable',
                    'river_glofas':  f'operational ({river_zones_ok}/{len(processed_data)} zones)' if river_zones_ok else 'unavailable',
                },
                processing_time_seconds=processing_time,
                next_update_in_seconds=next_update
            )
            
            # Store as latest output
            self.latest_output = output
            self.last_update = datetime.now(timezone.utc)

            # Step 7: Publish high-risk predictions to Redis → Agent 2
            if self.redis_client:
                for prediction in predictions:
                    if prediction.risk_score >= 0.5:
                        try:
                            msg = {
                                "message_id": str(__import__('uuid').uuid4()),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "sender_agent": "agent_1_environmental",
                                "receiver_agent": "agent_2_distress",
                                "message_type": "flood_alert",
                                "zone_id": str(prediction.zone.id),
                                "priority": 5 if prediction.risk_score >= 0.8 else 3,
                                "payload": {
                                    "zone_id": str(prediction.zone.id),
                                    "zone_name": prediction.zone.name,
                                    "risk_score": prediction.risk_score,
                                    "severity_level": prediction.severity_level.value,
                                    "confidence": prediction.confidence,
                                    "risk_factors": prediction.risk_factors.model_dump() if hasattr(prediction.risk_factors, "model_dump") else prediction.risk_factors,
                                    "timestamp": prediction.timestamp.isoformat(),
                                }
                            }
                            await self.redis_client.publish("flood_alert", json.dumps(msg))
                            logger.info(
                                f"📡 Published flood_alert for {prediction.zone.name} "
                                f"(risk={prediction.risk_score:.2f}, severity={prediction.severity_level.value})"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to publish flood_alert for {prediction.zone.name}: {e}")
            else:
                logger.debug("Redis not available — skipping flood_alert publish")

            # Log summary
            logger.info(f"✅ Monitoring cycle complete in {processing_time:.2f}s")
            logger.info(f"   Predictions: {len(predictions)}")
            logger.info(f"   Alerts: {len(alerts)} ({len(output.critical_alerts)} critical)")
            logger.info(f"   Next update in: {next_update}s")
            logger.info("=" * 60)
            
            return output
        
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
            raise
    
    def _calculate_next_update_interval(
        self,
        predictions: List
    ) -> float:
        """Calculate adaptive polling interval based on predictions"""
        if not self.config.enable_adaptive_polling:
            return self.config.monitoring_interval
        
        # Find highest risk level
        max_severity = SeverityLevel.MINIMAL
        for pred in predictions:
            if pred.severity_level.value > max_severity.value:
                max_severity = pred.severity_level
        
        # Map severity to interval
        intervals = {
            SeverityLevel.CRITICAL: 60,      # 1 minute
            SeverityLevel.HIGH: 300,         # 5 minutes
            SeverityLevel.MODERATE: 300,     # 5 minutes
            SeverityLevel.LOW: 900,          # 15 minutes
            SeverityLevel.MINIMAL: 180      # 3 minutes
        }
        
        return intervals.get(max_severity, self.config.monitoring_interval)
    
    async def start_monitoring(self):
        """Start continuous monitoring loop"""
        self.running = True
        logger.info("Starting continuous monitoring...")
        
        while self.running:
            try:
                # Run monitoring cycle
                output = await self.run_monitoring_cycle()
                
                # Wait for next update
                await asyncio.sleep(output.next_update_in_seconds)
                
            except asyncio.CancelledError:
                logger.info("Monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(60)
    
    def get_status(self) -> MonitoringStatus:
        """Get current monitoring status"""
        if not self.latest_output:
            return MonitoringStatus(
                active_zones=len(self.sentinel_zones),
                total_predictions=0,
                critical_alerts=0,
                last_update=datetime.now(timezone.utc),
                next_update=datetime.now(timezone.utc),
                data_freshness_seconds=float('inf')
            )
        
        freshness = (
            datetime.now(timezone.utc) - self.last_update
        ).total_seconds() if self.last_update else 0
        
        return MonitoringStatus(
            active_zones=len(self.sentinel_zones),
            total_predictions=len(self.latest_output.predictions),
            critical_alerts=len(self.latest_output.critical_alerts),
            last_update=self.last_update or datetime.now(timezone.utc),
            next_update=(self.last_update or datetime.now(timezone.utc)) + 
                       timedelta(seconds=self.latest_output.next_update_in_seconds),
            data_freshness_seconds=freshness
        )


# =====================================================================
# FASTAPI APPLICATION
# =====================================================================

# Global agent instance
agent: Optional[EnvironmentalIntelligenceAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global agent
    
    # Startup
    config = AgentConfig()
    agent = EnvironmentalIntelligenceAgent(config)
    await agent.startup()
    
    # Start monitoring in background
    agent.monitoring_task = asyncio.create_task(agent.start_monitoring())
    
    yield
    
    # Shutdown
    await agent.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Environmental Intelligence Agent",
    description="Agent 1: Real-time flood risk monitoring and prediction",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "agent": "Environmental Intelligence Agent",
        "status": "operational",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return HealthCheckResponse(
        status="healthy",
        agent_id=agent.config.agent_id,
        version="1.0.0",
        data_sources={
            'weather': agent.weather_collector is not None,
            'social_media': agent.social_collector is not None,
            'spatial_db': agent.spatial_analyzer is not None
        },
        database_connected=agent.db_pool is not None,
        cache_connected=agent.redis_client is not None
    )


@app.get("/status", response_model=MonitoringStatus)
async def get_status():
    """Get current monitoring status"""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.get_status()


@app.get("/output", response_model=AgentOutput)
async def get_latest_output():
    """Get latest agent output"""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent.latest_output:
        raise HTTPException(status_code=404, detail="No output available yet")
    
    return agent.latest_output


@app.post("/trigger")
async def trigger_monitoring_cycle(background_tasks: BackgroundTasks):
    """Manually trigger a monitoring cycle"""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    background_tasks.add_task(agent.run_monitoring_cycle)
    
    return {"message": "Monitoring cycle triggered"}


@app.get("/zones", response_model=List[SentinelZone])
async def get_zones():
    """Get all sentinel zones"""
    global agent
    
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.sentinel_zones


@app.get("/zones/{zone_id}/prediction")
async def get_zone_prediction(zone_id: str):
    """Get latest prediction for a specific zone"""
    global agent
    
    if not agent or not agent.latest_output:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    for pred in agent.latest_output.predictions:
        if str(pred.zone.id) == zone_id:
            return pred
    
    raise HTTPException(status_code=404, detail="Zone not found")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    """Main entry point for the agent"""
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Interrupt received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()