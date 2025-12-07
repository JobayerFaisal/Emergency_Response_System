"""
Data Collectors for Environmental Intelligence Agent
=====================================================
Asynchronous data collection from weather APIs and social media platforms.
Implements adaptive polling, error handling, and rate limiting.

Author: Environmental Intelligence Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, cast
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
import tweepy
from redis import asyncio as aioredis

from models import (
    WeatherData, WeatherMetrics, PrecipitationData, WeatherCondition,
    SocialMediaPost, GeoPoint, SentinelZone, DataSource, BoundingBox
)


from satellite_imagery_service import SatelliteImageryService
from feature_extractor import FeatureExtractor
from flood_detector_ml import FloodMLDetector
from depth_estimator_ml import DepthEstimator



# Configure logging
logger = logging.getLogger(__name__)


# =====================================================================
# WEATHER DATA COLLECTOR
# =====================================================================

class WeatherAPICollector:
    """
    Collects weather data from OpenWeatherMap API.
    Implements caching, rate limiting, and error recovery.
    """
    
    def __init__(
        self,
        api_key: str,
        cache_client: Optional[aioredis.Redis] = None,
        cache_ttl: int = 600  # 10 minutes
    ):
        """
        Initialize weather API collector.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_client: Redis client for caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl
        self.timeout = ClientTimeout(total=30)
        
        # Rate limiting
        self.max_calls_per_minute = 60
        self.call_timestamps: List[datetime] = []
        
        logger.info("WeatherAPICollector initialized")
    
    async def _check_rate_limit(self) -> None:
        """Enforce rate limiting"""
        now = datetime.utcnow()
        # Remove timestamps older than 1 minute
        self.call_timestamps = [
            ts for ts in self.call_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        if len(self.call_timestamps) >= self.max_calls_per_minute:
            wait_time = 60 - (now - self.call_timestamps[0]).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self.call_timestamps.append(now)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache"""
        if not self.cache_client:
            return None
        
        try:
            cached_data = await self.cache_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {cache_key}")
                import json
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache"""
        if not self.cache_client:
            return
        
        try:
            import json
            await self.cache_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data, default=str)
            )
            logger.debug(f"Cached data for {cache_key}")
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _map_condition(self, weather_id: int, main: str) -> WeatherCondition:
        """Map OpenWeatherMap condition codes to our enum"""
        # Thunderstorm (200-299)
        if 200 <= weather_id < 300:
            return WeatherCondition.THUNDERSTORM
        # Drizzle (300-399)
        elif 300 <= weather_id < 400:
            return WeatherCondition.DRIZZLE
        # Rain (500-599)
        elif 500 <= weather_id < 600:
            if weather_id >= 502:  # Heavy rain
                return WeatherCondition.HEAVY_RAIN
            return WeatherCondition.RAIN
        # Snow (600-699)
        elif 600 <= weather_id < 700:
            return WeatherCondition.SNOW
        # Atmosphere (700-799)
        elif 700 <= weather_id < 800:
            if main.lower() in ['mist', 'fog']:
                return WeatherCondition.FOG if 'fog' in main.lower() else WeatherCondition.MIST
            return WeatherCondition.MIST
        # Clear (800)
        elif weather_id == 800:
            return WeatherCondition.CLEAR
        # Clouds (801-809)
        else:
            return WeatherCondition.CLOUDS
    
    async def fetch_current_weather(
        self,
        location: GeoPoint,
        zone_id: Optional[str] = None
    ) -> Optional[WeatherData]:
        """
        Fetch current weather for a location.
        
        Args:
            location: Geographic point to fetch weather for
            zone_id: Optional zone identifier for caching
            
        Returns:
            WeatherData object or None if fetch fails
        """
        cache_key = f"weather:{location.latitude}:{location.longitude}"
        
        # Check cache first
        cached = await self._get_from_cache(cache_key)
        if cached:
            try:
                return WeatherData(**cached)
            except Exception as e:
                logger.error(f"Error parsing cached weather data: {e}")
        
        # Enforce rate limit
        await self._check_rate_limit()
        
        # Fetch from API
        params = {
            'lat': location.latitude,
            'lon': location.longitude,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        url = f"{self.base_url}/weather"
        
        try:
            async with ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        weather_data = self._parse_weather_response(data, location)
                        
                        # Cache the result
                        await self._set_cache(
                            cache_key,
                            weather_data.model_dump()
                        )
                        
                        logger.info(
                            f"Fetched weather for ({location.latitude}, {location.longitude}): "
                            f"{weather_data.condition.value}"
                        )
                        return weather_data
                    
                    elif response.status == 429:
                        logger.error("API rate limit exceeded")
                        # Wait and retry once
                        await asyncio.sleep(5)
                        return await self.fetch_current_weather(location, zone_id)
                    
                    else:
                        logger.error(
                            f"Weather API error: {response.status} - {await response.text()}"
                        )
                        return None
        
        except ClientError as e:
            logger.error(f"Network error fetching weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}", exc_info=True)
            return None
    
    def _parse_weather_response(
        self,
        data: Dict[str, Any],
        location: GeoPoint
    ) -> WeatherData:
        """Parse OpenWeatherMap API response"""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        rain = data.get('rain', {})
        snow = data.get('snow', {})
        clouds = data.get('clouds', {})
        
        # Map weather condition
        weather_id = weather.get('id', 800)
        weather_main = weather.get('main', 'Clear')
        condition = self._map_condition(weather_id, weather_main)
        
        # Create metrics
        metrics = WeatherMetrics(
            temperature=main.get('temp', 0.0),
            feels_like=main.get('feels_like', 0.0),
            humidity=main.get('humidity', 0.0),
            pressure=main.get('pressure', 0.0),
            wind_speed=wind.get('speed', 0.0),
            wind_direction=wind.get('deg'),
            visibility=data.get('visibility'),
            cloud_coverage=clouds.get('all', 0.0)
        )
        
        # Create precipitation data
        # Provide rain_24h if available and derive an intensity value (mm/h) from 1h rain as a fallback
        precipitation = PrecipitationData(
            rain_1h=rain.get('1h'),
            rain_3h=rain.get('3h'),
            rain_24h=rain.get('24h'),
            snow_1h=snow.get('1h'),
            snow_3h=snow.get('3h'),
            intensity=(rain.get('1h') or 0.0)
        )
        
        return WeatherData(
            location=location,
            timestamp=datetime.utcfromtimestamp(data.get('dt', datetime.utcnow().timestamp())),
            condition=condition,
            metrics=metrics,
            precipitation=precipitation,
            description=weather.get('description', 'Unknown'),
            source=DataSource.OPENWEATHERMAP,
            raw_data=data
        )
    
    async def fetch_forecast(
        self,
        location: GeoPoint,
        hours: int = 48
    ) -> List[WeatherData]:
        """
        Fetch weather forecast for a location.
        
        Args:
            location: Geographic point
            hours: Number of hours to forecast (max 120)
            
        Returns:
            List of WeatherData objects
        """
        cache_key = f"forecast:{location.latitude}:{location.longitude}:{hours}"
        
        # Check cache
        cached = await self._get_from_cache(cache_key)
        if cached:
            try:
                import json
                # Ensure cached is a list of dicts with string keys
                if isinstance(cached, str):
                    cached = json.loads(cached)
                return [WeatherData(**dict(item)) for item in cached if isinstance(item, dict)]
            except Exception as e:
                logger.error(f"Error parsing cached forecast: {e}")
        
        await self._check_rate_limit()
        
        params = {
            'lat': location.latitude,
            'lon': location.longitude,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': min(hours // 3, 40)  # API returns 3-hour intervals
        }
        
        url = f"{self.base_url}/forecast"
        
        try:
            async with ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        forecasts = [
                            self._parse_weather_response(item, location)
                            for item in data.get('list', [])
                        ]
                        
                        # Cache with shorter TTL (30 minutes)
                        await self._set_cache(
                            cache_key,
                            [f.model_dump() for f in forecasts]
                        )
                        
                        logger.info(f"Fetched {len(forecasts)} forecast data points")
                        return forecasts
                    else:
                        logger.error(f"Forecast API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}", exc_info=True)
            return []
    
    async def fetch_multiple_zones(
        self,
        zones: List[SentinelZone]
    ) -> Dict[str, Optional[WeatherData]]:
        """
        Fetch weather for multiple zones concurrently.
        
        Args:
            zones: List of sentinel zones
            
        Returns:
            Dictionary mapping zone IDs to weather data
        """
        tasks = [
            self.fetch_current_weather(zone.center, str(zone.id))
            for zone in zones
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        zone_results: Dict[str, Optional[WeatherData]] = {}
        for i, zone in enumerate(zones):
            item = results[i]
            if isinstance(item, BaseException):
                zone_results[str(zone.id)] = None
            else:
                # Cast to WeatherData to satisfy static typing
                zone_results[str(zone.id)] = cast(WeatherData, item)
        return zone_results


# =====================================================================
# SOCIAL MEDIA COLLECTOR
# =====================================================================

class SocialMediaCollector:
    """
    Collects social media posts from Twitter/X API.
    Filters for flood-related content in specified geographic areas.
    """
    
    def __init__(
        self,
        bearer_token: str,
        cache_client: Optional[aioredis.Redis] = None
    ):
        """
        Initialize social media collector.
        
        Args:
            bearer_token: Twitter API v2 bearer token
            cache_client: Redis client for caching
        """
        self.bearer_token = bearer_token
        self.cache_client = cache_client
        
        # Initialize Twitter client
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True
        )
        
        # Flood-related keywords (Bengali and English)
        self.flood_keywords = [
            'flood', 'flooding', 'flooded',
            'waterlogging', 'waterlogged',
            'বন্যা', 'জলাবদ্ধতা', 'পানি',
            'heavy rain', 'rainfall',
            'inundation', 'submerged',
            'emergency', 'rescue',
            'evacuation', 'stranded'
        ]
        
        logger.info("SocialMediaCollector initialized")
    
    def _build_query(self, zone: SentinelZone, max_keywords: int = 5) -> str:
        """
        Build Twitter search query for a zone.
        
        Args:
            zone: Sentinel zone to search in
            max_keywords: Maximum number of keywords to include
            
        Returns:
            Twitter API query string
        """
        # Use top keywords
        keywords = ' OR '.join(self.flood_keywords[:max_keywords])
        
        # Add geocode filter
        bbox = zone.get_bounding_box()
        # Twitter uses center point and radius
        query = f"({keywords}) point_radius:[{zone.center.longitude} {zone.center.latitude} {zone.radius_km}km]"
        
        # Exclude retweets for quality
        query += " -is:retweet"
        
        # Only recent tweets (last 7 days for free tier)
        query += " -is:reply"
        
        return query
    
    async def fetch_recent_posts(
        self,
        zone: SentinelZone,
        max_results: int = 100,
        since_hours: int = 24
    ) -> List[SocialMediaPost]:
        """
        Fetch recent social media posts for a zone.
        
        Args:
            zone: Sentinel zone to search
            max_results: Maximum number of posts to return
            since_hours: Look back this many hours
            
        Returns:
            List of SocialMediaPost objects
        """
        cache_key = f"social:{zone.id}:{since_hours}"
        
        # Check cache (shorter TTL for social media - 5 minutes)
        if self.cache_client:
            try:
                cached = await self.cache_client.get(cache_key)
                if cached:
                    import json
                    cached_data = json.loads(cached)
                    logger.debug(f"Cache hit for social media: {zone.name}")
                    return [SocialMediaPost(**post) for post in cached_data]
            except Exception as e:
                logger.error(f"Cache retrieval error: {e}")
        
        query = self._build_query(zone)
        start_time = datetime.utcnow() - timedelta(hours=since_hours)
        
        try:
            # Use asyncio thread pool for synchronous tweepy call
            loop = asyncio.get_event_loop()
            tweets = await loop.run_in_executor(
                None,
                lambda: self.client.search_recent_tweets(
                    query=query,
                    max_results=min(max_results, 100),
                    start_time=start_time,
                    tweet_fields=['created_at', 'author_id', 'geo', 'entities', 'public_metrics'],
                    expansions=['author_id', 'geo.place_id'],
                    place_fields=['geo', 'country_code']
                )
            )
            
            tweet_data = getattr(tweets, "data", None)
            if not tweet_data:
                logger.info(f"No tweets found for zone {zone.name}")
                return []
            
            # Parse tweets
            posts = []
            includes = getattr(tweets, 'includes', None)
            for tweet in tweet_data:
                post = self._parse_tweet(tweet, includes)
                if post:
                    posts.append(post)
            
            # Cache results
            if self.cache_client and posts:
                try:
                    import json
                    await self.cache_client.setex(
                        cache_key,
                        300,  # 5 minutes TTL
                        json.dumps([p.model_dump() for p in posts], default=str)
                    )
                except Exception as e:
                    logger.error(f"Cache storage error: {e}")
            
            logger.info(f"Fetched {len(posts)} posts for zone {zone.name}")
            return posts
        
        except tweepy.TooManyRequests:
            logger.error("Twitter API rate limit exceeded")
            return []
        except tweepy.TwitterServerError as e:
            logger.error(f"Twitter server error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}", exc_info=True)
            return []
    
    def _parse_tweet(
        self,
        tweet: Any,
        includes: Optional[Dict] = None
    ) -> Optional[SocialMediaPost]:
        """Parse Twitter API v2 tweet object"""
        try:
            # Extract location if available
            location = None
            if hasattr(tweet, 'geo') and tweet.geo:
                # Try to get coordinates
                if 'coordinates' in tweet.geo:
                    coords = tweet.geo['coordinates']
                    location = GeoPoint(
                        latitude=coords['coordinates'][1],
                        longitude=coords['coordinates'][0]
                    )
            
            # Extract hashtags and mentions
            hashtags = []
            mentions = []
            if hasattr(tweet, 'entities') and tweet.entities:
                if 'hashtags' in tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                if 'mentions' in tweet.entities:
                    mentions = [m['username'] for m in tweet.entities['mentions']]
            
            # Get engagement metrics
            engagement = {}
            if hasattr(tweet, 'public_metrics'):
                engagement = {
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0),
                    'quotes': tweet.public_metrics.get('quote_count', 0)
                }
            
            # Get author username
            author = str(tweet.author_id)
            if includes and 'users' in includes:
                for user in includes['users']:
                    if user.id == tweet.author_id:
                        author = user.username
                        break
            
            return SocialMediaPost(
                platform_id=str(tweet.id),
                platform='twitter',
                content=tweet.text,
                author=author,
                timestamp=tweet.created_at,
                location=location,
                hashtags=hashtags,
                mentions=mentions,
                engagement=engagement,
                source=DataSource.TWITTER,
                raw_data=tweet.data if hasattr(tweet, 'data') else None
            )
        
        except Exception as e:
            logger.error(f"Error parsing tweet: {e}")
            return None
    
    async def fetch_multiple_zones(
        self,
        zones: List[SentinelZone],
        max_results_per_zone: int = 50
    ) -> Dict[str, List[SocialMediaPost]]:
        """
        Fetch posts for multiple zones concurrently.
        
        Args:
            zones: List of sentinel zones
            max_results_per_zone: Max posts per zone
            
        Returns:
            Dictionary mapping zone IDs to lists of posts
        """
        tasks = [
            self.fetch_recent_posts(zone, max_results_per_zone)
            for zone in zones
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        zone_posts: Dict[str, List[SocialMediaPost]] = {}
        for i, zone in enumerate(zones):
            res = results[i]
            if isinstance(res, list):
                zone_posts[str(zone.id)] = res
            else:
                zone_posts[str(zone.id)] = []
        
        return zone_posts







# =====================================================================
# SATELLITE IMAGERY COLLECTOR
# =====================================================================



class SatelliteImageryCollector:
    """
    Collects satellite imagery and generates ML-based flood estimates.
    """

    def __init__(self):
        self.service = SatelliteImageryService()
        self.extractor = FeatureExtractor()
        self.detector = FloodMLDetector()
        self.depth_estimator = DepthEstimator()

    async def collect_zone_flood_data(self, zone):
        """
        Fetch imagery and compute flood indicators.
        """
        try:
            # Fetch satellite image around zone
            image = self.service.fetch_satellite_image(
                lat=zone.center.latitude,
                lon=zone.center.longitude,
                radius_km=zone.radius_km
            )

            # Extract features
            features = self.extractor.extract_features(image)

            # Run flood classifier
            flood_prob = self.detector.predict_flood_probability(features)

            # Estimate water depth
            depth = self.depth_estimator.estimate_depth(features)

            return {
                "flood_probability": float(flood_prob),
                "estimated_water_depth": float(depth),
                "raw_features": features
            }

        except Exception as e:
            print(f"[Satellite Collector] ERROR: {e}")
            return None












# =====================================================================
# COLLECTOR ORCHESTRATOR
# =====================================================================
# =====================================================================
# COLLECTOR ORCHESTRATOR
# =====================================================================

class DataCollectionOrchestrator:
    """
    Orchestrates data collection from weather, social media, and satellite sources.
    Implements adaptive polling based on risk levels.
    """
    
    def __init__(
        self,
        weather_collector: WeatherAPICollector,
        social_collector: SocialMediaCollector
    ):
        """
        Initialize orchestrator.
        """
        self.weather_collector = weather_collector
        self.social_collector = social_collector
        self.satellite_collector = SatelliteImageryCollector()   # NEW

        # Adaptive polling intervals (seconds)
        self.polling_intervals = {
            'minimal': 1800,   # 30 minutes
            'low': 900,        # 15 minutes
            'moderate': 300,   # 5 minutes
            'high': 180,       # 3 minutes
            'critical': 60     # 1 minute
        }
        
        logger.info("DataCollectionOrchestrator initialized (with satellite support)")
    
    def get_polling_interval(self, zone: SentinelZone) -> int:
        """Return adaptive polling interval for a zone."""
        return self.polling_intervals.get(
            zone.risk_level.value,
            self.polling_intervals['moderate']
        )
    
    async def collect_zone_data(
        self,
        zone: SentinelZone
    ) -> Dict[str, Any]:
        """
        Collect weather, forecast, social media, and satellite ML data for a zone.
        """
        logger.info(f"Collecting data for zone: {zone.name}")

        # Collect concurrently
        weather_task = self.weather_collector.fetch_current_weather(zone.center)
        forecast_task = self.weather_collector.fetch_forecast(zone.center, hours=24)
        social_task = self.social_collector.fetch_recent_posts(zone, max_results=100)

        # NEW: Satellite ML fetch (runs synchronously but wrapped in async)
        satellite_task = asyncio.to_thread(self.satellite_collector.collect_zone_flood_data, zone)

        weather, forecast, social_posts, satellite_data = await asyncio.gather(
            weather_task,
            forecast_task,
            social_task,
            satellite_task,
            return_exceptions=True
        )

        return {
            "zone": zone,
            "weather": weather if not isinstance(weather, Exception) else None,
            "forecast": forecast if not isinstance(forecast, Exception) else [],
            "social_posts": social_posts if not isinstance(social_posts, Exception) else [],
            "satellite": satellite_data if not isinstance(satellite_data, Exception) else None,  # NEW
            "collected_at": datetime.utcnow()
        }
    
    async def collect_all_zones(
        self,
        zones: List[SentinelZone]
    ) -> List[Dict[str, Any]]:
        """
        Collect all data for all zones concurrently.
        """
        logger.info(f"Collecting data for {len(zones)} zones")

        tasks = [self.collect_zone_data(zone) for zone in zones]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: List[Dict[str, Any]] = [
            r for r in results if isinstance(r, dict)
        ]

        logger.info(
            f"Successfully collected data for {len(final_results)}/{len(zones)} zones"
        )

        return final_results
