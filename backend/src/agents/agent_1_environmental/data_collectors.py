# backend/src/agents/agent_1_environmental/data_collectors.py

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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, cast
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
import tweepy
from redis import asyncio as aioredis
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))



from backend.src.agents.agent_1_environmental.models import (
    WeatherData, WeatherMetrics, PrecipitationData, WeatherCondition,
    SocialMediaPost, GeoPoint, SentinelZone, DataSource, BoundingBox
)
from backend.src.agents.agent_1_environmental.services.satellite_service import SatelliteDataCollector, SatelliteData

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
        now = datetime.now(timezone.utc)
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
            timestamp=datetime.fromtimestamp(data.get('dt', datetime.now(timezone.utc).timestamp()), tz=timezone.utc),
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
        start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        
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
            logger.warning("Twitter API rate limit exceeded — skipping social data")
            return []
        except tweepy.Unauthorized:
            logger.warning("Twitter API credentials invalid or missing — skipping social data")
            return []
        except tweepy.TwitterServerError as e:
            logger.warning(f"Twitter server error — skipping social data: {e}")
            return []
        except Exception as e:
            logger.warning(f"Twitter unavailable — skipping social data: {e}")
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
# RIVER LEVEL COLLECTOR (GloFAS via Open-Meteo Flood API)
# =====================================================================

class RiverLevelCollector:
    """
    Collects river discharge data from the Open-Meteo Flood API,
    which is powered by GloFAS v4 (Global Flood Awareness System).

    No API key required — free and open.
    Endpoint: https://flood-api.open-meteo.com/v1/flood

    For each zone it returns:
      - current_discharge_m3s  : today's river discharge (m³/s)
      - past_7day_avg_m3s      : 7-day rolling average (baseline)
      - forecast_7day_peak_m3s : highest forecast discharge in next 7 days
      - discharge_ratio        : current / past_avg (>1.5 = elevated, >2.5 = high risk)
      - trend                  : 'rising', 'stable', or 'falling'
      - risk_factor            : normalized 0-1 score for the predictor
    """

    # Flood stage thresholds for Bangladesh rivers (m³/s).
    # These represent typical values — rivers feeding the Brahmaputra/Jamuna
    # basin are much larger than those feeding the Surma/Barak basin.
    # The risk_factor is computed from discharge_ratio so it stays
    # scale-independent, making these thresholds only used for logging.
    FLOOD_STAGE_APPROX = {
        "Sunamganj Sadar": 5_000,    # Surma river
        "Sylhet City":     4_000,    # Surma/Kushiyara
        "Netrokona Sadar": 3_000,    # Old Brahmaputra
        "Sirajganj Sadar": 50_000,   # Jamuna (main channel)
        "Jamalpur Sadar":  40_000,   # Jamuna/Brahmaputra
    }

    API_URL = "https://flood-api.open-meteo.com/v1/flood"

    def __init__(
        self,
        cache_client=None,
        cache_ttl: int = 3600,      # 1 hour — GloFAS updates daily
        request_timeout: int = 15,
    ):
        self.cache_client = cache_client
        self.cache_ttl    = cache_ttl
        self.timeout      = aiohttp.ClientTimeout(total=request_timeout)
        logger.info("RiverLevelCollector initialized (GloFAS v4 via Open-Meteo)")

    def _compute_risk_factor(
        self,
        current: float,
        past_avg: float,
        forecast_peak: float,
    ) -> float:
        """
        Convert raw discharge numbers into a 0-1 risk factor.

        Uses the discharge ratio (current / past_7day_avg) as the
        primary signal so the factor is scale-independent across the
        vastly different river sizes in Bangladesh.

        Mapping:
          ratio < 1.0  → 0.0  (discharge falling / below average)
          ratio = 1.0  → 0.10 (normal flow)
          ratio = 1.5  → 0.30 (elevated — worth watching)
          ratio = 2.0  → 0.50 (significantly elevated)
          ratio = 3.0  → 0.70 (high flood risk)
          ratio = 5.0  → 0.90 (very high)
          ratio >= 7.0 → 1.00 (extreme)

        A secondary boost is applied when the 7-day forecast peak
        is also elevated (rising trend confirmed by model).
        """
        if past_avg <= 0:
            return 0.0

        ratio = current / past_avg

        if ratio < 1.0:
            base = 0.0
        elif ratio < 1.5:
            base = 0.0 + ((ratio - 1.0) / 0.5) * 0.30
        elif ratio < 2.0:
            base = 0.30 + ((ratio - 1.5) / 0.5) * 0.20
        elif ratio < 3.0:
            base = 0.50 + ((ratio - 2.0) / 1.0) * 0.20
        elif ratio < 5.0:
            base = 0.70 + ((ratio - 3.0) / 2.0) * 0.20
        elif ratio < 7.0:
            base = 0.90 + ((ratio - 5.0) / 2.0) * 0.05
        else:
            base = min(0.95 + ((ratio - 7.0) / 5.0) * 0.05, 1.0)

        # Boost if forecast peak is also elevated (trend confirmation)
        if past_avg > 0 and forecast_peak > 0:
            forecast_ratio = forecast_peak / past_avg
            if forecast_ratio > ratio:          # river still rising
                boost = min((forecast_ratio - ratio) / 5.0, 0.10)
                base  = min(base + boost, 1.0)

        return round(base, 4)

    @staticmethod
    def _compute_trend(values: list[float]) -> str:
        """Determine trend from a time-ordered list of discharge values."""
        if len(values) < 3:
            return "stable"
        recent = values[-3:]
        if recent[-1] > recent[0] * 1.05:
            return "rising"
        elif recent[-1] < recent[0] * 0.95:
            return "falling"
        return "stable"

    async def fetch_river_data(self, zone) -> Optional[dict]:
        """
        Fetch GloFAS river discharge for a single zone.

        Returns a dict with keys:
            zone_name, current_discharge_m3s, past_7day_avg_m3s,
            forecast_7day_peak_m3s, discharge_ratio, trend, risk_factor
        Or None if the API is unreachable or returns no data.
        """
        cache_key = f"river:{zone.id}"

        # Check cache first
        if self.cache_client:
            try:
                cached = await self.cache_client.get(cache_key)
                if cached:
                    import json
                    return json.loads(cached)
            except Exception:
                pass  # Cache miss — continue to API

        params = {
            "latitude":      zone.center.latitude,
            "longitude":     zone.center.longitude,
            "daily":         "river_discharge",
            "past_days":     7,
            "forecast_days": 7,
        }

        try:
            async with ClientSession(timeout=self.timeout) as session:
                async with session.get(self.API_URL, params=params) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            f"[{zone.name}] GloFAS API {resp.status}: {body[:120]}"
                        )
                        return None

                    data = await resp.json()

        except asyncio.TimeoutError:
            logger.warning(f"[{zone.name}] GloFAS request timed out")
            return None
        except ClientError as e:
            logger.warning(f"[{zone.name}] GloFAS connection error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[{zone.name}] GloFAS unexpected error: {e}")
            return None

        # Parse response
        daily      = data.get("daily", {})
        times      = daily.get("time", [])
        discharges = daily.get("river_discharge", [])

        if not times or not discharges:
            logger.warning(
                f"[{zone.name}] GloFAS returned empty data — "
                "river may be outside 5 km grid cell. "
                "Consider shifting coordinates by ±0.1°."
            )
            return None

        today = datetime.now(timezone.utc).date().isoformat()

        past_vals     = []
        current_val   = None
        forecast_vals = []

        for t, d in zip(times, discharges):
            if d is None:
                continue
            if t < today:
                past_vals.append(d)
            elif t == today:
                current_val = d
            else:
                forecast_vals.append(d)

        if current_val is None and past_vals:
            current_val = past_vals[-1]   # Use most recent past if today missing

        if current_val is None:
            logger.warning(f"[{zone.name}] GloFAS: no current discharge value")
            return None

        past_avg      = sum(past_vals) / len(past_vals) if past_vals else current_val
        forecast_peak = max(forecast_vals) if forecast_vals else current_val
        ratio         = current_val / past_avg if past_avg > 0 else 1.0
        trend         = self._compute_trend(past_vals + [current_val])
        risk_factor   = self._compute_risk_factor(current_val, past_avg, forecast_peak)

        result = {
            "zone_name":               zone.name,
            "current_discharge_m3s":   round(current_val,   1),
            "past_7day_avg_m3s":       round(past_avg,      1),
            "forecast_7day_peak_m3s":  round(forecast_peak, 1),
            "discharge_ratio":         round(ratio,         3),
            "trend":                   trend,
            "risk_factor":             risk_factor,
            "actual_lat":              data.get("latitude",  zone.center.latitude),
            "actual_lon":              data.get("longitude", zone.center.longitude),
        }

        logger.info(
            f"  [{zone.name}] River: {current_val:.1f} m³/s "
            f"(avg={past_avg:.1f}, ratio={ratio:.2f}x, "
            f"trend={trend}, risk={risk_factor:.3f})"
        )

        # Cache result
        if self.cache_client:
            try:
                import json
                await self.cache_client.setex(
                    cache_key, self.cache_ttl, json.dumps(result)
                )
            except Exception:
                pass

        return result

    async def fetch_multiple_zones(self, zones: list) -> dict[str, Optional[dict]]:
        """Fetch river data for all zones concurrently."""
        tasks   = [self.fetch_river_data(zone) for zone in zones]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        zone_data: dict[str, Optional[dict]] = {}
        for zone, result in zip(zones, results):
            zid = str(zone.id)
            if isinstance(result, Exception):
                logger.error(f"River fetch error for {zone.name}: {result}")
                zone_data[zid] = None
            elif isinstance(result, dict):
                zone_data[zid] = result
            else:
                zone_data[zid] = None

        return zone_data

# =====================================================================
# COLLECTOR ORCHESTRATOR
# =====================================================================

class DataCollectionOrchestrator:
    """
    Orchestrates data collection from multiple sources.
    Implements adaptive polling based on risk levels.
    """
    
    def __init__(
        self,
        weather_collector: WeatherAPICollector,
        social_collector: Optional[SocialMediaCollector],  # FIX: Optional — may be None when token missing
        satellite_collector: Optional[SatelliteDataCollector] = None,
        river_collector: Optional[RiverLevelCollector] = None,    # 9th factor
    ):
        """
        Initialize orchestrator.

        Args:
            weather_collector: Weather API collector instance
            social_collector: Social media collector instance (None if token not configured)
            satellite_collector: Satellite imagery collector instance (optional)
            river_collector: River level collector instance (optional)
        """
        self.weather_collector = weather_collector
        self.social_collector = social_collector
        self.satellite_collector = satellite_collector
        self.river_collector = river_collector
        
        # Adaptive polling intervals (seconds)
        self.polling_intervals = {
            'minimal': 1800,   # 30 minutes
            'low': 900,        # 15 minutes
            'moderate': 300,   # 5 minutes
            'high': 180,       # 3 minutes
            'critical': 60     # 1 minute
        }
        
        logger.info("DataCollectionOrchestrator initialized")
    
    def get_polling_interval(self, zone: SentinelZone) -> int:
        """
        Get adaptive polling interval for a zone based on risk level.
        
        Args:
            zone: Sentinel zone
            
        Returns:
            Polling interval in seconds
        """
        return self.polling_intervals.get(
            zone.risk_level.value,
            self.polling_intervals['moderate']
        )
    
    async def collect_zone_data(
        self,
        zone: SentinelZone
    ) -> Dict[str, Any]:
        """
        Collect all data for a single zone.
        
        Args:
            zone: Sentinel zone to collect data for
            
        Returns:
            Dictionary with weather, social media, AND satellite data
        """
        logger.info(f"Collecting data for zone: {zone.name}")
        
        # Build task list — weather always, social only if collector available
        weather_task = self.weather_collector.fetch_current_weather(zone.center)
        forecast_task = self.weather_collector.fetch_forecast(zone.center, hours=24)

        # FIX 2: social_collector may be None when Twitter token is not configured
        if self.social_collector is not None and self.satellite_collector is not None:
            satellite_task = self.satellite_collector.fetch_satellite_data(zone, str(zone.id))
            weather, forecast, social_posts, satellite = await asyncio.gather(
                weather_task, forecast_task,
                self.social_collector.fetch_recent_posts(zone, max_results=100),
                satellite_task,
                return_exceptions=True
            )
        elif self.social_collector is not None:
            weather, forecast, social_posts = await asyncio.gather(
                weather_task, forecast_task,
                self.social_collector.fetch_recent_posts(zone, max_results=100),
                return_exceptions=True
            )
            satellite = None
        elif self.satellite_collector is not None:
            satellite_task = self.satellite_collector.fetch_satellite_data(zone, str(zone.id))
            weather, forecast, satellite = await asyncio.gather(
                weather_task, forecast_task, satellite_task,
                return_exceptions=True
            )
            social_posts = []
        else:
            weather, forecast = await asyncio.gather(
                weather_task, forecast_task,
                return_exceptions=True
            )
            social_posts = []
            satellite = None
        
        # Fetch river level data independently (fast HTTP, always async)
        river_data = None
        if self.river_collector is not None:
            try:
                river_data = await self.river_collector.fetch_river_data(zone)
            except Exception as e:
                logger.warning(f"  [{zone.name}] River data fetch failed: {e}")

        # Build merged result
        result = {
            'zone': zone,
            'weather': weather if not isinstance(weather, Exception) else None,
            'forecast': forecast if not isinstance(forecast, Exception) else [],
            'social_posts': social_posts if not isinstance(social_posts, Exception) else [],
            'satellite': satellite if (satellite is not None and not isinstance(satellite, Exception)) else None,
            'river': river_data if (river_data is not None and not isinstance(river_data, Exception)) else None, # GloFAS discharge dict or None
            'collected_at': datetime.now(timezone.utc)
        }
        
        # Log satellite status for this zone
        sat_data = result.get('satellite')
        if sat_data and hasattr(sat_data, 'sar_available') and sat_data.sar_available:
            fd = sat_data.flood_detection
            logger.info(
                f"  [{zone.name}] Satellite: {fd.risk_level} risk, "
                f"{fd.flood_percentage:.1f}% flood ({fd.flood_area_km2:.2f} km²)"
            )
        elif sat_data:
            logger.info(f"  [{zone.name}] Satellite: data unavailable ({getattr(sat_data.flood_detection, 'error', 'unknown')})")
        else:
            logger.info(f"  [{zone.name}] Satellite: collector not configured")
        
        return result
    
    async def collect_all_zones(
        self,
        zones: List[SentinelZone]
    ) -> List[Dict[str, Any]]:
        """
        Collect data for all zones concurrently.
        
        Args:
            zones: List of sentinel zones
            
        Returns:
            List of collection results
        """
        logger.info(f"Collecting data for {len(zones)} zones")
        
        tasks = [self.collect_zone_data(zone) for zone in zones]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Keep only dict results (exclude exceptions) and ensure proper typing
        final_results: List[Dict[str, Any]] = [
            r for r in results if isinstance(r, dict)
        ]
        
        logger.info(
            f"Successfully collected data for {len(final_results)}/{len(zones)} zones"
        )
        
        return final_results