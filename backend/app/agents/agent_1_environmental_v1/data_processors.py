"""
Data Processors for Environmental Intelligence Agent
====================================================
LLM-powered data enrichment, normalization, and analysis.
Processes raw weather and social media data for prediction models.

Author: Environmental Intelligence Team
Version: 1.0.0
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, cast
import openai
from openai import AsyncOpenAI
import json

from models import (
    WeatherData, SocialMediaPost, EnrichedSocialPost,
    PrecipitationData, WeatherCondition, SeverityLevel
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================================================
# LLM ENRICHMENT PROCESSOR
# =====================================================================

class LLMEnrichmentProcessor:
    """
    Uses OpenAI GPT to enrich social media posts with structured analysis.
    Extracts flood-relevant information, sentiment, and severity indicators.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 5
    ):
        """
        Initialize LLM enrichment processor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini for cost efficiency)
            max_concurrent: Maximum concurrent API calls
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Prompt template
        self.enrichment_prompt = """Analyze this social media post about potential flooding in Bangladesh.

Post: "{content}"
Author: {author}
Timestamp: {timestamp}
Location: {location}

Extract the following information in JSON format:
{{
    "relevance_score": <float 0-1, how relevant is this to urban flooding>,
    "sentiment": "<positive|negative|neutral|urgent>",
    "extracted_locations": [<list of specific location names mentioned>],
    "severity_indicators": [<list of words/phrases indicating flood severity>],
    "flood_keywords": [<list of flood-related keywords found>],
    "summary": "<brief 1-2 sentence summary>",
    "contains_flood_report": <true|false, is this an actual flood report>,
    "credibility_score": <float 0-1, estimated credibility based on content>,
    "water_depth_mentioned": <null or string if depth is mentioned>,
    "affected_areas": [<list of specific areas/neighborhoods mentioned>]
}}

Consider:
- Bangla language content (flood = বন্যা, water = পানি)
- Local location names in Dhaka/Bangladesh
- Urgency indicators (emergency, rescue, help needed)
- Specific details (water depth, duration, affected roads)
- Source credibility (verified accounts, specific details vs rumors)"""
        
        logger.info(f"LLMEnrichmentProcessor initialized with model: {model}")
    
    async def enrich_post(self, post: SocialMediaPost) -> EnrichedSocialPost:
        """
        Enrich a single social media post with LLM analysis.
        
        Args:
            post: Raw social media post
            
        Returns:
            Enriched post with LLM-extracted information
        """
        async with self.semaphore:
            try:
                # Format prompt
                prompt = self.enrichment_prompt.format(
                    content=post.content,
                    author=post.author,
                    timestamp=post.timestamp.isoformat(),
                    location=f"Lat: {post.location.latitude}, Lon: {post.location.longitude}" if post.location else "Unknown"
                )
                
                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in disaster response and flood analysis in Bangladesh. Extract structured information from social media posts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                content_str = None
                # Safely extract content; SDK versions may provide different structures
                try:
                    content_attr = getattr(response.choices[0], 'message', None)
                    if isinstance(content_attr, dict):
                        content_str = content_attr.get('content')
                    else:
                        content_str = getattr(content_attr, 'content', None)
                    # fallback to choice.text if present
                    if content_str is None:
                        content_str = getattr(response.choices[0], 'text', None)
                except Exception:
                    content_str = None

                if content_str is None:
                    logger.warning("Empty or missing LLM response content, using fallback enrichment")
                    return self._create_fallback_enrichment(post)

                try:
                    analysis = json.loads(content_str)
                except (TypeError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to parse LLM response content: {e}")
                    return self._create_fallback_enrichment(post)
                
                # Create enriched post
                enriched = EnrichedSocialPost(
                    **post.model_dump(),
                    enriched_at=datetime.utcnow(),
                    relevance_score=float(analysis.get('relevance_score', 0.5)),
                    sentiment=analysis.get('sentiment', 'neutral'),
                    extracted_locations=analysis.get('extracted_locations', []),
                    severity_indicators=analysis.get('severity_indicators', []),
                    flood_keywords=analysis.get('flood_keywords', []),
                    llm_summary=analysis.get('summary', ''),
                    contains_flood_report=bool(analysis.get('contains_flood_report', False)),
                    credibility_score=float(analysis.get('credibility_score', 0.5))
                )
                
                logger.debug(
                    f"Enriched post {post.platform_id}: "
                    f"relevance={enriched.relevance_score:.2f}, "
                    f"flood_report={enriched.contains_flood_report}"
                )
                
                return enriched
            
            except openai.RateLimitError:
                logger.error("OpenAI rate limit exceeded")
                # Return post with default enrichment
                return self._create_fallback_enrichment(post)
            
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return self._create_fallback_enrichment(post)
            
            except Exception as e:
                logger.error(f"Error enriching post: {e}", exc_info=True)
                return self._create_fallback_enrichment(post)
    
    def _create_fallback_enrichment(self, post: SocialMediaPost) -> EnrichedSocialPost:
        """Create basic enrichment using rule-based approach as fallback"""
        # Simple keyword matching
        content_lower = post.content.lower()
        
        flood_keywords = [
            kw for kw in ['flood', 'flooding', 'বন্যা', 'waterlogg', 'rain', 'water']
            if kw in content_lower
        ]
        
        # Estimate relevance based on keywords
        relevance_score = min(len(flood_keywords) * 0.25, 1.0)
        
        # Check for urgency indicators
        urgent_words = ['emergency', 'help', 'rescue', 'urgent', 'critical', 'danger']
        sentiment = 'urgent' if any(w in content_lower for w in urgent_words) else 'neutral'
        
        return EnrichedSocialPost(
            **post.model_dump(),
            enriched_at=datetime.utcnow(),
            relevance_score=relevance_score,
            sentiment=sentiment,
            extracted_locations=[],
            severity_indicators=urgent_words if sentiment == 'urgent' else [],
            flood_keywords=flood_keywords,
            llm_summary=post.content[:100] + "..." if len(post.content) > 100 else post.content,
            contains_flood_report=relevance_score > 0.5,
            credibility_score=0.5
        )
    
    async def enrich_posts_batch(
        self,
        posts: List[SocialMediaPost],
        filter_threshold: float = 0.3
    ) -> List[EnrichedSocialPost]:
        """
        Enrich multiple posts concurrently and filter by relevance.
        
        Args:
            posts: List of raw posts
            filter_threshold: Minimum relevance score to keep
            
        Returns:
            List of enriched posts above threshold
        """
        if not posts:
            return []
        
        logger.info(f"Enriching {len(posts)} posts")
        
        # Enrich all posts concurrently
        tasks = [self.enrich_post(post) for post in posts]
        enriched_posts = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and low-relevance posts
        valid_posts = []
        for post in enriched_posts:
            # skip exceptions returned by asyncio.gather
            if isinstance(post, Exception):
                continue
            # Use getattr to safely access relevance_score (avoids attribute lookup on BaseException)
            relevance = getattr(post, 'relevance_score', None)
            if relevance is not None and relevance >= filter_threshold:
                valid_posts.append(post)
        
        logger.info(
            f"Enrichment complete: {len(valid_posts)}/{len(posts)} posts passed filter "
            f"(threshold={filter_threshold})"
        )
        
        return valid_posts


# =====================================================================
# WEATHER DATA NORMALIZER
# =====================================================================

class WeatherDataNormalizer:
    """
    Normalizes weather data for consistent analysis.
    Converts measurements to normalized scales (0-1).
    """
    
    def __init__(self):
        """Initialize normalizer with reference values for normalization"""
        # Reference values for normalization (based on Bangladesh climate)
        self.reference_values = {
            'temperature': {'min': 15, 'max': 40},  # Celsius
            'humidity': {'min': 40, 'max': 100},    # Percentage
            'pressure': {'min': 980, 'max': 1020},  # hPa
            'wind_speed': {'min': 0, 'max': 20},    # m/s
            'rainfall_1h': {'min': 0, 'max': 50},   # mm (heavy rain threshold)
            'rainfall_3h': {'min': 0, 'max': 100},  # mm
            'rainfall_24h': {'min': 0, 'max': 200}, # mm (extreme)
            'cloud_coverage': {'min': 0, 'max': 100}  # Percentage
        }
        
        logger.info("WeatherDataNormalizer initialized")
    
    def normalize_value(
        self,
        value: float,
        metric: str,
        clamp: bool = True
    ) -> float:
        """
        Normalize a value to 0-1 range.
        
        Args:
            value: Value to normalize
            metric: Metric name (must be in reference_values)
            clamp: Whether to clamp result to [0, 1]
            
        Returns:
            Normalized value
        """
        if metric not in self.reference_values:
            logger.warning(f"Unknown metric: {metric}, returning raw value")
            return value
        
        ref = self.reference_values[metric]
        min_val, max_val = ref['min'], ref['max']
        
        # Min-max normalization
        normalized = (value - min_val) / (max_val - min_val)
        
        if clamp:
            normalized = max(0.0, min(1.0, normalized))
        
        return normalized
    
    def calculate_rainfall_intensity(self, precip: PrecipitationData) -> float:
        """
        Calculate normalized rainfall intensity (0-1).
        
        Args:
            precip: Precipitation data
            
        Returns:
            Normalized intensity score
        """
        # Priority: 1h > 3h > 24h
        if precip.rain_1h is not None and precip.rain_1h > 0:
            return self.normalize_value(precip.rain_1h, 'rainfall_1h')
        elif precip.rain_3h is not None and precip.rain_3h > 0:
            # Convert to hourly equivalent
            hourly_equiv = precip.rain_3h / 3
            return self.normalize_value(hourly_equiv, 'rainfall_1h')
        elif precip.rain_24h is not None and precip.rain_24h > 0:
            # Convert to hourly equivalent
            hourly_equiv = precip.rain_24h / 24
            return self.normalize_value(hourly_equiv, 'rainfall_1h')
        
        return 0.0
    
    def calculate_accumulated_rainfall(self, precip: PrecipitationData) -> float:
        """
        Calculate normalized accumulated rainfall (0-1).
        
        Args:
            precip: Precipitation data
            
        Returns:
            Normalized accumulation score
        """
        total = precip.total_rain
        return self.normalize_value(total, 'rainfall_24h')
    
    def calculate_weather_severity(self, weather: WeatherData) -> float:
        """
        Calculate overall weather severity score (0-1).
        
        Args:
            weather: Weather data
            
        Returns:
            Severity score combining multiple factors
        """
        # Condition severity mapping
        condition_severity = {
            WeatherCondition.CLEAR: 0.0,
            WeatherCondition.CLOUDS: 0.1,
            WeatherCondition.MIST: 0.2,
            WeatherCondition.FOG: 0.2,
            WeatherCondition.DRIZZLE: 0.3,
            WeatherCondition.RAIN: 0.5,
            WeatherCondition.HEAVY_RAIN: 0.8,
            WeatherCondition.THUNDERSTORM: 0.9,
            WeatherCondition.SNOW: 0.4
        }
        
        condition_score = condition_severity.get(weather.condition, 0.5)
        
        # Factor in other metrics
        rainfall_score = self.calculate_rainfall_intensity(weather.precipitation)
        wind_score = self.normalize_value(weather.metrics.wind_speed, 'wind_speed')
        humidity_score = self.normalize_value(weather.metrics.humidity, 'humidity')
        
        # Weighted combination
        severity = (
            condition_score * 0.4 +
            rainfall_score * 0.4 +
            wind_score * 0.1 +
            humidity_score * 0.1
        )
        
        return min(1.0, severity)
    
    def normalize_weather_data(self, weather: WeatherData) -> Dict[str, float]:
        """
        Normalize all weather metrics.
        
        Args:
            weather: Raw weather data
            
        Returns:
            Dictionary of normalized values
        """
        return {
            'temperature': self.normalize_value(
                weather.metrics.temperature, 'temperature'
            ),
            'humidity': self.normalize_value(
                weather.metrics.humidity, 'humidity'
            ),
            'pressure': self.normalize_value(
                weather.metrics.pressure, 'pressure'
            ),
            'wind_speed': self.normalize_value(
                weather.metrics.wind_speed, 'wind_speed'
            ),
            'cloud_coverage': self.normalize_value(
                weather.metrics.cloud_coverage, 'cloud_coverage'
            ),
            'rainfall_intensity': self.calculate_rainfall_intensity(
                weather.precipitation
            ),
            'accumulated_rainfall': self.calculate_accumulated_rainfall(
                weather.precipitation
            ),
            'weather_severity': self.calculate_weather_severity(weather)
        }


# =====================================================================
# SOCIAL MEDIA ANALYZER
# =====================================================================

class SocialMediaAnalyzer:
    """
    Analyzes enriched social media posts for flood patterns.
    Aggregates multiple posts to assess situation severity.
    """
    
    def __init__(self):
        """Initialize analyzer"""
        logger.info("SocialMediaAnalyzer initialized")
    
    def calculate_report_density(
        self,
        posts: List[EnrichedSocialPost],
        area_km2: float
    ) -> float:
        """
        Calculate density of flood reports in an area.
        
        Args:
            posts: Enriched posts
            area_km2: Area size in square kilometers
            
        Returns:
            Normalized report density (0-1)
        """
        if area_km2 <= 0:
            return 0.0
        
        # Count high-relevance flood reports
        flood_reports = [
            p for p in posts
            if p.contains_flood_report and p.relevance_score >= 0.6
        ]
        
        # Calculate density (reports per km²)
        density = len(flood_reports) / area_km2
        
        # Normalize (assume 1 report/km² is very high)
        normalized = min(density, 1.0)
        
        return normalized
    
    def calculate_urgency_score(self, posts: List[EnrichedSocialPost]) -> float:
        """
        Calculate overall urgency from posts.
        
        Args:
            posts: Enriched posts
            
        Returns:
            Urgency score (0-1)
        """
        if not posts:
            return 0.0
        
        urgent_posts = [p for p in posts if p.sentiment == 'urgent']
        negative_posts = [p for p in posts if p.sentiment == 'negative']
        
        # Weight urgent posts more heavily
        urgency = (
            (len(urgent_posts) * 1.0 + len(negative_posts) * 0.5) /
            len(posts)
        )
        
        return min(urgency, 1.0)
    
    def extract_severity_indicators(
        self,
        posts: List[EnrichedSocialPost]
    ) -> Dict[str, int]:
        """
        Extract and count severity indicators from posts.
        
        Args:
            posts: Enriched posts
            
        Returns:
            Dictionary of indicator counts
        """
        indicators = {}
        
        for post in posts:
            for indicator in post.severity_indicators:
                indicator_lower = indicator.lower()
                indicators[indicator_lower] = indicators.get(indicator_lower, 0) + 1
        
        return dict(sorted(indicators.items(), key=lambda x: x[1], reverse=True))
    
    def aggregate_credibility(self, posts: List[EnrichedSocialPost]) -> float:
        """
        Calculate aggregate credibility score.
        
        Args:
            posts: Enriched posts
            
        Returns:
            Average credibility (0-1)
        """
        if not posts:
            return 0.0
        
        # Weight by relevance
        weighted_sum = sum(
            p.credibility_score * p.relevance_score
            for p in posts
        )
        weight_total = sum(p.relevance_score for p in posts)
        
        if weight_total == 0:
            return 0.0
        
        return weighted_sum / weight_total
    
    def analyze_posts_for_zone(
        self,
        posts: List[EnrichedSocialPost],
        area_km2: float
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of posts for a zone.
        
        Args:
            posts: Enriched posts
            area_km2: Zone area
            
        Returns:
            Analysis results dictionary
        """
        # Filter to high-relevance posts
        relevant_posts = [p for p in posts if p.relevance_score >= 0.5]
        
        return {
            'total_posts': len(posts),
            'relevant_posts': len(relevant_posts),
            'flood_reports': len([p for p in relevant_posts if p.contains_flood_report]),
            'report_density': self.calculate_report_density(relevant_posts, area_km2),
            'urgency_score': self.calculate_urgency_score(relevant_posts),
            'average_credibility': self.aggregate_credibility(relevant_posts),
            'severity_indicators': self.extract_severity_indicators(relevant_posts),
            'most_mentioned_locations': self._extract_top_locations(relevant_posts, top_n=5)
        }
    
    def _extract_top_locations(
        self,
        posts: List[EnrichedSocialPost],
        top_n: int = 5
    ) -> List[Tuple[str, int]]:
        """Extract most frequently mentioned locations"""
        location_counts = {}
        
        for post in posts:
            for location in post.extracted_locations:
                loc_lower = location.lower()
                location_counts[loc_lower] = location_counts.get(loc_lower, 0) + 1
        
        # Return top N
        sorted_locations = sorted(
            location_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_locations[:top_n]


# =====================================================================
# DATA PROCESSOR ORCHESTRATOR
# =====================================================================

class DataProcessingOrchestrator:
    """
    Orchestrates all data processing operations.
    Coordinates LLM enrichment, normalization, and analysis.
    """
    
    def __init__(
        self,
        llm_processor: LLMEnrichmentProcessor,
        weather_normalizer: WeatherDataNormalizer,
        social_analyzer: SocialMediaAnalyzer
    ):
        """
        Initialize orchestrator.
        
        Args:
            llm_processor: LLM enrichment processor
            weather_normalizer: Weather data normalizer
            social_analyzer: Social media analyzer
        """
        self.llm_processor = llm_processor
        self.weather_normalizer = weather_normalizer
        self.social_analyzer = social_analyzer
        
        logger.info("DataProcessingOrchestrator initialized")
    
    async def process_zone_data(
        self,
        zone_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process all collected data for a zone.
        
        Args:
            zone_data: Raw collected data (from DataCollectionOrchestrator)
            
        Returns:
            Processed and enriched data
        """
        zone = zone_data['zone']
        weather = zone_data.get('weather')
        social_posts = zone_data.get('social_posts', [])
        
        logger.info(f"Processing data for zone: {zone.name}")
        
        # Process weather data
        normalized_weather = None
        if weather:
            normalized_weather = self.weather_normalizer.normalize_weather_data(weather)
        
        # Enrich social media posts
        enriched_posts = await self.llm_processor.enrich_posts_batch(
            social_posts,
            filter_threshold=0.3
        )
        
        # Analyze social media
        social_analysis = self.social_analyzer.analyze_posts_for_zone(
            enriched_posts,
            zone.radius_km ** 2 * 3.14159  # Approximate area
        )
        
        return {
            'zone': zone,
            'weather': weather,
            'normalized_weather': normalized_weather,
            'enriched_posts': enriched_posts,
            'social_analysis': social_analysis,
            'processed_at': datetime.utcnow()
        }
    
    async def process_all_zones(
        self,
        collected_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process data for all zones concurrently.
        
        Args:
            collected_data: List of collected zone data
            
        Returns:
            List of processed zone data
        """
        logger.info(f"Processing data for {len(collected_data)} zones")
        
        tasks = [
            self.process_zone_data(zone_data)
            for zone_data in collected_data
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and cast to the expected return type for static checkers
        valid_results = cast(List[Dict[str, Any]], [r for r in results if not isinstance(r, Exception)])
        
        logger.info(
            f"Processing complete: {len(valid_results)}/{len(collected_data)} zones"
        )
        
        return valid_results