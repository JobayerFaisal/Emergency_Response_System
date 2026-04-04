"""
OpenAI-Powered Flood Detection Processor
=========================================
Analyzes tweets using GPT-3.5-turbo for flood event detection and classification.
Supports batch processing for cost optimization.

Author: Mahmudul Hasan
Project: Autonomous Multi-Agent System for Urban Flood Response
Component: Agent 1 - Environmental Intelligence (Social Media)
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from openai import OpenAI, RateLimitError, APIError
from openai.types.chat import ChatCompletionMessageParam

from config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL, 
    DETECTION_CONFIG,
    DHAKA_ZONES,
)


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class ExtractedInfo:
    """Additional extracted information from tweet"""
    water_level_mentioned: Optional[str] = None
    affected_infrastructure: List[str] = field(default_factory=list)
    time_reference: Optional[str] = None
    contact_info: Optional[str] = None


@dataclass
class FloodDetectionResult:
    """Complete flood detection result for a single tweet"""
    tweet_id: str
    original_text: str
    is_flood_related: bool
    confidence: float
    flood_severity: str  # none|minor|moderate|severe|critical
    location_mentioned: Optional[str]
    zone_detected: Optional[str]
    flood_type: str  # waterlogging|river_overflow|flash_flood|drainage_failure|unknown
    urgency_level: str  # low|medium|high|critical
    needs_rescue: bool
    extracted_info: ExtractedInfo
    processed_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['extracted_info'] = asdict(self.extracted_info)
        return result


# =============================================================================
# OPENAI PROMPT ENGINEERING
# =============================================================================

SYSTEM_PROMPT: str = """You are an expert flood detection AI system specialized in analyzing social media posts from Dhaka, Bangladesh. Your task is to analyze tweets and determine if they report flood events.

You understand:
- Bengali (বাংলা), English, and mixed "Banglish" text
- Local area names in Dhaka (Mirpur, Uttara, Mohammadpur, Dhanmondi, Badda, Gulshan, etc.)
- Flood-related terminology in both languages
- The difference between actual flood reports vs. general weather/rain mentions

KNOWN FLOOD-PRONE ZONES IN DHAKA:
- Mirpur (মিরপুর) - including Pallabi, Kazipara
- Uttara (উত্তরা) - including Azampur, Diabari  
- Mohammadpur (মোহাম্মদপুর) - including Shyamoli, Adabor
- Dhanmondi (ধানমন্ডি) - including Kalabagan
- Badda (বাড্ডা) - including Gulshan, Baridhara

CLASSIFICATION GUIDELINES:
1. is_flood_related: TRUE only if the tweet reports actual flooding, waterlogging, or water-related emergency
2. flood_severity:
   - "critical": Life-threatening, people trapped, need immediate rescue
   - "severe": Significant flooding, property damage, evacuation needed
   - "moderate": Notable waterlogging, traffic disruption, some property at risk
   - "minor": Light waterlogging, minor inconvenience
   - "none": Not flood-related
3. needs_rescue: TRUE only if someone explicitly needs help/evacuation
4. urgency_level: Based on immediate threat to life/property

OUTPUT FORMAT: Return ONLY valid JSON array, no markdown or explanation."""


def create_batch_prompt(tweets: List[Dict[str, Any]]) -> str:
    """Create prompt for batch tweet analysis"""
    
    tweet_list = "\n".join([
        f'{i+1}. [ID: {t["id"]}] "{t["text"]}"' 
        for i, t in enumerate(tweets)
    ])
    
    return f"""Analyze these {len(tweets)} tweets for flood detection. Return a JSON array with one object per tweet.

TWEETS TO ANALYZE:
{tweet_list}

For each tweet, return:
{{
    "tweet_id": "the tweet ID",
    "is_flood_related": true/false,
    "confidence": 0.0-1.0,
    "flood_severity": "none|minor|moderate|severe|critical",
    "location_mentioned": "location name or null",
    "zone_detected": "mirpur|uttara|mohammadpur|dhanmondi|badda|null",
    "flood_type": "waterlogging|river_overflow|flash_flood|drainage_failure|unknown|none",
    "urgency_level": "low|medium|high|critical",
    "needs_rescue": true/false,
    "water_level_mentioned": "description or null",
    "affected_infrastructure": ["list", "of", "items"],
    "time_reference": "time mentioned or null"
}}

Return ONLY the JSON array, no other text:"""


# =============================================================================
# OPENAI API INTERFACE
# =============================================================================

class OpenAIFloodDetector:
    """
    OpenAI-powered flood detection processor
    
    Features:
    - Batch processing for cost optimization
    - Automatic retry with exponential backoff
    - Structured output parsing
    - Cost tracking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key: str = api_key if api_key is not None else OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model: str = OPENAI_MODEL
        self.config: Dict[str, Any] = DETECTION_CONFIG
        
        # Token tracking for cost estimation
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_calls: int = 0
        
    def _call_openai(self, messages: List[ChatCompletionMessageParam], retries: int = 3) -> Optional[str]:
        """Make OpenAI API call with retry logic"""
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent classification
                    max_tokens=2000
                )
                
                # Track tokens
                if response.usage is not None:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                self.api_calls += 1
                
                content = response.choices[0].message.content
                return content if content is not None else None
                
            except RateLimitError:
                wait_time = (2 ** attempt) * 5
                print(f"⏳ Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except APIError as e:
                print(f"⚠️ API Error (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2)
                
        return None
    
    def _parse_response(self, response: Optional[str], tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse OpenAI response into structured results"""
        
        if response is None:
            return [{"tweet_id": t["id"], "error": "no_response"} for t in tweets]
        
        try:
            # Clean response - remove markdown if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                parts = clean_response.split("```")
                if len(parts) > 1:
                    clean_response = parts[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            
            results = json.loads(clean_response)
            
            if not isinstance(results, list):
                results = [results]
                
            return results
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"Response was: {response[:500]}...")
            # Return empty results for failed parse
            return [{"tweet_id": t["id"], "error": "parse_failed"} for t in tweets]
    
    def process_tweets(
        self, 
        tweets: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[FloodDetectionResult]:
        """
        Process tweets in batches for flood detection
        
        Args:
            tweets: List of tweet dictionaries (Twitter API v2 format)
            batch_size: Number of tweets per API call (default from config)
            
        Returns:
            List of FloodDetectionResult objects
        """
        
        actual_batch_size: int = batch_size if batch_size is not None else int(self.config["batch_size"])
        all_results: List[FloodDetectionResult] = []
        
        total_batches = (len(tweets) + actual_batch_size - 1) // actual_batch_size
        
        print(f"\n🔄 Processing {len(tweets)} tweets in {total_batches} batches...")
        print("-" * 50)
        
        for i in range(0, len(tweets), actual_batch_size):
            batch = tweets[i:i+actual_batch_size]
            batch_num = (i // actual_batch_size) + 1
            
            print(f"📦 Batch {batch_num}/{total_batches} ({len(batch)} tweets)...", end=" ")
            
            # Create messages for API
            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": create_batch_prompt(batch)}
            ]
            
            # Call OpenAI
            response = self._call_openai(messages)
            
            if response is not None:
                parsed = self._parse_response(response, batch)
                
                # Convert to structured results
                for j, result_dict in enumerate(parsed):
                    tweet = batch[j] if j < len(batch) else batch[-1]
                    
                    try:
                        # Handle affected_infrastructure - ensure it's a list
                        affected_infra = result_dict.get("affected_infrastructure")
                        if affected_infra is None:
                            affected_infra = []
                        elif not isinstance(affected_infra, list):
                            affected_infra = [str(affected_infra)]
                        
                        result = FloodDetectionResult(
                            tweet_id=str(result_dict.get("tweet_id", tweet["id"])),
                            original_text=str(tweet["text"]),
                            is_flood_related=bool(result_dict.get("is_flood_related", False)),
                            confidence=float(result_dict.get("confidence", 0.5)),
                            flood_severity=str(result_dict.get("flood_severity", "none")),
                            location_mentioned=result_dict.get("location_mentioned"),
                            zone_detected=result_dict.get("zone_detected"),
                            flood_type=str(result_dict.get("flood_type", "unknown")),
                            urgency_level=str(result_dict.get("urgency_level", "low")),
                            needs_rescue=bool(result_dict.get("needs_rescue", False)),
                            extracted_info=ExtractedInfo(
                                water_level_mentioned=result_dict.get("water_level_mentioned"),
                                affected_infrastructure=affected_infra,
                                time_reference=result_dict.get("time_reference"),
                                contact_info=result_dict.get("contact_info")
                            ),
                            processed_at=datetime.now().isoformat()
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"\n⚠️ Error creating result for tweet {tweet['id']}: {e}")
                
                print("✅ Done")
            else:
                print("❌ Failed")
            
            # Small delay between batches to avoid rate limits
            if i + actual_batch_size < len(tweets):
                time.sleep(0.5)
        
        return all_results
    
    def get_cost_estimate(self) -> Dict[str, Union[int, float]]:
        """
        Calculate estimated API cost
        
        GPT-3.5-turbo pricing (as of 2024):
        - Input: $0.0005 / 1K tokens
        - Output: $0.0015 / 1K tokens
        """
        input_cost = (self.total_input_tokens / 1000) * 0.0005
        output_cost = (self.total_output_tokens / 1000) * 0.0015
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "api_calls": self.api_calls,
            "estimated_cost_usd": round(input_cost + output_cost, 4)
        }


# =============================================================================
# RESULT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_results(results: List[FloodDetectionResult]) -> Dict[str, Any]:
    """Generate comprehensive analysis of detection results"""
    
    total = len(results)
    if total == 0:
        return {"error": "no_results"}
    
    flood_detected = [r for r in results if r.is_flood_related]
    rescue_needed = [r for r in results if r.needs_rescue]
    
    # Severity distribution
    severity_dist: Dict[str, int] = {}
    for r in results:
        sev = r.flood_severity
        severity_dist[sev] = severity_dist.get(sev, 0) + 1
    
    # Zone distribution
    zone_dist: Dict[str, int] = {}
    for r in flood_detected:
        zone = r.zone_detected if r.zone_detected is not None else "unknown"
        zone_dist[zone] = zone_dist.get(zone, 0) + 1
    
    # Urgency distribution
    urgency_dist: Dict[str, int] = {}
    for r in flood_detected:
        urg = r.urgency_level
        urgency_dist[urg] = urgency_dist.get(urg, 0) + 1
    
    # Flood type distribution
    type_dist: Dict[str, int] = {}
    for r in flood_detected:
        ft = r.flood_type
        type_dist[ft] = type_dist.get(ft, 0) + 1
    
    # High-priority alerts
    high_priority = [
        r for r in results 
        if r.needs_rescue or r.flood_severity in ["severe", "critical"]
    ]
    
    return {
        "summary": {
            "total_tweets_analyzed": total,
            "flood_related_tweets": len(flood_detected),
            "flood_detection_rate": round(len(flood_detected) / total * 100, 1) if total > 0 else 0,
            "rescue_situations": len(rescue_needed),
            "high_priority_alerts": len(high_priority),
            "average_confidence": round(
                sum(r.confidence for r in results) / total, 2
            ) if total > 0 else 0
        },
        "distributions": {
            "severity": severity_dist,
            "zones": zone_dist,
            "urgency": urgency_dist,
            "flood_types": type_dist
        },
        "high_priority_alerts": [r.to_dict() for r in high_priority],
        "analysis_timestamp": datetime.now().isoformat()
    }


def evaluate_accuracy(
    results: List[FloodDetectionResult], 
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate detection accuracy against ground truth
    
    Returns:
        Accuracy metrics including precision, recall, F1
    """
    
    if len(results) != len(ground_truth):
        print(f"⚠️ Length mismatch: {len(results)} results vs {len(ground_truth)} ground truth")
        return {"error": "length_mismatch"}
    
    # Binary classification metrics (flood vs non-flood)
    tp = fp = tn = fn = 0
    
    for result, truth in zip(results, ground_truth):
        pred_flood = result.is_flood_related
        actual_flood = truth["is_flood"]
        
        if pred_flood and actual_flood:
            tp += 1
        elif pred_flood and not actual_flood:
            fp += 1
        elif not pred_flood and actual_flood:
            fn += 1
        else:
            tn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
    
    # Rescue detection accuracy
    rescue_tp = rescue_fp = rescue_fn = 0
    for result, truth in zip(results, ground_truth):
        if result.needs_rescue and truth["needs_rescue"]:
            rescue_tp += 1
        elif result.needs_rescue and not truth["needs_rescue"]:
            rescue_fp += 1
        elif not result.needs_rescue and truth["needs_rescue"]:
            rescue_fn += 1
    
    rescue_precision = rescue_tp / (rescue_tp + rescue_fp) if (rescue_tp + rescue_fp) > 0 else 0
    rescue_recall = rescue_tp / (rescue_tp + rescue_fn) if (rescue_tp + rescue_fn) > 0 else 0
    
    return {
        "flood_detection": {
            "accuracy": round(accuracy * 100, 1),
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1 * 100, 1),
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        },
        "rescue_detection": {
            "precision": round(rescue_precision * 100, 1),
            "recall": round(rescue_recall * 100, 1),
            "true_positives": rescue_tp,
            "false_positives": rescue_fp,
            "false_negatives": rescue_fn
        }
    }