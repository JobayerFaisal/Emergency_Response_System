"""
Configuration for Social Media Flood Detection Module
======================================================
Autonomous Multi-Agent System for Real-Time Urban Flood Response
Agent 1 - Environmental Intelligence: Social Media Component
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# API CONFIGURATION
# =============================================================================
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
OPENAI_MODEL: str = "gpt-3.5-turbo"  # Cost-effective, accurate for classification

# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
DETECTION_CONFIG = {
    "batch_size": 10,           # Tweets processed per API call (reduces costs)
    "confidence_threshold": 0.7, # Minimum confidence for flood detection
    "max_retries": 3,            # API retry attempts
    "timeout": 30                # API timeout in seconds
}

# =============================================================================
# MONITORED ZONES - DHAKA DIVISIONS
# =============================================================================
DHAKA_ZONES = {
    "mirpur": {
        "name": "Mirpur",
        "coordinates": [23.8223, 90.3654],
        "flood_prone": True,
        "keywords_bn": ["মিরপুর", "পল্লবী", "কাজীপাড়া"],
        "keywords_en": ["mirpur", "pallabi", "kazipara"]
    },
    "uttara": {
        "name": "Uttara",
        "coordinates": [23.8759, 90.3795],
        "flood_prone": True,
        "keywords_bn": ["উত্তরা", "আজমপুর", "দিয়াবাড়ি"],
        "keywords_en": ["uttara", "azampur", "diabari"]
    },
    "mohammadpur": {
        "name": "Mohammadpur",
        "coordinates": [23.7662, 90.3589],
        "flood_prone": True,
        "keywords_bn": ["মোহাম্মদপুর", "শ্যামলী", "আদাবর"],
        "keywords_en": ["mohammadpur", "shyamoli", "adabor"]
    },
    "dhanmondi": {
        "name": "Dhanmondi",
        "coordinates": [23.7461, 90.3742],
        "flood_prone": False,
        "keywords_bn": ["ধানমন্ডি", "কলাবাগান"],
        "keywords_en": ["dhanmondi", "kalabagan"]
    },
    "badda": {
        "name": "Badda",
        "coordinates": [23.7806, 90.4261],
        "flood_prone": True,
        "keywords_bn": ["বাড্ডা", "গুলশান", "বারিধারা"],
        "keywords_en": ["badda", "gulshan", "baridhara"]
    }
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_CONFIG = {
    "output_dir": "output",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "save_raw_responses": True,    # Save OpenAI API responses
    "generate_summary": True,       # Generate summary report
    "export_formats": ["json", "csv"]
}

# =============================================================================
# TWITTER DATA SCHEMA (Simulated)
# =============================================================================
TWITTER_SCHEMA = {
    "id": "string",
    "text": "string",
    "created_at": "datetime",
    "author_id": "string",
    "geo": {
        "place_id": "string",
        "coordinates": "array"
    },
    "public_metrics": {
        "retweet_count": "int",
        "reply_count": "int",
        "like_count": "int"
    },
    "lang": "string"
}

# =============================================================================
# FLOOD DETECTION OUTPUT SCHEMA
# =============================================================================
DETECTION_SCHEMA = {
    "tweet_id": "string",
    "original_text": "string",
    "is_flood_related": "boolean",
    "confidence": "float (0-1)",
    "flood_severity": "enum: none|minor|moderate|severe|critical",
    "location_mentioned": "string or null",
    "zone_detected": "string or null",
    "flood_type": "enum: waterlogging|river_overflow|flash_flood|drainage_failure|unknown",
    "urgency_level": "enum: low|medium|high|critical",
    "needs_rescue": "boolean",
    "extracted_info": {
        "water_level_mentioned": "string or null",
        "affected_infrastructure": "array",
        "time_reference": "string or null"
    },
    "processed_at": "datetime"
}