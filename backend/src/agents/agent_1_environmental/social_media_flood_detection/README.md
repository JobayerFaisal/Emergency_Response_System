# Social Media Flood Detection Module

## Agent 1 - Environmental Intelligence | Social Media Component

**Autonomous Multi-Agent System for Real-Time Urban Flood Response Coordination in Bangladesh**

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Technical Implementation](#technical-implementation)
5. [Dataset Format](#dataset-format)
6. [Detection Schema](#detection-schema)
7. [Installation & Usage](#installation--usage)
8. [API Cost Analysis](#api-cost-analysis)
9. [Output Files](#output-files)
10. [Integration with Main System](#integration-with-main-system)

---

## Overview

This module is responsible for analyzing social media data (simulated Twitter/X data) to detect flood events in Dhaka, Bangladesh. It uses OpenAI's GPT-3.5-turbo model for natural language understanding and classification.

### Key Features

- **Bilingual Support**: Processes Bengali, English, and mixed "Banglish" text
- **Batch Processing**: Optimized API calls for cost efficiency
- **Structured Output**: Comprehensive flood detection results with severity, location, and urgency
- **Accuracy Evaluation**: Built-in ground truth comparison
- **Report-Ready Output**: JSON, CSV, and formatted console reports

### Position in Multi-Agent System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT 1: ENVIRONMENTAL INTELLIGENCE               │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │    Weather      │  │   Social Media   │  │    Satellite     │   │
│  │    Monitor      │  │   ◄── THIS ──►   │  │    Imagery       │   │
│  │  (OpenWeather)  │  │     MODULE       │  │  (Sentinel-1)    │   │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                    │                     │              │
│           └────────────────────┼─────────────────────┘              │
│                                │                                    │
│                     ┌──────────▼──────────┐                        │
│                     │    Data Fusion &    │                        │
│                     │     Risk Score      │                        │
│                     └──────────┬──────────┘                        │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │   Message Bus       │
                      │      (Redis)        │
                      └─────────────────────┘
```

---

## System Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SOCIAL MEDIA FLOOD DETECTION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. DATA ACQUISITION                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Twitter API v2 (Simulated)                                  │   │
│  │  • Search: "flood OR বন্যা OR waterlog" + Dhaka geo-fence   │   │
│  │  • Format: JSON with tweet text, geo, metrics               │   │
│  │  • Sample: 100 tweets (70 flood + 30 non-flood)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  2. BATCH PROCESSING                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Batch Size: 10 tweets per API call                         │   │
│  │  • Reduces API costs by 90%                                 │   │
│  │  • Maintains context for better classification              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  3. OPENAI CLASSIFICATION                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Model: GPT-3.5-turbo                                        │   │
│  │  Temperature: 0.1 (consistent classification)               │   │
│  │  Output: Structured JSON per tweet                          │   │
│  │                                                              │   │
│  │  Classification Tasks:                                       │   │
│  │  ├── Flood detection (binary)                               │   │
│  │  ├── Severity classification (5 levels)                     │   │
│  │  ├── Location extraction                                    │   │
│  │  ├── Zone mapping                                           │   │
│  │  ├── Urgency assessment                                     │   │
│  │  └── Rescue need detection                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  4. POST-PROCESSING                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Result aggregation                                        │   │
│  │  • Zone-wise statistics                                     │   │
│  │  • High-priority alert extraction                           │   │
│  │  • Accuracy evaluation (if ground truth available)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  5. OUTPUT GENERATION                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ├── flood_detection_results.json (complete results)        │   │
│  │  ├── flood_detections.csv (tabular format)                  │   │
│  │  └── high_priority_alerts.json (rescue situations)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Input: Simulated Twitter API v2 Response

The system generates realistic Twitter data that matches the Twitter API v2 response format:

```json
{
  "data": [
    {
      "id": "1734567890123456789",
      "text": "URGENT: Water level rising rapidly in Mirpur-10! Already knee-deep.",
      "created_at": "2024-12-10T14:30:00.000Z",
      "author_id": "987654321",
      "lang": "en",
      "geo": {
        "place_id": "place_mirpur_abc123",
        "coordinates": {
          "type": "Point",
          "coordinates": [90.3654, 23.8223]
        }
      },
      "public_metrics": {
        "retweet_count": 150,
        "reply_count": 45,
        "like_count": 200
      }
    }
  ],
  "meta": {
    "result_count": 100,
    "newest_id": "...",
    "oldest_id": "..."
  }
}
```

### Output: Flood Detection Results

```json
{
  "tweet_id": "1734567890123456789",
  "original_text": "URGENT: Water level rising rapidly in Mirpur-10!...",
  "is_flood_related": true,
  "confidence": 0.95,
  "flood_severity": "severe",
  "location_mentioned": "Mirpur-10",
  "zone_detected": "mirpur",
  "flood_type": "waterlogging",
  "urgency_level": "critical",
  "needs_rescue": false,
  "extracted_info": {
    "water_level_mentioned": "knee-deep",
    "affected_infrastructure": ["roads"],
    "time_reference": null
  },
  "processed_at": "2024-12-10T15:00:00.000000"
}
```

---

## Technical Implementation

### OpenAI Prompt Engineering

The system uses a carefully crafted system prompt that:

1. **Establishes Domain Expertise**: Flood detection specialist for Dhaka
2. **Defines Classification Criteria**: Clear guidelines for severity levels
3. **Handles Multilingual Content**: Bengali, English, and mixed text
4. **Maps Local Geography**: Known flood-prone zones in Dhaka

### Batch Processing Strategy

```
100 tweets ÷ 10 tweets/batch = 10 API calls

Without batching: 100 API calls (~$0.10)
With batching:     10 API calls (~$0.01)
                   
Savings: 90%
```

### Classification Schema

| Severity | Criteria | Example |
|----------|----------|---------|
| **Critical** | Life-threatening, people trapped | "Family trapped on rooftop!" |
| **Severe** | Major flooding, evacuation needed | "Cars floating in Uttara" |
| **Moderate** | Significant waterlogging | "Roads flooded, traffic stopped" |
| **Minor** | Light waterlogging | "Some puddles forming" |
| **None** | Not flood-related | "Beautiful rain today" |

---

## Dataset Format

### Generated Dataset Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Total Tweets** | 100 | 100% |
| Flood - Severe | 15 | 15% |
| Flood - Moderate | 25 | 25% |
| Flood - Minor | 20 | 20% |
| Rescue Needed | 10 | 10% |
| Non-Flood | 30 | 30% |

### Monitored Zones

| Zone | Bengali | Flood-Prone | Keywords |
|------|---------|-------------|----------|
| Mirpur | মিরপুর | ✅ Yes | Pallabi, Kazipara |
| Uttara | উত্তরা | ✅ Yes | Azampur, Diabari |
| Mohammadpur | মোহাম্মদপুর | ✅ Yes | Shyamoli, Adabor |
| Dhanmondi | ধানমন্ডি | ❌ No | Kalabagan |
| Badda | বাড্ডা | ✅ Yes | Gulshan, Baridhara |

---

## Detection Schema

### Complete Output Schema

```python
FloodDetectionResult = {
    "tweet_id": str,              # Original tweet ID
    "original_text": str,         # Tweet text
    "is_flood_related": bool,     # Binary classification
    "confidence": float,          # 0.0 - 1.0
    "flood_severity": enum,       # none|minor|moderate|severe|critical
    "location_mentioned": str,    # Extracted location name
    "zone_detected": str,         # Mapped zone ID
    "flood_type": enum,           # waterlogging|river_overflow|flash_flood|drainage_failure
    "urgency_level": enum,        # low|medium|high|critical
    "needs_rescue": bool,         # Rescue situation detected
    "extracted_info": {
        "water_level_mentioned": str,
        "affected_infrastructure": list,
        "time_reference": str,
        "contact_info": str
    },
    "processed_at": datetime
}
```

---

## Installation & Usage

### Prerequisites

- Python 3.9+
- OpenAI API key with GPT-3.5-turbo access

### Installation

```bash
# Navigate to module directory
cd social_media_flood_detection

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Pipeline

```bash
# Full pipeline (generate data + process)
python main.py

# Generate dataset only
python main.py --generate-only

# Process existing dataset
python main.py --data data/sample_tweets.json

# Specify output directory
python main.py --output results/
```

### Programmatic Usage

```python
from main import run_pipeline

# Run complete pipeline
results = run_pipeline(
    api_key="your-key",
    generate_data=True,
    output_dir="output"
)

# Access results
print(f"Accuracy: {results['evaluation']['flood_detection']['accuracy']}%")
print(f"Cost: ${results['cost']['estimated_cost_usd']}")
```

---

## API Cost Analysis

### GPT-3.5-turbo Pricing (as of 2024)

| Token Type | Price per 1K tokens |
|------------|---------------------|
| Input | $0.0005 |
| Output | $0.0015 |

### Cost Estimation for 100 Tweets

| Metric | Value |
|--------|-------|
| API Calls | ~10 |
| Input Tokens | ~5,000 |
| Output Tokens | ~3,000 |
| **Total Cost** | **~$0.007** |

### Budget Projection ($3 USD)

```
$3.00 ÷ $0.007 per 100 tweets = ~42,857 tweets

For prototype demonstration: MORE than sufficient
```

---

## Output Files

### 1. Complete Results (JSON)

`flood_detection_results_YYYYMMDD_HHMMSS.json`

Contains:
- All detection results
- Analysis summary
- Accuracy evaluation
- Cost breakdown

### 2. Tabular Export (CSV)

`flood_detections_YYYYMMDD_HHMMSS.csv`

| Column | Description |
|--------|-------------|
| tweet_id | Tweet identifier |
| is_flood_related | Boolean classification |
| confidence | Detection confidence |
| flood_severity | Severity level |
| zone_detected | Mapped zone |
| needs_rescue | Rescue flag |

### 3. High Priority Alerts (JSON)

`high_priority_alerts_YYYYMMDD_HHMMSS.json`

```json
{
  "alert_type": "HIGH PRIORITY FLOOD ALERTS",
  "total_alerts": 15,
  "alerts": [
    {
      "priority": "RESCUE_NEEDED",
      "tweet_id": "...",
      "text": "Help! Family trapped...",
      "severity": "critical",
      "location": "Mirpur-2",
      "zone": "mirpur"
    }
  ]
}
```

---

## Integration with Main System

### Redis Message Format

When integrated with the main disaster response system, this module publishes to Redis:

```python
# Channel: flood_detection_events
message = {
    "event_type": "SOCIAL_MEDIA_FLOOD_DETECTION",
    "timestamp": "2024-12-10T15:00:00",
    "source": "agent_1_social_media",
    "data": {
        "zone": "mirpur",
        "severity": "severe",
        "confidence": 0.95,
        "needs_rescue": false,
        "location": "Mirpur-10"
    }
}
```

### Database Schema (PostgreSQL)

```sql
CREATE TABLE social_media_detections (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(64) UNIQUE,
    detected_at TIMESTAMP,
    is_flood_related BOOLEAN,
    confidence DECIMAL(3,2),
    severity VARCHAR(20),
    zone VARCHAR(50),
    location_text TEXT,
    needs_rescue BOOLEAN,
    raw_text TEXT,
    extracted_info JSONB
);
```

---

## File Structure

```
social_media_flood_detection/
├── config.py                 # Configuration settings
├── sample_dataset_generator.py   # Twitter data simulator
├── openai_processor.py       # OpenAI detection logic
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── data/
│   └── sample_tweets.json    # Generated dataset
├── output/
│   ├── flood_detection_results_*.json
│   ├── flood_detections_*.csv
│   └── high_priority_alerts_*.json
└── docs/
    └── (additional documentation)
```

---

## Author

**Mahmudul Hasan**  
Computer Science Undergraduate  
Capstone Project: Autonomous Multi-Agent System for Urban Flood Response

---

## License

MIT License - See main project repository for details.
