# Social Media Flood Detection - Technical Architecture Document

## For Academic Report & Defense Presentation

---

## 1. System Overview

### 1.1 Purpose

This module detects flood events from social media posts using Natural Language Processing (NLP) powered by OpenAI's GPT-3.5-turbo model. It serves as a critical component of Agent 1 (Environmental Intelligence) in the multi-agent disaster response system.

### 1.2 Design Rationale

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Local NLP Model** | Free after setup, Offline capable | Needs training data, Complex setup | ❌ Not selected |
| **OpenAI API** | High accuracy, No training needed, Handles Bengali | Cost per request | ✅ Selected |
| **Keyword Matching** | Very fast, No cost | Low accuracy, Misses context | ❌ Not selected |

**Selection Justification**: OpenAI API selected for prototype phase due to:
- Zero-shot classification capability (no labeled training data required)
- Excellent multilingual support (Bengali + English)
- Rapid development timeline (1 month prototype)
- Acceptable cost (~$0.01 per 100 tweets)

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SOCIAL MEDIA FLOOD DETECTION MODULE                    │
│                    Agent 1 - Environmental Intelligence                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         DATA LAYER                                  │  │
│  │  ┌──────────────────┐         ┌──────────────────────────────────┐ │  │
│  │  │  Twitter API v2  │ ──────► │  Sample Dataset Generator        │ │  │
│  │  │   (Simulated)    │         │  • 100 tweets                    │ │  │
│  │  │                  │         │  • Bengali + English + Banglish  │ │  │
│  │  │  Real API would  │         │  • With ground truth labels      │ │  │
│  │  │  replace this    │         └──────────────────────────────────┘ │  │
│  │  └──────────────────┘                                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                       PROCESSING LAYER                              │  │
│  │                                                                     │  │
│  │  ┌─────────────┐    ┌─────────────────────────────────────────────┐│  │
│  │  │   Batch     │    │           OpenAI GPT-3.5-turbo              ││  │
│  │  │  Processor  │───►│                                             ││  │
│  │  │             │    │  System Prompt:                             ││  │
│  │  │ 10 tweets   │    │  • Flood detection expert for Dhaka         ││  │
│  │  │ per batch   │    │  • Bilingual (Bengali + English)            ││  │
│  │  │             │    │  • Zone-aware (Mirpur, Uttara, etc.)        ││  │
│  │  │ Reduces     │    │                                             ││  │
│  │  │ API costs   │    │  Classification Output:                     ││  │
│  │  │ by 90%      │    │  • is_flood_related (boolean)               ││  │
│  │  └─────────────┘    │  • flood_severity (5 levels)                ││  │
│  │                     │  • location_mentioned                       ││  │
│  │                     │  • zone_detected                            ││  │
│  │                     │  • urgency_level                            ││  │
│  │                     │  • needs_rescue                             ││  │
│  │                     └─────────────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│                                    ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                        OUTPUT LAYER                                 │  │
│  │                                                                     │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │  │
│  │  │  Complete JSON   │  │    CSV Export    │  │  High Priority   │ │  │
│  │  │     Results      │  │                  │  │     Alerts       │ │  │
│  │  │                  │  │  For analysis    │  │                  │ │  │
│  │  │  • All results   │  │  in Excel/       │  │  • Rescue cases  │ │  │
│  │  │  • Analysis      │  │  pandas          │  │  • Severe floods │ │  │
│  │  │  • Evaluation    │  │                  │  │  • Immediate     │ │  │
│  │  │  • Cost info     │  │                  │  │    action        │ │  │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Tweet     │     │   Batch     │     │   OpenAI    │     │  Structured │
│   Input     │────►│   Queue     │────►│    API      │────►│   Result    │
│             │     │             │     │             │     │             │
│ "Mirpur e   │     │ [10 tweets] │     │ Classify    │     │ {           │
│  pani       │     │             │     │ each tweet  │     │  "flood":   │
│  badhche!"  │     │             │     │             │     │   true,     │
│             │     │             │     │             │     │  "zone":    │
│             │     │             │     │             │     │  "mirpur"   │
│             │     │             │     │             │     │ }           │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 4. Classification Logic

### 4.1 Severity Classification Matrix

| Level | Water Depth | Impact | Action Required |
|-------|-------------|--------|-----------------|
| **Critical** | >3 feet | Life threat | Immediate rescue |
| **Severe** | 2-3 feet | Property damage | Evacuation |
| **Moderate** | 1-2 feet | Traffic blocked | Monitoring |
| **Minor** | <1 foot | Inconvenience | Awareness |
| **None** | N/A | Not flood | None |

### 4.2 Zone Detection

The system maps locations to predefined zones:

```
Tweet: "Pallabi te pani dhukche!"
       ↓
Keyword Match: "Pallabi" → mirpur zone
       ↓
Output: zone_detected = "mirpur"
```

### 4.3 Rescue Detection Triggers

| Keyword Pattern | Language | Example |
|-----------------|----------|---------|
| "help", "trapped", "rescue" | English | "Help! Family trapped!" |
| "উদ্ধার", "আটকা" | Bengali | "উদ্ধার দরকার!" |
| "SOS", "EMERGENCY" | Universal | "SOS! Need help!" |

---

## 5. API Cost Analysis

### 5.1 Token Estimation

```
Per Tweet:
  Input Tokens  ≈ 50 (tweet) + 200 (prompt) = 250 tokens
  Output Tokens ≈ 100 tokens

Per Batch (10 tweets):
  Input Tokens  ≈ 500 (tweets) + 200 (prompt) = 700 tokens
  Output Tokens ≈ 1000 tokens
```

### 5.2 Cost Calculation

```
GPT-3.5-turbo Pricing:
  Input:  $0.0005 / 1K tokens
  Output: $0.0015 / 1K tokens

For 100 tweets (10 batches):
  Input Cost:  (7000 / 1000) × $0.0005 = $0.0035
  Output Cost: (10000 / 1000) × $0.0015 = $0.0150
  Total: ~$0.02

Budget: $3.00
Capacity: ~15,000 tweets
```

---

## 6. Output Schema

### 6.1 Detection Result Object

```json
{
  "tweet_id": "7404451667367691357",
  "original_text": "URGENT: Water rising in Mirpur!",
  "is_flood_related": true,
  "confidence": 0.95,
  "flood_severity": "severe",
  "location_mentioned": "Mirpur",
  "zone_detected": "mirpur",
  "flood_type": "waterlogging",
  "urgency_level": "high",
  "needs_rescue": false,
  "extracted_info": {
    "water_level_mentioned": "rising rapidly",
    "affected_infrastructure": ["roads"],
    "time_reference": null,
    "contact_info": null
  },
  "processed_at": "2024-12-10T15:30:00.000000"
}
```

### 6.2 Summary Statistics Object

```json
{
  "summary": {
    "total_tweets_analyzed": 100,
    "flood_related_tweets": 70,
    "flood_detection_rate": 70.0,
    "rescue_situations": 10,
    "high_priority_alerts": 25
  },
  "distributions": {
    "severity": {
      "critical": 10,
      "severe": 15,
      "moderate": 25,
      "minor": 20,
      "none": 30
    },
    "zones": {
      "mirpur": 20,
      "uttara": 15,
      "mohammadpur": 18,
      "badda": 12,
      "unknown": 5
    }
  }
}
```

---

## 7. Evaluation Metrics

### 7.1 Confusion Matrix

```
                    Predicted
                 Flood    No Flood
              ┌─────────┬─────────┐
Actual  Flood │   TP    │   FN    │
              ├─────────┼─────────┤
     No Flood │   FP    │   TN    │
              └─────────┴─────────┘
```

### 7.2 Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | >85% |
| **Precision** | TP/(TP+FP) | >80% |
| **Recall** | TP/(TP+FN) | >90% |
| **F1 Score** | 2×(P×R)/(P+R) | >85% |

---

## 8. Integration Points

### 8.1 Input Integration (Future: Real Twitter API)

```python
# Current: Simulated
dataset = generate_dataset()

# Future: Real Twitter API
import tweepy
client = tweepy.Client(bearer_token=TWITTER_TOKEN)
tweets = client.search_recent_tweets(
    query="flood OR বন্যা place_country:BD",
    max_results=100
)
```

### 8.2 Output Integration (Redis Pub/Sub)

```python
# Publish to message bus
import redis
r = redis.Redis()

for result in flood_detections:
    if result.needs_rescue:
        r.publish('flood_alerts', json.dumps({
            'type': 'RESCUE_NEEDED',
            'zone': result.zone_detected,
            'location': result.location_mentioned,
            'confidence': result.confidence
        }))
```

---

## 9. Error Handling

| Error Type | Handling Strategy |
|------------|-------------------|
| API Rate Limit | Exponential backoff (5s, 10s, 20s) |
| JSON Parse Error | Return empty result, log error |
| Network Timeout | Retry 3 times with 30s timeout |
| Invalid Response | Skip tweet, continue batch |

---

## 10. Limitations & Future Work

### Current Limitations

1. **Simulated Data**: Uses generated dataset, not real Twitter API
2. **Batch Latency**: ~2-3 seconds per batch (10 tweets)
3. **Zone Coverage**: Only 5 major zones mapped

### Future Enhancements

1. **Real-time Stream Processing**: Twitter Streaming API integration
2. **Image Analysis**: Flood images in tweets using Vision API
3. **Local Model**: Train BERT model on labeled data for offline use
4. **More Zones**: Expand to all 93 wards of Dhaka

---

*Document prepared for Capstone Project Defense*
*Author: Mahmudul Hasan*
