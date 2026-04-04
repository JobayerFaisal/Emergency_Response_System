"""
Sample Twitter Dataset Generator
================================
Generates realistic flood-related tweets simulating Twitter API v2 response format.
Includes Bengali, English, and mixed-language tweets typical of Dhaka residents.

Author: Mahmudul Hasan
Project: Autonomous Multi-Agent System for Urban Flood Response
Component: Agent 1 - Environmental Intelligence (Social Media)
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os

# =============================================================================
# TWEET TEMPLATES - REALISTIC FLOOD SCENARIOS
# =============================================================================

# Flood-related tweets (positive samples)
FLOOD_TWEETS: Dict[str, List[str]] = {
    "severe": [
        # English - Severe
        "URGENT: Water level rising rapidly in Mirpur-10! Already knee-deep. Need help evacuating elderly neighbors. #DhakaFlood",
        "Major flooding in Uttara Sector 7. Cars are floating! Road completely impassable. Stay safe everyone! 🚨",
        "Badda area is completely underwater. My house ground floor is flooded. Need rescue for my family NOW!",
        "EMERGENCY at Mohammadpur! Water entering homes. Children trapped on rooftops. Please send help immediately!",
        "Dangerous situation in Pallabi. Drainage system failed completely. Water rose 4 feet in 30 minutes!",
        
        # Bengali - Severe
        "জরুরি! মিরপুরে পানি দ্রুত বাড়ছে! হাঁটু পানি। বয়স্ক প্রতিবেশীদের সরাতে সাহায্য দরকার। #ঢাকাবন্যা",
        "উত্তরা সেক্টর ৭ এ ভয়াবহ বন্যা। গাড়ি ভাসছে! রাস্তা সম্পূর্ণ বন্ধ।",
        "বাড্ডা এলাকা সম্পূর্ণ পানির নিচে। আমার বাড়ির নিচতলা ডুবে গেছে। এখনই উদ্ধার দরকার!",
        
        # Mixed (Banglish) - Severe
        "Mohammadpur e HEAVY flooding! Pani ghor e dhukche! Amra rooftop e trapped. Keu help korun please! 🆘",
        "Mirpur 12 te disaster! Pani 5 feet high. Rickshaw, car shob floating. Sob route blocked!",
    ],
    
    "moderate": [
        # English - Moderate
        "Significant waterlogging in Dhanmondi road 27. Traffic completely stopped. Water above ankles.",
        "Gulshan area experiencing flooding after 2 hours of rain. Underground parking filling with water.",
        "Shyamoli main road flooded again. Same problem every monsoon. When will WASA fix drainage?",
        "Water accumulating fast in Kazipara. Schools closed early. Students sent home.",
        "Adabor area flood situation worsening. Local market underwater. Shop owners moving goods.",
        
        # Bengali - Moderate  
        "ধানমন্ডি ২৭ নম্বর রোডে জলাবদ্ধতা। যানবাহন চলাচল বন্ধ। পানি গোড়ালি পর্যন্ত।",
        "গুলশানে ২ ঘণ্টা বৃষ্টির পর বন্যা। আন্ডারগ্রাউন্ড পার্কিং পানিতে ভরছে।",
        "কাজীপাড়ায় দ্রুত পানি জমছে। স্কুল তাড়াতাড়ি ছুটি দিয়েছে।",
        
        # Mixed - Moderate
        "Uttara Diabari te flooding start hoise. Water level badhche slowly. Stay alert everyone!",
        "Baridhara r puro rasta pani te bhore gache. Amra office theke baire berote parchi na.",
    ],
    
    "minor": [
        # English - Minor
        "Some waterlogging near Mirpur stadium after morning rain. Should clear in an hour.",
        "Minor flooding at Mohammadpur bus stand. Buses still running but slow.",
        "Light waterlogging in Uttara sector 3. Drainage working but slow.",
        "Small puddles forming in Badda main road. Nothing serious yet but rain continuing.",
        
        # Bengali - Minor
        "মিরপুর স্টেডিয়ামের কাছে সামান্য জলাবদ্ধতা। এক ঘণ্টায় শুকিয়ে যাবে।",
        "মোহাম্মদপুর বাস স্ট্যান্ডে সামান্য পানি। বাস চলছে তবে ধীরে।",
        
        # Mixed - Minor
        "Dhanmondi lake er pashe ektu pani jomche. Not too bad though.",
    ]
}

# Non-flood related tweets (negative samples - for model accuracy testing)
NON_FLOOD_TWEETS: List[str] = [
    # Weather but not flood
    "Beautiful rain in Dhaka today! Perfect weather for chai and pakora. ☔",
    "Monsoon vibes in Gulshan. Love this weather! #DhakaRain",
    "ঢাকায় আজ সুন্দর বৃষ্টি! চা আর পাকোড়ার আবহাওয়া। 🌧️",
    
    # Normal city life
    "Stuck in traffic at Mirpur road. As usual! 🚗",
    "Had amazing biryani at Mohammadpur today. Must try!",
    "Uttara to Motijheel takes forever. Dhaka traffic is the worst.",
    "Office party at Dhanmondi. Great evening with colleagues!",
    
    # Water-related but not flood
    "WASA water supply interrupted in Badda. No water since morning.",
    "Swimming pool in Gulshan club is so refreshing!",
    "Water filter kinte hobe. Tap water is not safe to drink.",
    
    # Random tweets
    "Bangladesh won the cricket match! 🏏 #BCB",
    "New restaurant opened in Banani. Food is amazing!",
    "Load shedding again in Mirpur. When will it end?",
    "Eid shopping at New Market. So crowded!",
    "Traffic jam at Airport road. Flight might be missed!",
    "University exam tomorrow. So stressed! 📚",
    "Dhaka weather is so unpredictable these days.",
    "Metro rail construction causing road blocks in Uttara.",
]

# Rescue-needed specific tweets
RESCUE_TWEETS: List[str] = [
    "HELP! Family trapped in Mirpur-2. Water entering 2nd floor. 4 people including baby. Call 01712XXXXXX",
    "SOS! Elderly couple stuck in Mohammadpur. Cannot evacuate. Ground floor flooded. Need boat!",
    "উদ্ধার দরকার! বাড্ডায় ৬ জন আটকা পড়েছি। পানি বাড়ছে। ফোন: 01819XXXXXX",
    "EMERGENCY: Disabled person trapped in Uttara Sector 11. Wheelchair bound. Needs immediate evacuation!",
    "Help needed! Hospital patients being moved to upper floors in Shyamoli. Running out of medicine!",
]


def generate_tweet_id() -> str:
    """Generate realistic Twitter snowflake ID"""
    # Twitter IDs are 64-bit integers, roughly time-based
    timestamp = int(datetime.now().timestamp() * 1000)
    sequence = random.randint(0, 4095)
    return str((timestamp << 22) | sequence)


def generate_author_id() -> str:
    """Generate realistic Twitter author ID"""
    return str(random.randint(10**8, 10**18))


def generate_timestamp(hours_ago_max: int = 6) -> str:
    """Generate ISO format timestamp within recent hours"""
    minutes_ago = random.randint(1, hours_ago_max * 60)
    timestamp = datetime.now() - timedelta(minutes=minutes_ago)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def get_zone_from_tweet(text: str, zones: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[float]]]:
    """Extract zone information from tweet text"""
    text_lower = text.lower()
    for zone_id, zone_info in zones.items():
        keywords_en: List[str] = zone_info.get("keywords_en", [])
        keywords_bn: List[str] = zone_info.get("keywords_bn", [])
        all_keywords = keywords_en + keywords_bn
        for keyword in all_keywords:
            if keyword.lower() in text_lower:
                coords: List[float] = zone_info.get("coordinates", [0.0, 0.0])
                return zone_id, coords
    return None, None


def generate_single_tweet(
    text: str,
    zones: Dict[str, Any],
    include_geo: bool = True
) -> Dict[str, Any]:
    """
    Generate a single tweet in Twitter API v2 format
    
    Returns:
        Dict matching Twitter API v2 response schema
    """
    zone_id, coordinates = get_zone_from_tweet(text, zones)
    
    tweet: Dict[str, Any] = {
        "id": generate_tweet_id(),
        "text": text,
        "created_at": generate_timestamp(),
        "author_id": generate_author_id(),
        "lang": "bn" if any('\u0980' <= c <= '\u09FF' for c in text) else "en",
        "public_metrics": {
            "retweet_count": random.randint(0, 500) if "URGENT" in text or "EMERGENCY" in text else random.randint(0, 50),
            "reply_count": random.randint(0, 100),
            "like_count": random.randint(0, 200),
            "quote_count": random.randint(0, 20)
        },
        "possibly_sensitive": False,
        "source": random.choice(["Twitter for Android", "Twitter for iPhone", "Twitter Web App"])
    }
    
    # Add geo data if available and requested
    if include_geo and zone_id is not None and coordinates is not None:
        # Add some randomness to coordinates
        coord_offset = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
        tweet["geo"] = {
            "place_id": f"place_{zone_id}_{uuid.uuid4().hex[:8]}",
            "coordinates": {
                "type": "Point",
                "coordinates": [
                    coordinates[1] + coord_offset[1],  # longitude
                    coordinates[0] + coord_offset[0]   # latitude
                ]
            }
        }
    
    return tweet


def generate_dataset(
    n_flood_severe: int = 15,
    n_flood_moderate: int = 25,
    n_flood_minor: int = 20,
    n_rescue: int = 10,
    n_non_flood: int = 30,
    zones: Optional[Dict[str, Any]] = None,
    shuffle: bool = True
) -> Dict[str, Any]:
    """
    Generate complete dataset simulating Twitter API v2 search response
    
    Args:
        n_flood_severe: Number of severe flood tweets
        n_flood_moderate: Number of moderate flood tweets  
        n_flood_minor: Number of minor flood tweets
        n_rescue: Number of rescue-needed tweets
        n_non_flood: Number of non-flood related tweets
        zones: Zone configuration dictionary
        shuffle: Whether to shuffle the final dataset
        
    Returns:
        Dict in Twitter API v2 response format with metadata
    """
    from config import DHAKA_ZONES
    actual_zones: Dict[str, Any] = zones if zones is not None else DHAKA_ZONES
    
    tweets: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []  # Ground truth for evaluation
    
    # Generate flood tweets by severity
    for _ in range(n_flood_severe):
        text = random.choice(FLOOD_TWEETS["severe"])
        tweets.append(generate_single_tweet(text, actual_zones))
        labels.append({"severity": "severe", "is_flood": True, "needs_rescue": "rescue" in text.lower() or "help" in text.lower()})
    
    for _ in range(n_flood_moderate):
        text = random.choice(FLOOD_TWEETS["moderate"])
        tweets.append(generate_single_tweet(text, actual_zones))
        labels.append({"severity": "moderate", "is_flood": True, "needs_rescue": False})
    
    for _ in range(n_flood_minor):
        text = random.choice(FLOOD_TWEETS["minor"])
        tweets.append(generate_single_tweet(text, actual_zones))
        labels.append({"severity": "minor", "is_flood": True, "needs_rescue": False})
    
    # Generate rescue tweets
    for _ in range(n_rescue):
        text = random.choice(RESCUE_TWEETS)
        tweets.append(generate_single_tweet(text, actual_zones))
        labels.append({"severity": "critical", "is_flood": True, "needs_rescue": True})
    
    # Generate non-flood tweets
    for _ in range(n_non_flood):
        text = random.choice(NON_FLOOD_TWEETS)
        tweets.append(generate_single_tweet(text, actual_zones, include_geo=random.random() > 0.5))
        labels.append({"severity": "none", "is_flood": False, "needs_rescue": False})
    
    # Combine tweets with their labels
    combined = list(zip(tweets, labels))
    
    if shuffle:
        random.shuffle(combined)
    
    if combined:
        tweets_list, labels_list = zip(*combined)
        final_tweets: List[Dict[str, Any]] = list(tweets_list)
        final_labels: List[Dict[str, Any]] = list(labels_list)
    else:
        final_tweets = []
        final_labels = []
    
    # Create Twitter API v2 style response
    dataset: Dict[str, Any] = {
        "data": final_tweets,
        "meta": {
            "newest_id": final_tweets[0]["id"] if final_tweets else None,
            "oldest_id": final_tweets[-1]["id"] if final_tweets else None,
            "result_count": len(final_tweets),
            "next_token": f"next_{uuid.uuid4().hex[:16]}"
        },
        "_ground_truth": final_labels,  # For evaluation (wouldn't exist in real API)
        "_generation_config": {
            "n_flood_severe": n_flood_severe,
            "n_flood_moderate": n_flood_moderate,
            "n_flood_minor": n_flood_minor,
            "n_rescue": n_rescue,
            "n_non_flood": n_non_flood,
            "generated_at": datetime.now().isoformat(),
            "total_tweets": len(final_tweets)
        }
    }
    
    return dataset


def save_dataset(dataset: Dict[str, Any], output_path: str = "data/sample_tweets.json") -> str:
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Total tweets: {len(dataset['data'])}")
    print(f"   Flood-related: {sum(1 for l in dataset['_ground_truth'] if l['is_flood'])}")
    print(f"   Rescue-needed: {sum(1 for l in dataset['_ground_truth'] if l['needs_rescue'])}")
    
    return output_path


if __name__ == "__main__":
    # Generate default dataset
    print("=" * 60)
    print("SOCIAL MEDIA FLOOD DETECTION - DATASET GENERATOR")
    print("=" * 60)
    print()
    
    dataset = generate_dataset(
        n_flood_severe=15,
        n_flood_moderate=25,
        n_flood_minor=20,
        n_rescue=10,
        n_non_flood=30
    )
    
    output_path = save_dataset(dataset)
    
    print()
    print("Dataset Statistics:")
    print("-" * 40)
    
    # Calculate statistics
    ground_truth = dataset["_ground_truth"]
    flood_count = sum(1 for l in ground_truth if l["is_flood"])
    rescue_count = sum(1 for l in ground_truth if l["needs_rescue"])
    
    severity_counts: Dict[str, int] = {}
    for label in ground_truth:
        sev = label["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print(f"Total Tweets: {len(dataset['data'])}")
    print(f"Flood-related: {flood_count} ({100*flood_count/len(dataset['data']):.1f}%)")
    print(f"Non-flood: {len(dataset['data']) - flood_count}")
    print(f"Rescue needed: {rescue_count}")
    print()
    print("Severity Distribution:")
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev}: {count}")