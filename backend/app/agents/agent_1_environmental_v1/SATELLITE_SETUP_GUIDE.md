# Satellite Imagery Module - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Google Earth Engine Setup](#google-earth-engine-setup)
3. [Installation](#installation)
4. [Authentication](#authentication)
5. [Project Structure Integration](#project-structure-integration)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

### Required Accounts
- ✅ Google Earth Engine account (you mentioned you have this)
- ✅ Google Cloud account (for service account - optional for development)

### Software Requirements
```bash
Python >= 3.9
pip >= 21.0
```

---

## 🌍 Google Earth Engine Setup

### Step 1: Verify Your GEE Account

1. Go to: https://code.earthengine.google.com/
2. Sign in with your Google account
3. You should see the Code Editor

### Step 2: Register Your Project (if not done)

1. Go to: https://code.earthengine.google.com/register
2. Register for a non-commercial or commercial project
3. Wait for approval (usually instant for non-commercial)

### Step 3: Test GEE Access

Open the Code Editor and run this test script:
```javascript
// Test script
var point = ee.Geometry.Point([90.4125, 23.8103]);  // Dhaka
var image = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(point)
  .first();

print('Sentinel-1 Image:', image);
Map.centerObject(point, 10);
```

If this works, you're ready to proceed!

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
# Navigate to your project root
cd /path/to/disaster-response-system

# Activate your virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install required packages
pip install earthengine-api
pip install geemap
pip install google-auth
pip install google-auth-oauthlib
pip install google-auth-httplib2
```

### Step 2: Update requirements.txt

Add these lines to your `requirements.txt`:
```
earthengine-api>=0.1.300
geemap>=0.20.0
google-auth>=2.16.0
google-auth-oauthlib>=0.8.0
google-auth-httplib2>=0.1.0
```

Then install:
```bash
pip install -r requirements.txt
```

---

## 🔐 Authentication

### Method 1: Interactive Authentication (Recommended for Development)

This is the easiest method for your capstone project:

```bash
# Run this command
python -c "import ee; ee.Authenticate()"
```

This will:
1. Open your browser
2. Ask you to sign in to Google
3. Request permissions for Earth Engine
4. Save credentials to your computer

**Location of saved credentials:**
- Linux/Mac: `~/.config/earthengine/credentials`
- Windows: `C:\Users\YourName\.config\earthengine\credentials`

### Method 2: Service Account (For Production)

For production deployment, you'll want a service account:

#### 2a. Create Service Account

1. Go to: https://console.cloud.google.com/
2. Create/select a project
3. Go to **IAM & Admin** > **Service Accounts**
4. Click **Create Service Account**
5. Name it: `disaster-response-gee`
6. Grant role: **Earth Engine Resource Writer**
7. Click **Create Key** > **JSON**
8. Save the JSON file securely

#### 2b. Register Service Account with GEE

```bash
# Install gcloud CLI first if not installed
# Then run:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Register service account
earthengine authenticate --service-account-file=/path/to/key.json
```

#### 2c. Use in Code

```python
from satellite_imagery_service import SatelliteImageryService

# Initialize with service account
service = SatelliteImageryService(
    service_account_key_path='/path/to/service-account-key.json'
)
```

---

## 📁 Project Structure Integration

### Current Structure (Based on your README):
```
disaster-response-system/
├── config/
├── src/
│   └── agents/
│       └── agent_1_environmental/
│           ├── __init__.py
│           ├── main.py
│           ├── weather_monitor.py
│           ├── social_media_monitor.py
│           └── models.py
├── requirements.txt
└── README.md
```

### Updated Structure (After Integration):
```
disaster-response-system/
├── config/
│   └── gee_credentials.json  # Service account key (don't commit!)
├── src/
│   └── agents/
│       └── agent_1_environmental/
│           ├── __init__.py
│           ├── main.py
│           ├── weather_monitor.py
│           ├── social_media_monitor.py
│           ├── satellite_monitor.py        # ← NEW
│           ├── models.py
│           └── services/
│               ├── __init__.py
│               └── satellite_service.py    # ← NEW
├── data/
│   └── satellite/
│       ├── geojson/                        # ← NEW
│       └── cache/                          # ← NEW
├── tests/
│   └── test_satellite_monitor.py           # ← NEW
├── requirements.txt
└── README.md
```

### File Placement

**1. Core Service Module:**
```bash
mkdir -p src/agents/agent_1_environmental/services
mv satellite_imagery_service.py src/agents/agent_1_environmental/services/
```

**2. Monitor Integration:**
```bash
mv satellite_imagery_monitor.py src/agents/agent_1_environmental/satellite_monitor.py
```

**3. Update Imports:**

In `src/agents/agent_1_environmental/satellite_monitor.py`:
```python
from .services.satellite_service import SatelliteImageryService, FloodDetectionResult
from .models import ThreatLevel, Location
from .database import DatabaseManager
from .redis_client import RedisClient
```

---

## 🧪 Testing

### Test 1: Basic GEE Connection

Create `tests/test_gee_connection.py`:

```python
"""Test Google Earth Engine connection"""
import ee

def test_gee_connection():
    """Test that GEE is properly initialized"""
    try:
        ee.Initialize()
        
        # Try to access a simple dataset
        point = ee.Geometry.Point([90.4125, 23.8103])
        image_count = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(point) \
            .size() \
            .getInfo()
        
        print(f"✅ GEE Connection Successful!")
        print(f"✅ Found {image_count} Sentinel-1 images for Dhaka")
        return True
        
    except Exception as e:
        print(f"❌ GEE Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_gee_connection()
```

Run:
```bash
python tests/test_gee_connection.py
```

### Test 2: Flood Detection (Dry Run)

Create `tests/test_flood_detection.py`:

```python
"""Test flood detection with recent data"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.agents.agent_1_environmental.services.satellite_service import SatelliteImageryService
from datetime import datetime, timedelta

def test_flood_detection():
    """Test flood detection for Dhaka"""
    
    print("Initializing Satellite Imagery Service...")
    service = SatelliteImageryService()
    
    print("Testing flood detection for Dhaka...")
    print("This may take 1-2 minutes...\n")
    
    # Dhaka coordinates
    dhaka = (23.8103, 90.4125)
    
    # Use recent dates
    now = datetime.now()
    after_end = now.strftime('%Y-%m-%d')
    after_start = (now - timedelta(days=3)).strftime('%Y-%m-%d')
    before_end = (now - timedelta(days=20)).strftime('%Y-%m-%d')
    before_start = (now - timedelta(days=50)).strftime('%Y-%m-%d')
    
    try:
        result = service.detect_flood(
            location=dhaka,
            radius_km=30,
            before_start=before_start,
            before_end=before_end,
            after_start=after_start,
            after_end=after_end
        )
        
        print("="*60)
        print("FLOOD DETECTION TEST RESULTS")
        print("="*60)
        print(f"Location: Dhaka, Bangladesh")
        print(f"Analysis Period: {after_start} to {after_end}")
        print(f"Baseline Period: {before_start} to {before_end}")
        print(f"\nFlood Area Detected: {result.flood_area_km2:.2f} km²")
        print(f"Number of Flooded Pixels: {result.flood_pixels:,}")
        print(f"Detection Confidence: {result.detection_confidence:.2%}")
        print(f"\nAffected Regions: {len(result.affected_regions)}")
        for region in result.affected_regions:
            print(f"  - {region['name']}")
        
        print(f"\nMap URLs Generated: {len(result.image_urls)}")
        print("="*60)
        
        # Save GeoJSON
        import json
        with open('flood_test_result.geojson', 'w') as f:
            json.dump(result.geojson, f, indent=2)
        print("\n✅ Test completed! GeoJSON saved to flood_test_result.geojson")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flood_detection()
    sys.exit(0 if success else 1)
```

Run:
```bash
python tests/test_flood_detection.py
```

### Test 3: Monitor Integration

Create `tests/test_monitor_integration.py`:

```python
"""Test full monitor integration"""
import asyncio
from src.agents.agent_1_environmental.satellite_monitor import SatelliteImageryMonitor

async def test_monitor():
    """Test the satellite monitor"""
    
    print("Initializing Satellite Monitor...")
    monitor = SatelliteImageryMonitor()
    
    print("Running flood check...")
    result = await monitor.check_for_floods()
    
    if result:
        print(f"\n✅ Monitor Test Successful!")
        print(f"Threat Level: {result.threat_level}")
        print(f"Flood Detected: {result.flood_detected}")
        print(f"Area: {result.flood_area_km2:.2f} km²")
    else:
        print("\n❌ Monitor test failed - no results")
    
    return result is not None

if __name__ == "__main__":
    success = asyncio.run(test_monitor())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
```

Run:
```bash
python tests/test_monitor_integration.py
```

---

## 🔧 Troubleshooting

### Issue 1: "Please authenticate before initializing"

**Solution:**
```bash
python -c "import ee; ee.Authenticate()"
```

### Issue 2: "No images found for the specified dates"

**Cause:** Sentinel-1 doesn't cover your location during those dates

**Solution:**
- Check Sentinel-1 coverage: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1
- Adjust date ranges to wider windows
- Check if your location is in coverage area

### Issue 3: "Module 'ee' has no attribute 'Initialize'"

**Solution:**
```bash
pip uninstall earthengine-api
pip install earthengine-api --upgrade
```

### Issue 4: Rate Limit Errors

**Cause:** Too many requests to GEE

**Solution:**
- Add delays between requests
- Use service account for higher quotas
- Implement caching

### Issue 5: Memory Errors

**Cause:** Processing too large an area

**Solution:**
```python
# Reduce radius
result = service.detect_flood(
    location=dhaka,
    radius_km=20,  # Instead of 50
    ...
)
```

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GEE
python -c "import ee; ee.Authenticate()"

# 3. Test connection
python tests/test_gee_connection.py

# 4. Run flood detection test
python tests/test_flood_detection.py

# 5. Test monitor integration
python tests/test_monitor_integration.py

# 6. Start monitoring (in your main.py)
python -m src.agents.agent_1_environmental.main
```

---

## 📊 Expected Output

When everything is working, you should see:

```
✅ GEE Connection Successful!
✅ Found 156 Sentinel-1 images for Dhaka
Checking satellite imagery for floods...
Found 12 images from 2024-10-01 to 2024-10-15
Found 8 images from 2024-11-20 to 2024-11-30

============================================================
FLOOD DETECTION TEST RESULTS
============================================================
Location: Dhaka, Bangladesh
Flood Area Detected: 15.42 km²
Detection Confidence: 82.5%
Threat Level: MODERATE
============================================================
```

---

## 🎓 Next Steps

1. ✅ Complete authentication
2. ✅ Run all tests
3. ✅ Integrate with your main.py
4. 📊 Add database schema for satellite data
5. 🗺️ Create visualization dashboard
6. 📱 Add to your presentation demo

---

## 📚 Additional Resources

- [GEE Python API Docs](https://developers.google.com/earth-engine/guides/python_install)
- [Sentinel-1 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar)
- [Flood Mapping Tutorial](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping)

---

## 💬 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify your GEE authentication
3. Check Sentinel-1 data availability for your dates
4. Ensure your internet connection is stable

Good luck with your capstone project! 🎓
