import ee

ee.Initialize()
ee.Authenticate()
ee.Initialize(project='caramel-pulsar-475810-e7')

print("🛰️ Fetching Sentinel-1 SAR imagery for Dhaka, Bangladesh...")

# Define Dhaka area (bounding box)
dhaka_bounds = ee.Geometry.Rectangle([90.25, 23.65, 90.50, 23.90])

# Get recent Sentinel-1 images for Dhaka
collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(dhaka_bounds)
    .filterDate('2024-01-01', '2024-12-31')  # Last year
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))  # Interferometric Wide
    .select('VV')  # VV polarization - best for flood detection
)

# Get count
count = collection.size().getInfo()
print(f"✅ Found {count} Sentinel-1 images for Dhaka in 2024")

# Get the most recent image
if count > 0:
    latest = collection.sort('system:time_start', False).first()
    info = latest.getInfo()
    
    # Extract date
    timestamp = info['properties']['system:time_start']
    from datetime import datetime
    date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')
    
    print(f"\n📅 Most Recent Image:")
    print(f"   Date: {date}")
    print(f"   ID: {info['id']}")
    print(f"   Mode: {info['properties'].get('instrumentMode', 'N/A')}")
    print(f"   Orbit: {info['properties'].get('orbitProperties_pass', 'N/A')}")
    
    # Get image statistics for Dhaka
    stats = latest.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=dhaka_bounds,
        scale=100
    ).getInfo()
    
    print(f"   Mean VV backscatter: {stats.get('VV', 'N/A'):.2f} dB")
    
    print("\n🎉 Satellite imagery fetch successful!")
    print("=" * 50)
    print("You can now proceed to flood detection training!")
else:
    print("❌ No images found. Check the date range or area.")