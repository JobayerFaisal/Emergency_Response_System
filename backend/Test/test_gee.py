import ee

print("Step 1: Authenticating...")
ee.Authenticate()

print("Step 2: Initializing with project...")
ee.Initialize(project='caramel-pulsar-475810-e7')

print("✅ GEE Connected Successfully!")

print("\nStep 3: Testing image fetch...")

try:
    # Get Sentinel-1 collection
    collection = ee.ImageCollection('COPERNICUS/S1_GRD')
    
    # Check collection size
    size = collection.size().getInfo()
    print(f"Collection size: {size} images")
    
    # Get first image
    image = collection.first()
    
    # Get info separately
    info = image.getInfo()
    
    if info is None:
        print("❌ image.getInfo() returned None")
        print("This might be a permissions issue.")
    else:
        print(f"Image ID: {info.get('id', 'No ID found')}")
        print("\n🎉 Everything is working!")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"Message: {e}")