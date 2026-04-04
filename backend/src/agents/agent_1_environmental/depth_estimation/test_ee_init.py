"""
Simple Earth Engine Test - Find the right initialization method
"""

import ee
import os
import json

print("="*70)
print("  EARTH ENGINE INITIALIZATION TEST")
print("="*70)
print()

# Find credentials file
home = os.path.expanduser("~")
creds_path = os.path.join(home, ".config", "earthengine", "credentials")
if not os.path.exists(creds_path):
    creds_path = os.path.join(home, ".earthengine", "credentials")

print(f"Credentials file: {creds_path}")
print(f"Exists: {os.path.exists(creds_path)}")
print()

if os.path.exists(creds_path):
    try:
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        print("Credentials content:")
        for key in creds.keys():
            if 'token' not in key.lower() and 'secret' not in key.lower():
                print(f"  {key}: {creds[key]}")
        print()
    except Exception as e:
        print(f"Could not read credentials: {e}")
        print()

# Try different initialization methods
print("Testing initialization methods...")
print()

# Method 1: Standard
print("[1] Standard initialization (no project)...")
try:
    ee.Initialize()
    print("  ✓ SUCCESS! Use: ee.Initialize()")
    print()
    print("="*70)
    print("  EARTH ENGINE IS WORKING!")
    print("="*70)
    exit(0)
except Exception as e:
    print(f"  ✗ Failed: {e}")
print()

# Method 2: With project from credentials
print("[2] With project from credentials...")
try:
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        
        project_id = creds.get('project_id') or creds.get('project') or creds.get('quota_project_id')
        
        if project_id:
            print(f"  Found project in credentials: {project_id}")
            ee.Initialize(project=project_id)
            print(f"  ✓ SUCCESS! Use: ee.Initialize(project='{project_id}')")
            print()
            print("="*70)
            print("  EARTH ENGINE IS WORKING!")
            print("="*70)
            print()
            print(f"Add this to your code:")
            print(f"  ee.Initialize(project='{project_id}')")
            exit(0)
        else:
            print("  No project found in credentials")
except Exception as e:
    print(f"  ✗ Failed: {e}")
print()

# Method 3: Try common project ID from folder name
print("[3] Trying project: caramel-pulsar-475819...")
try:
    ee.Initialize(project='caramel-pulsar-475819')
    print("  ✓ SUCCESS! Use: ee.Initialize(project='caramel-pulsar-475819')")
    print()
    print("="*70)
    print("  EARTH ENGINE IS WORKING!")
    print("="*70)
    print()
    print("Add this to your code:")
    print("  ee.Initialize(project='caramel-pulsar-475819')")
    exit(0)
except Exception as e:
    print(f"  ✗ Failed: {e}")
print()

# Method 4: High-volume endpoint
print("[4] High-volume endpoint...")
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("  ✓ SUCCESS! Use: ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')")
    print()
    print("="*70)
    print("  EARTH ENGINE IS WORKING!")
    print("="*70)
    exit(0)
except Exception as e:
    print(f"  ✗ Failed: {e}")
print()

# All failed
print("="*70)
print("  ALL METHODS FAILED")
print("="*70)
print()
print("Solutions to try:")
print()
print("1. Create/Enable a Google Cloud Project:")
print("   - Go to: https://console.cloud.google.com/")
print("   - Create a new project (or use existing)")
print("   - Enable 'Earth Engine API'")
print("   - Copy the project ID")
print("   - Use: ee.Initialize(project='YOUR-PROJECT-ID')")
print()
print("2. Use service account:")
print("   - Create a service account in Google Cloud")
print("   - Download JSON key")
print("   - Use: ee.Initialize(ee.ServiceAccountCredentials(email, key_file))")
print()
print("3. Check credentials:")
print("   earthengine authenticate --force")
print()