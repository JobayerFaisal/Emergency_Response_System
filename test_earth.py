import os
import ee
from pathlib import Path
from google.oauth2 import service_account

# Auto-load .env file from the project root (two levels up from this script,
# or the current working directory — whichever exists first).
# This means you don't need to manually set environment variables in your shell.
def load_env_file():
    search_paths = [
        Path(__file__).parent / ".env",           # same folder as this script
        Path(__file__).parent.parent / ".env",    # one level up (backend/)
        Path(__file__).parent.parent.parent / ".env",  # two levels up (project root)
        Path.cwd() / ".env",                      # wherever you run the script from
    ]
    for env_path in search_paths:
        if env_path.is_file():
            print(f"Loading .env from: {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            return
    print("Warning: No .env file found. Relying on shell environment variables.")

load_env_file()

# Read key file path from environment
key_file_path = os.environ.get("GEE_KEY_FILE")

if not key_file_path:
    raise EnvironmentError(
        "GEE_KEY_FILE environment variable is not set.\n"
        "Add this line to your .env file:\n"
        "  GEE_KEY_FILE=D:/Emergency_Response_System/geecap-42a29310865f.json"
    )

if not os.path.isfile(key_file_path):
    raise FileNotFoundError(
        f"Service account key file not found at: {key_file_path}\n"
        "Check that the path in GEE_KEY_FILE is correct."
    )

# Define the required scopes for Earth Engine
SCOPES = ['https://www.googleapis.com/auth/earthengine.readonly']

# Create credentials from the service account JSON key
credentials = service_account.Credentials.from_service_account_file(
    key_file_path,
    scopes=SCOPES
)

# Initialize Earth Engine with the service account credentials
ee.Initialize(credentials=credentials)

print("Earth Engine initialized successfully with the service account!")
