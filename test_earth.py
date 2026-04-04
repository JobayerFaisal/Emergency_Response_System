import ee
from google.oauth2 import service_account

# Path to your downloaded service account JSON key
key_file_path = 'D:/Emergency_Response_System/geecap-42a29310865f.json'  # Replace with your file path

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