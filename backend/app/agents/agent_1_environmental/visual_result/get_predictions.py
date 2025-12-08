import requests
import pandas as pd
import json
from datetime import datetime

# Fetch predictions from API
response = requests.get('http://localhost:8001/output')
data = response.json()

# Extract predictions into list of dictionaries
predictions_list = []
for pred in data['predictions']:
    predictions_list.append({
        'Zone': pred['zone']['name'],
        'Risk Score': f"{pred['risk_score']:.2%}",
        'Severity': pred['severity_level'],
        'Confidence': f"{pred['confidence']:.2%}",
        'Time to Impact (hrs)': pred['time_to_impact_hours'],
        'Affected Area (km²)': f"{pred['affected_area_km2']:.2f}",
        'Est. Population': pred['estimated_affected_population'],
        'Alert Level': pred['alert_level'],
        'Timestamp': pred['timestamp']
    })

# Create DataFrame
df = pd.DataFrame(predictions_list)

# Display
print("\n" + "="*100)
print(f"FLOOD RISK PREDICTIONS - Agent: {data['agent_id']}")
print(f"Updated: {data['timestamp']}")
print("="*100)
print(df.to_string(index=False))
print("="*100)
print(f"\nTotal Predictions: {data['total_predictions']}")
print(f"Critical Alerts: {data['critical_alerts']}")
print(f"Processing Time: {data['processing_time_seconds']:.2f}s")