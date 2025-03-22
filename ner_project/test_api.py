import requests
import json

# Test prediction
url = "http://localhost:8000/predict"
test_texts = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "Barack Obama was the 44th president of the United States",
    "Paris is the capital of France"
]

for text in test_texts:
    try:
        payload = {"text": text}
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Print full response for debugging
        print(f"Input: {text}")
        print("Full Response:", json.dumps(response.json(), indent=2))
        print("---")
        health_response = requests.get("http://localhost:8000/health")
        print(health_response.json())
    except requests.RequestException as e:
        print(f"Error processing text '{text}': {e}")
        # Print error details
        if hasattr(e, 'response'):
            print("Error Response:", e.response.text)
        