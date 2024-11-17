import requests
import json

url = 'http://heart-attack-prediction-env.eba-cxneqs5v.us-west-2.elasticbeanstalk.com/predict'
url_local = 'http://localhost:9696/predict'

patient = {
    "age": 51,
    "sex": 1,
    "cp": 2,
    "trtbps": 110,
    "chol": 175,
    "fbs": 0,
    "restecg": 1,
    "thalachh": 123,
    "exng": 0,
    "oldpeak": 0.6,
    "slp": 2,
    "caa": 0,
    "thall": 2
}

try:
    # Try sending the request to the cloud URL
    response = requests.post(url, json=patient)
    response.raise_for_status()  # Raise an exception for HTTP errors
except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
    print(f"Cloud URL failed.")
    print("Falling back to local URL...")
    try:
        # Try sending the request to the local URL
        response = requests.post(url_local, json=patient)
        response.raise_for_status()
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
        print(f"Local URL failed.")
        response = None

if response:
    # Parse and print the result if successful
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print("Both cloud and local API requests failed.")