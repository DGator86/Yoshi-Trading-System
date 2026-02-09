import requests
import os

token = "8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs"
url = f"https://api.telegram.org/bot{token}/getMe"

print(f"Testing token: {token}")
try:
    response = requests.get(url, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
