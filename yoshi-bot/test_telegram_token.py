import os

import requests

token = os.getenv("TELEGRAM_BOT_TOKEN")
if not token:
    print("TELEGRAM_BOT_TOKEN is not set. Export it first.")
    raise SystemExit(1)

url = f"https://api.telegram.org/bot{token}/getMe"

try:
    response = requests.get(url, timeout=20)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as exc:
    print(f"Error: {exc}")
    raise SystemExit(1)
