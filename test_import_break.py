import requests
import sys
import os

token = "8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs"
url = f"https://api.telegram.org/bot{token}/getMe"

print("1. Testing directly (no imports)...")
try:
    resp = requests.get(url, timeout=10)
    print(f"   Direct test: {resp.status_code}")
except Exception as e:
    print(f"   Direct test FAILED: {e}")

print("2. Importing from gnosis...")
# Ensure project root is on path
sys.path.insert(0, os.path.join(os.getcwd(), "clawdbot"))
from gnosis.telegram.bot import TelegramAPI

print("3. Testing again after import...")
try:
    resp = requests.get(url, timeout=10)
    print(f"   Post-import test: {resp.status_code}")
except Exception as e:
    print(f"   Post-import test FAILED: {e}")
