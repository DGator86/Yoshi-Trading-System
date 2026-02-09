import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.getcwd(), "clawdbot"))

from gnosis.telegram.bot import TelegramAPI

token = "8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs"
api = TelegramAPI(token)

print(f"Testing via TelegramAPI class...")
me = api.get_me()
if me:
    print(f"Success! Bot name: {me.get('username')}")
else:
    print("Failed via TelegramAPI class.")
