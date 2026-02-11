import subprocess
import json
import os

raw_tokens = os.getenv("TELEGRAM_BOT_TOKENS", "")
tokens = [tok.strip() for tok in raw_tokens.split(",") if tok.strip()]
if not tokens:
    single = os.getenv("TELEGRAM_BOT_TOKEN")
    if single:
        tokens = [single]

vps_ip = os.getenv("VPS_IP")
if not vps_ip:
    raise SystemExit("VPS_IP is not set.")
if not tokens:
    raise SystemExit("Set TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKENS.")

for token in tokens:
    print(f"\nChecking Token: {token}")
    curl_cmd = f"curl -s -S https://api.telegram.org/bot{token}/getMe"
    remote_cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=10 root@{vps_ip} \"{curl_cmd}\""

    try:
        result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        if output.startswith('{'):
            try:
                 data = json.loads(output)
                 if data.get('ok'):
                     bot_user = data.get('result', {}).get('username')
                     print(f"✅ SUCCESS! Valid Token for bot @{bot_user}")
                 else:
                     print(f"❌ API Error: {data.get('description')}")
            except json.JSONDecodeError:
                 print("Failed to decode JSON response")
        else:
            print(f"Unexpected Output: {output}")

    except Exception as e:
        print(f"Error: {e}")
