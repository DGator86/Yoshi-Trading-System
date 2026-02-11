import subprocess
import json
import os

token = os.getenv("TELEGRAM_BOT_TOKEN")
vps_ip = os.getenv("VPS_IP")

if not token:
    raise SystemExit("TELEGRAM_BOT_TOKEN is not set.")
if not vps_ip:
    raise SystemExit("VPS_IP is not set.")

# Using curl avoids python One-Liner quoting hell in SSH
# -s for silent (no progress bar), -S to show errors
curl_cmd = f"curl -s -S https://api.telegram.org/bot{token}/getUpdates"

remote_cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=10 root@{vps_ip} \"{curl_cmd}\""

print(f"Executing: {remote_cmd}")
try:
    result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Check if we got valid JSON
    output = result.stdout.strip()
    if output.startswith('{'):
        try:
             data = json.loads(output)
             if data.get('ok'):
                 print("\nSUCCESS: Valid Token.")
                 updates = data.get('result', [])
                 if not updates:
                     print("No updates found. Send a message to the bot to generate a Chat ID.")
                 else:
                     for res in updates:
                         chat = res.get('message', {}).get('chat', {})
                         print(f"Chat ID found: {chat.get('id')} (Type: {chat.get('type')}, User: {chat.get('username')})")
             else:
                 print("\nAPI Error:", data)
        except json.JSONDecodeError:
             print("\nFailed to decode JSON response")
    else:
        print("Output does not look like JSON.")

except Exception as e:
    print(f"Error: {e}")
