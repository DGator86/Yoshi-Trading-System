import subprocess
import json

tokens = [
    '8501633363:AAHaBepg65Uu-pKjZhFD1zOuiHV_zypGVQY', # The one in get_tg_updates.py
    '8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs', # The one in .env
]
vps_ip = "165.245.140.115"

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
