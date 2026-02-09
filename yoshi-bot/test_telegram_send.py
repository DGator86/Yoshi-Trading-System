import subprocess
import json

token = '8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs'
chat_id = '8236163940'
vps_ip = "165.245.140.115"
ssh_opts = "-o BatchMode=yes -o ConnectTimeout=10"

def run_remote_curl(endpoint, data=None):
    url = f"https://api.telegram.org/bot{token}/{endpoint}"
    if data:
        json_data = json.dumps(data).replace('"', '\\"')
        cmd = f"ssh {ssh_opts} root@{vps_ip} \"curl -s -S -H 'Content-Type: application/json' -d '{json_data}' {url}\""
    else:
        cmd = f"ssh {ssh_opts} root@{vps_ip} \"curl -s -S {url}\""
        
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.stdout.strip().startswith('{'):
            return json.loads(res.stdout)
        else:
            print("Raw Output:", res.stdout)
            print("Stderr:", res.stderr)
    except Exception as e:
        print(f"Error checking {endpoint}: {e}")
    return None

print(f"Testing Telegram with Token: ...{token[-5:]} and Chat ID: {chat_id}")

msg = "🔔 Telegram Configuration Test: SUCCESS!"
res = run_remote_curl("sendMessage", {"chat_id": chat_id, "text": msg})

if res and res.get('ok'):
    print("\n[SUCCESS] Message sent successfully to Chat ID!")
    print(f"Message ID: {res['result']['message_id']}")
else:
    print("\n[ERROR] Failed to send message.")
    if res:
        print(f"API Error: {res.get('description')}")
        if res.get('error_code') == 400 or res.get('error_code') == 401 or res.get('error_code') == 403:
             print("   -> The Chat ID might be invalid or the bot was blocked.")
             print("   -> Please message @Crypto_Gnosis_Bot to get a new Chat ID.")
    else:
        print("No response from API.")
