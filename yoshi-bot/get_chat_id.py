import subprocess
import json
import time

token = '8567967677:AAE9LFfLwPfVXqxU2n19TY-aWRbZflGCpEs'
vps_ip = "165.245.140.115"
ssh_opts = "-o BatchMode=yes -o ConnectTimeout=10"

def run_remote_curl(endpoint):
    url = f"https://api.telegram.org/bot{token}/{endpoint}"
    cmd = f"ssh {ssh_opts} root@{vps_ip} \"curl -s -S {url}\""
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.stdout.strip().startswith('{'):
            return json.loads(res.stdout)
    except Exception as e:
        print(f"Error checking {endpoint}: {e}")
    return None

print("1. Verifying Token...")
me = run_remote_curl("getMe")
if me and me.get('ok'):
    bot_user = me['result']['username']
    print(f"[OK] Token Verified! Bot: @{bot_user}")
    
    print("\n2. Checking for updates to find Chat ID...")
    tries = 3
    found_chat_id = None
    
    for i in range(tries):
        updates = run_remote_curl("getUpdates")
        if updates and updates.get('ok'):
            for res in updates['result']:
                chat = res.get('message', {}).get('chat', {})
                if chat.get('id'):
                    found_chat_id = chat['id']
                    print(f"\n[FOUND] CHAT ID: {found_chat_id}")
                    print(f"User: {chat.get('username')}")
                    break
        
        if found_chat_id:
            break
            
        print(f"   No messages found yet... (Attempt {i+1}/{tries})")
        if i < tries - 1:
            print(f"   PLEASE SEND A MESSAGE to @{bot_user} on Telegram NOW!")
            time.sleep(5)
            
    if not found_chat_id:
        print(f"\n[ERROR] Could not find Chat ID. Please message @{bot_user} and run this script again.")
else:
    print("[ERROR] Token Invalid or Connection Failed.")
    if me: print(me)
