import subprocess
import json

token = '8501633363:AAHaBepg65Uu-pKjZhFD1zOuiHV_zypGVQY'
cmd = f"python3 -c \"import requests; res = requests.get('https://api.telegram.org/bot{token}/getUpdates'); print(res.text)\""
remote_cmd = f"ssh root@165.245.140.115 {cmd}"

result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
