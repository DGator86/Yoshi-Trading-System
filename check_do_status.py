import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gnosis.utils.digitalocean_client import DigitalOceanClient

load_dotenv()

def check_droplet():
    client = DigitalOceanClient()
    vps_ip = os.getenv("VPS_IP")
    print(f"Checking status for VPS IP: {vps_ip}")
    
    droplets = client.list_droplets()
    if not droplets:
        print("No droplets found in this account.")
        return

    found = False
    for d in droplets:
        # Check all public IPv4 addresses
        ips = [net['ip_address'] for net in d.get('networks', {}).get('v4', []) if net['type'] == 'public']
        print(f"Found Droplet: {d['name']} (ID: {d['id']}, Status: {d['status']}, IPs: {ips})")
        if vps_ip in ips:
            found = True
            print(f"✅ Match found! Droplet {d['name']} is {d['status']}.")
            if d['status'] != 'active':
                print("⚠️ Droplet is not active. It might be powered off.")
    
    if not found:
        print(f"❌ Could not find a droplet with IP {vps_ip} in your account.")

if __name__ == "__main__":
    check_droplet()
