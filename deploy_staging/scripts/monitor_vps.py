#!/usr/bin/env python3
"""VPS Health Monitor & Auto-Recovery for Yoshi-Bot.

This script runs on a timer to:
1. Check if the Kalshi Scanner process is active.
2. Monitor VPS health via DigitalOcean API.
3. Alert via Telegram if the scanner has crashed.
4. Optionally trigger a VPS reboot if the system is unresponsive.
"""
import os
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

from src.gnosis.utils.notifications import send_telegram_alert_sync
from src.gnosis.utils.digitalocean_client import DigitalOceanClient

load_dotenv()

def is_scanner_running():
    """Check if the kalshi_scanner.py process is active."""
    try:
        # Search for the process name in the system process list
        output = subprocess.check_output(["pgrep", "-f", "kalshi_scanner.py"]).decode()
        return len(output.strip()) > 0
    except subprocess.CalledProcessError:
        return False

def monitor():
    vps_ip = os.getenv("VPS_IP")
    do_token = os.getenv("DIGITALOCEAN_TOKEN")
    
    print(f"[{datetime.now()}] Starting Health Monitor for {vps_ip}...")
    
    # 1. Check Scanner Status (Local Process Check)
    if not is_scanner_running():
        msg = f"⚠️ **YOSHI CRITICAL ALERT**\n\nKalshi Scanner is NOT running on {vps_ip}.\nAttempting to restart..."
        print(msg)
        send_telegram_alert_sync(msg)
        
        # Attempt Restart
        try:
            # We run it in the background using nohup
            subprocess.Popen(
                "nohup python3 scripts/kalshi_scanner.py --symbol BTCUSDT --loop --interval 300 --threshold 0.10 > scanner.log 2>&1 &",
                shell=True
            )
            time.sleep(5)
            if is_scanner_running():
                send_telegram_alert_sync("✅ Scanner restarted successfully.")
            else:
                send_telegram_alert_sync("❌ Failed to restart scanner. Manual intervention required.")
        except Exception as e:
            send_telegram_alert_sync(f"❌ Error during restart attempt: {e}")

    # 2. Check VPS Status via DigitalOcean
    try:
        do = DigitalOceanClient(do_token)
        droplet = do.get_droplet_by_ip(vps_ip)
        
        if droplet:
            status = droplet.get("status")
            if status != "active":
                msg = f"🚨 **VPS INFRASTRUCTURE ALERT**\n\nDroplet {droplet.get('name')} is in state: `{status}`\nIP: {vps_ip}"
                send_telegram_alert_sync(msg)
        else:
            print("Could not find droplet via API. Check DIGITALOCEAN_TOKEN and VPS_IP.")
            
    except Exception as e:
        print(f"DigitalOcean API Error: {e}")

if __name__ == "__main__":
    # In production, this can be run via Crontab or a simple loop
    while True:
        monitor()
        time.sleep(1800) # Check every 30 minutes
