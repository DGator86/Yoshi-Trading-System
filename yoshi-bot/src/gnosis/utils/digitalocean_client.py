import os
import requests
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # dotenv is optional; environment variables may be provided by systemd/CI.
    pass

class DigitalOceanClient:
    """Simple client for DigitalOcean API (v2)."""
    
    BASE_URL = "https://api.digitalocean.com/v2"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("DIGITALOCEAN_TOKEN")
        if not self.token:
            raise ValueError("DIGITALOCEAN_TOKEN missing in .env")
            
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def list_ssh_keys(self) -> List[Dict[str, Any]]:
        """List all SSH keys in the account."""
        response = requests.get(f"{self.BASE_URL}/account/keys", headers=self.headers)
        if response.status_code == 200:
            return response.json().get("ssh_keys", [])
        return []

    def add_ssh_key(self, name: str, public_key: str) -> Optional[Dict[str, Any]]:
        """Add a new SSH key to the DO account."""
        payload = {"name": name, "public_key": public_key}
        response = requests.post(f"{self.BASE_URL}/account/keys", headers=self.headers, json=payload)
        if response.status_code == 201:
            return response.json().get("ssh_key")
        return None

    def list_droplets(self) -> List[Dict[str, Any]]:
        """List all droplets in the account."""
        response = requests.get(f"{self.BASE_URL}/droplets", headers=self.headers)
        if response.status_code == 200:
            return response.json().get("droplets", [])
        return []

    def get_droplet_by_ip(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Find a specific droplet by its IPv4 address."""
        droplets = self.list_droplets()
        for d in droplets:
            for net in d.get("networks", {}).get("v4", []):
                if net.get("ip_address") == ip_address:
                    return d
        return None

    def reboot_droplet(self, droplet_id: int) -> bool:
        """Trigger a reboot action on a droplet."""
        payload = {"type": "reboot"}
        response = requests.post(
            f"{self.BASE_URL}/droplets/{droplet_id}/actions",
            headers=self.headers,
            json=payload
        )
        return response.status_code == 201
