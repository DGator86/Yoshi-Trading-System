import requests

token = '8501633363:AAHaBepg65Uu-pKjZhFD1zOuiHV_zypGVQY'
url = f'https://api.telegram.org/bot{token}/getMe'

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
