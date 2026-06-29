import urllib.request
import json

try:
    req = urllib.request.Request('http://127.0.0.1:8001/docs')
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"Status: {resp.status}")
        data = resp.read()
        print(f"Content length: {len(data)}")
except Exception as e:
    print(f"Error accessing /docs: {e}")

try:
    req = urllib.request.Request('http://127.0.0.1:8001/health')
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        print(f"Health: {data.decode()}")
except Exception as e:
    print(f"Error accessing /health: {e}")