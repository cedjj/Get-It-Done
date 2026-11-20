"""Fixed-loopback HTTP transport for benchmark inference only."""
import requests

BASE = 'http://127.0.0.1:11434'


def api(path, body=None, timeout=300):
    response = (requests.get(BASE + path, timeout=timeout) if body is None else
                requests.post(BASE + path, json=body, timeout=timeout))
    response.raise_for_status()
    return response.json()
