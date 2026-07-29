"""
Handles communication with FastAPI backend.
"""

import time
import requests

from config import API_URL, REQUEST_TIMEOUT


class APIClient:

    def __init__(self):
        self.base_url = API_URL

    def health(self):

        start = time.time()

        try:

            response = requests.get(
                f"{self.base_url}/health",
                timeout=REQUEST_TIMEOUT
            )

            latency = round((time.time() - start) * 1000)

            response.raise_for_status()

            data = response.json()

            return {
                "connected": True,
                "latency": latency,
                "status": data["status"],
                "model_version": data["model_version"]
            }

        except Exception:

            return {
                "connected": False,
                "latency": None,
                "status": "offline",
                "model_version": "-"
            }

    def predict(self, payload):

        start = time.time()

        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        latency = round((time.time() - start) * 1000)

        try:
            response.raise_for_status()
        except requests.RequestException as e:
             raise RuntimeError(
                 f"Backend request failed: {e}" )

        result = response.json()

        result["latency"] = latency

        return result