"""
Configuration for UpgradeIQ Frontend
"""

import os

# Backend API URL (Default: http://localhost:8080)
API_URL = os.environ.get("API_URL", "http://localhost:8080")

REQUEST_TIMEOUT = 30

APP_NAME = "UpgradeIQ"

VERSION = "1.0.0"

ENABLE_HEALTH_CHECK = True