"""
Gunicorn Production Config for AyurParam AI
"""
import multiprocessing

import os

# Server socket
port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"


# Workers — Use 1 worker for ML model (shared GPU memory)
workers = 1
threads = 4
worker_class = "gthread"

# Timeout — model loading can take a while
timeout = 300
graceful_timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "ayurparam-ai"

# Do NOT preload app. Preloading massive ML models blocks Gunicorn's master process and causes Memory Limits/SIGKILL.
preload_app = False
