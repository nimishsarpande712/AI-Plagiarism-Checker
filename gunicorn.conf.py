# Gunicorn config tuned for low-memory hosts (Render free-tier)
# Explicitly disable preload to avoid double-loading models into parent + worker
preload_app = False

# Keep workers/threads low to reduce memory pressure
workers = 1
threads = 1

# Give generous timeout for heavy requests (model loading/inference)
timeout = 300

# Logging
loglevel = 'debug'
capture_output = True
accesslog = '-'
errorlog = '-'

# Allow forwarded headers from Render
forwarded_allow_ips = '*'

# Keepalive
keepalive = 2
