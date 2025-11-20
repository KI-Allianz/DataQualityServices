#Its a gunicorn config file
import multiprocessing

# Bind
bind = "0.0.0.0:8000"

# Worker tuning: (2*CPU)+1 rule
workers = multiprocessing.cpu_count() * 2 + 1

# Thread tuning
threads = 2

# Timeout for heavy ML tasks
timeout = 120
graceful_timeout = 30

# Restart worker after N reqs to avoid mem leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"

# Preload app for faster start & shared memory (optional)
preload_app = True

# Worker class (keep sync for CPU-bound ML)
# worker_class = "sync"   # default
