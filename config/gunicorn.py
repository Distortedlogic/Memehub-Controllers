bind = "0.0.0.0:8000"
accesslog = "-"
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" in %(D)sµs'
)
log_level = "debug"
reload = True
workers = 1
timeout = 120
