
import logging
import logging.handlers

"""
    Returns a logging handler with a default format
"""
def get_logging_handler(log_filename):

    handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=2*1024*1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)

    return handler

