import logging
from enum import StrEnum    


LOG_FORMAT_DEBUG = "%(levelname)s:%(message)s:%(pathname)s:%(funcName)s:%(lineno)d"


class LogLevels(StrEnum):
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    debug = "DEBUG"


def configure_logging(log_level: str = LogLevels.error):
    log_level = str(log_level).upper()
    log_levels = [level.value for level in LogLevels]

    if log_level not in log_levels:
        log_level = LogLevels.error

    # Get root logger and configure it directly (works even if basicConfig was already called)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    if log_level == LogLevels.debug:
        formatter = logging.Formatter(LOG_FORMAT_DEBUG)
    else:
        formatter = logging.Formatter("%(levelname)s:%(message)s")
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)