import logging
import os

def create_logger():
    log = logging.getLogger()
    log_level = logging.DEBUG if os.environ.get('DEBUG', 0) == '1' else logging.INFO
    log.setLevel(log_level)
    format_str = '%(asctime)s - %(levelname)-2s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)

logger = create_logger()