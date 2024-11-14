import logging
import json
from pathlib import Path

class JsonFileHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(json.loads(log_entry))
        with open(self.filename, 'w') as f:
            json.dump(self.logs, f, indent=4)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'time': self.formatTime(record),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
        }
        return json.dumps(log_record)