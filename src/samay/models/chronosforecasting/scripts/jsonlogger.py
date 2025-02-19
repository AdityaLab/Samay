import logging
import json

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
    def __init__(self, log_type):
        super().__init__()
        self.log_type = log_type

    def format(self, record):
        if self.log_type == "evaluation":
            log_record = {
                'column_id': record.msg.get('column_id'),
                'num_samples': record.msg.get('num_samples'),
                'predictions': record.msg.get('predictions'),
                'eval_results': record.msg.get('eval_results')
            }
        else:
            log_record = {
                'time': self.formatTime(record, self.datefmt),
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage()
            }
        return json.dumps(log_record)