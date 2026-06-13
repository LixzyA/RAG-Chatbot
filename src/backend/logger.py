import os
import logging
import logging.config
import yaml
import json

class JSONFormatter(logging.Formatter):
    """Custom formatter to output log messages in JSON format."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

_logging_configured = False

def configure_logging(name: str) -> logging.Logger:
    """
    Initialize logging system from logger.yaml once and return a logger instance.
    
    Args:
        name: Usually __name__ of the calling module.
    """
    global _logging_configured
    if not _logging_configured:
        config_path = os.path.join(os.path.dirname(__file__), "logger.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
                # Resolve filename relative to the backend directory
                if "handlers" in config and "file" in config["handlers"]:
                    file_handler = config["handlers"]["file"]
                    if "filename" in file_handler:
                        backend_dir = os.path.dirname(os.path.abspath(__file__))
                        project_dir = os.path.abspath(os.path.join(backend_dir, "..", ".."))
                        abs_log_path = os.path.abspath(os.path.join(project_dir, "logs", file_handler["filename"]))
                        # Ensure containing directory exists
                        log_dir = os.path.dirname(abs_log_path)
                        if log_dir:
                            os.makedirs(log_dir, exist_ok=True)
                        file_handler["filename"] = abs_log_path
                        
                logging.config.dictConfig(config)
        else:
            # Fallback configuration if logger.yaml is missing
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d in %(funcName)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        _logging_configured = True
    return logging.getLogger(name)