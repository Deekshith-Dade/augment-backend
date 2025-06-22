"""
Using StructLog
Json logging and Console logging
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import (
    Any, List
)

import structlog
from dotenv import load_dotenv


load_dotenv()

# LOG DETAILS
LOG_DIR_NAME = os.getenv("LOG_DIR", "logs")
LOG_DIR = Path(__file__).parent.parent.parent / LOG_DIR_NAME
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
LOG_FORMAT = os.getenv("LOG_FORMAT", "console")

def get_log_file_path() -> Path:
    return LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"


class JsonlFileHandler(logging.Handler):
    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.pathname,
            }
            if hasattr(record, "extra"):
                log_entry.update(record.extra)
            
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            self.handleError(record)
    
    def close(self)->None:
        super().close()
        
def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_file_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.PATHNAME,
                }
            )
        )
    
    processors.append(lambda _, __, event_dict: {**event_dict, "environment": "ENV"})
    
    return processors

                
def setup_logging() -> None:
    for log_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "numba", "langsmith", "langsmith.tracing", "urllib3", "httpx"]:
        log = logging.getLogger(log_name)
        log.propagate = False
        log.addHandler(logging.NullHandler())
    
    file_handler = JsonlFileHandler(get_log_file_path())
    file_handler.setLevel(LOG_LEVEL)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    
    shared_processors = get_structlog_processors(
        include_file_info=True
    )
    
    logging.basicConfig(
        format="%(message)s",
        level=LOG_LEVEL,
        handlers=[file_handler, console_handler]
    )
    
    if LOG_FORMAT == "console":
        # Developer Logging
        structlog.configure(
            processors = [
                *shared_processors,
                structlog.dev.ConsoleRenderer(), # pretty outputs
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    


setup_logging()

logger = structlog.get_logger()
logger.info(
    "logging_initialized",
    
)