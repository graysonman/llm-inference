import json
import logging
from typing import Any, Dict


def get_logger() -> logging.Logger:
    logger = logging.getLogger("llm_server")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")  # keep logs JSON-only
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_json(logger: logging.Logger, payload: Dict[str, Any]) -> None:
    # Ensure everything is JSON serializable
    safe = {}
    for k, v in payload.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)

    logger.info(json.dumps(safe, ensure_ascii=False))
