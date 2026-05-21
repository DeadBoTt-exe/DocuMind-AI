"""Logging and observability utilities: structured logging and latency tracking."""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable


def setup_logging(level: int | str = logging.INFO):
    """Configure root logger. Accepts a log level int or string (e.g. 'DEBUG')."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_latency(operation_name: str):
    """Decorator that logs execution time for both sync and async functions."""
    def decorator(func: Callable):
        logger = logging.getLogger(func.__module__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(f"{operation_name} | latency_ms={latency_ms:.2f} | status=success")
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{operation_name} | latency_ms={latency_ms:.2f} | status=error | error={e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(f"{operation_name} | latency_ms={latency_ms:.2f} | status=success")
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{operation_name} | latency_ms={latency_ms:.2f} | status=error | error={e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator