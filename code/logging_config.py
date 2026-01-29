"""Logging and observability utilities: structured logging and latency tracking."""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_latency(operation_name: str):
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


class QueryMetrics:
    def __init__(self):
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.validation_failures = 0
        self.total_latency_ms = 0.0
    
    def record_query(self, success: bool, latency_ms: float, validation_passed: bool):
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        if not validation_passed:
            self.validation_failures += 1
    
    def get_stats(self) -> dict:
        avg_latency = self.total_latency_ms / self.total_queries if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "validation_failures": self.validation_failures,
            "avg_latency_ms": round(avg_latency, 2),
        }
