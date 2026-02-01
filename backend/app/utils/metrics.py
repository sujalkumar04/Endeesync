"""
Performance Metrics.

Utilities for measuring and logging execution times.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


@dataclass
class TimingMetrics:
    """
    Container for timing measurements.

    Tracks individual operation times and provides aggregation.
    """

    timings: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, duration_ms: float) -> None:
        """Add a timing measurement."""
        self.timings[name] = duration_ms

    def get(self, name: str) -> float | None:
        """Get a timing measurement."""
        return self.timings.get(name)

    @property
    def total_ms(self) -> float:
        """Get total time across all measurements."""
        return sum(self.timings.values())

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        result = dict(self.timings)
        result["total_ms"] = self.total_ms
        return result

    def log(self, operation: str) -> None:
        """Log all timings."""
        parts = [f"{k}={v:.1f}ms" for k, v in self.timings.items()]
        parts.append(f"total={self.total_ms:.1f}ms")
        logger.info(f"{operation} timings: {', '.join(parts)}")


class Timer:
    """
    High-precision timer for performance measurement.

    Example:
        timer = Timer()
        # ... do work ...
        elapsed = timer.elapsed_ms()
    """

    __slots__ = ("_start", "_end")

    def __init__(self):
        self._start = time.perf_counter()
        self._end: float | None = None

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        self._end = time.perf_counter()
        return self.elapsed_ms()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        end = self._end if self._end else time.perf_counter()
        return (end - self._start) * 1000

    def reset(self) -> None:
        """Reset the timer."""
        self._start = time.perf_counter()
        self._end = None


@contextmanager
def timed(name: str, metrics: TimingMetrics | None = None) -> Generator[Timer, None, None]:
    """
    Context manager for timing a block of code.

    Args:
        name: Name for this timing.
        metrics: Optional TimingMetrics to record to.

    Yields:
        Timer instance.

    Example:
        metrics = TimingMetrics()
        with timed("embedding", metrics):
            embeddings = embedder.embed(text)
    """
    timer = Timer()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        if metrics:
            metrics.add(name, elapsed)
        logger.debug(f"{name}: {elapsed:.1f}ms")


def timed_async(name: str):
    """
    Decorator for timing async functions.

    Args:
        name: Name for this timing.

    Returns:
        Decorated function that logs timing.

    Example:
        @timed_async("search")
        async def search(query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            timer = Timer()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = timer.stop()
                logger.info(f"{name}: {elapsed:.1f}ms")
        return wrapper
    return decorator


def timed_sync(name: str):
    """
    Decorator for timing sync functions.

    Args:
        name: Name for this timing.

    Returns:
        Decorated function that logs timing.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = Timer()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = timer.stop()
                logger.info(f"{name}: {elapsed:.1f}ms")
        return wrapper
    return decorator


class RequestMetrics:
    """
    Per-request metrics collector.

    Collects timing and other metrics for a single API request.

    Example:
        metrics = RequestMetrics()
        with metrics.time("embedding"):
            ...
        with metrics.time("retrieval"):
            ...
        return {"timings": metrics.get_timings()}
    """

    __slots__ = ("_timings", "_start", "_metadata")

    def __init__(self):
        self._timings = TimingMetrics()
        self._start = Timer()
        self._metadata: dict[str, Any] = {}

    @contextmanager
    def time(self, name: str) -> Generator[Timer, None, None]:
        """Time a named operation."""
        with timed(name, self._timings) as timer:
            yield timer

    def add_timing(self, name: str, duration_ms: float) -> None:
        """Add a timing measurement directly."""
        self._timings.add(name, duration_ms)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the metrics."""
        self._metadata[key] = value

    def get_timings(self) -> dict[str, float]:
        """Get all timing measurements."""
        result = self._timings.to_dict()
        result["total_request_ms"] = self._start.elapsed_ms()
        return result

    def get_metadata(self) -> dict[str, Any]:
        """Get all metadata."""
        return dict(self._metadata)

    def log_summary(self, operation: str) -> None:
        """Log a summary of all metrics."""
        self._timings.log(operation)

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics as a dictionary."""
        return {
            "timings": self.get_timings(),
            "metadata": self.get_metadata(),
        }
