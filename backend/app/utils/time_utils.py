"""
Time Utilities.

Helper functions for timestamp handling and formatting.
"""

from datetime import datetime, timezone
from typing import overload


def format_timestamp(dt: datetime | None = None) -> str:
    """
    Format a datetime as ISO 8601 string.

    Args:
        dt: Datetime to format. Uses current UTC time if None.

    Returns:
        ISO 8601 formatted string.
    """
    if dt is None:
        dt = now_utc()

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string.

    Handles various ISO 8601 formats:
    - 2024-01-15T10:30:00Z
    - 2024-01-15T10:30:00+00:00
    - 2024-01-15T10:30:00.123456Z

    Args:
        timestamp_str: ISO 8601 formatted string.

    Returns:
        Parsed datetime object (timezone-aware).

    Raises:
        ValueError: If parsing fails.
    """
    # Handle 'Z' suffix (Zulu time)
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(timestamp_str)

        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def now_utc() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone.
    """
    return datetime.now(timezone.utc)


def timestamp_to_epoch(dt: datetime) -> float:
    """
    Convert datetime to Unix epoch timestamp.

    Args:
        dt: Datetime to convert.

    Returns:
        Unix epoch timestamp (seconds since 1970-01-01).
    """
    return dt.timestamp()


def epoch_to_timestamp(epoch: float) -> datetime:
    """
    Convert Unix epoch to datetime.

    Args:
        epoch: Unix epoch timestamp (seconds since 1970-01-01).

    Returns:
        Datetime object in UTC.
    """
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time difference from now.

    Args:
        dt: Datetime to compare.

    Returns:
        Human-readable string like "5 minutes ago".
    """
    now = now_utc()

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 0:
        return "in the future"

    if seconds < 60:
        return "just now"

    minutes = seconds / 60
    if minutes < 60:
        m = int(minutes)
        return f"{m} minute{'s' if m != 1 else ''} ago"

    hours = minutes / 60
    if hours < 24:
        h = int(hours)
        return f"{h} hour{'s' if h != 1 else ''} ago"

    days = hours / 24
    if days < 30:
        d = int(days)
        return f"{d} day{'s' if d != 1 else ''} ago"

    months = days / 30
    if months < 12:
        m = int(months)
        return f"{m} month{'s' if m != 1 else ''} ago"

    years = days / 365
    y = int(years)
    return f"{y} year{'s' if y != 1 else ''} ago"


def is_within_range(
    dt: datetime,
    start: datetime | None = None,
    end: datetime | None = None,
) -> bool:
    """
    Check if datetime is within a range.

    Args:
        dt: Datetime to check.
        start: Range start (inclusive). None means no lower bound.
        end: Range end (inclusive). None means no upper bound.

    Returns:
        True if datetime is within range.
    """
    if start is not None and dt < start:
        return False
    if end is not None and dt > end:
        return False
    return True


def format_date(dt: datetime) -> str:
    """
    Format datetime as a simple date string.

    Args:
        dt: Datetime to format.

    Returns:
        Date string like "Jan 15, 2024".
    """
    return dt.strftime("%b %d, %Y")


def format_datetime(dt: datetime) -> str:
    """
    Format datetime as a human-readable string.

    Args:
        dt: Datetime to format.

    Returns:
        Datetime string like "Jan 15, 2024 10:30 AM".
    """
    return dt.strftime("%b %d, %Y %I:%M %p")
