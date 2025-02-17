import json
import re

def format_json(obj: dict) -> str:
    return json.dumps(obj, indent=2)

def parse_iso8601_duration(duration_str: str) -> float:
    match = re.match(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', duration_str)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration format: {duration_str}")
    days, hours, minutes, seconds = match.groups()
    total_seconds = 0
    if days:
        total_seconds += int(days) * 86400
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += float(seconds)
    return total_seconds
