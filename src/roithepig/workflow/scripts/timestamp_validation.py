"""Validate decoded video presentation timestamps."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math


@dataclass(frozen=True)
class TimestampValidation:
    frames: int
    max_error_seconds: float
    source_duration_seconds: float
    output_duration_seconds: float


def read_presentation_times(path):
    import av

    timestamps = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing presentation timestamp in {path}")
            timestamps.append(float(frame.pts * frame.time_base))
    if not timestamps:
        raise ValueError(f"No video frames in {path}")
    return timestamps


def validate_timestamps(source_times: Sequence[float], output_times: Sequence[float],
                        *, start_frame: int, end_frame: int,
                        tolerance_seconds: float = 0.001) -> TimestampValidation:
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if start_frame < 0 or end_frame < start_frame or end_frame >= len(source_times):
        raise ValueError("Invalid source range")
    expected = source_times[start_frame:end_frame + 1]
    if len(output_times) != len(expected):
        raise ValueError(f"Frame count mismatch: expected {len(expected)}, got {len(output_times)}")
    for label, times in (("source", expected), ("output", output_times)):
        if any(not math.isfinite(t) for t in times):
            raise ValueError(f"Nonfinite {label} timestamp")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"Non-increasing {label} timestamps")
    source_relative = [t - expected[0] for t in expected]
    output_relative = [t - output_times[0] for t in output_times]
    error = max(abs(a - b) for a, b in zip(source_relative, output_relative))
    if error > tolerance_seconds:
        raise ValueError(
            f"Timestamp mismatch: max error {error:.6f}s exceeds "
            f"tolerance {tolerance_seconds:.6f}s"
        )
    return TimestampValidation(len(expected), error, source_relative[-1], output_relative[-1])


def repair_timestamp_collisions(times, *, tolerance_seconds=0.001,
                                tick_seconds=0.000001):
    """Repair small timestamp collisions without inventing frame-rate intervals.

    Returns (repaired, corrections). Every correction is limited by the
    tolerance relative to the original timestamp, including accumulated drift.
    These are encoding-time repairs, not recovered camera acquisition times.
    """
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("tick_seconds must be positive and finite")
    if any(not math.isfinite(t) for t in times):
        raise ValueError("Nonfinite source timestamp")
    repaired, corrections = [], []
    for idx, original in enumerate(times):
        adjusted = original
        if repaired and original <= repaired[-1]:
            adjusted = repaired[-1] + tick_seconds
            if adjusted - original > tolerance_seconds + 1e-12:
                raise ValueError(
                    f"Timestamp anomaly at frame {idx}: source={original:.9f}s, "
                    f"required={adjusted:.9f}s, exceeds "
                    f"{tolerance_seconds * 1000:.3f}ms repair limit"
                )
            corrections.append((idx, original, adjusted))
        repaired.append(adjusted)
    return repaired, corrections
