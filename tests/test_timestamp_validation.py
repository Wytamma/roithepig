import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "roithepig" / "workflow" / "scripts"))
from timestamp_validation import validate_timestamps, repair_timestamp_collisions


def test_variable_frame_intervals_preserved():
    result = validate_timestamps(
        [1.0, 1.052, 1.114, 1.165, 1.227],
        [0.0, 0.052, 0.114, 0.165, 0.227],
        start_frame=0, end_frame=4,
    )
    assert result.max_error_seconds < 1e-9


def test_constant_fps_reencoding_rejected():
    with pytest.raises(ValueError, match="Timestamp mismatch"):
        validate_timestamps(
            [1.0, 1.052, 1.114, 1.165, 1.227],
            [0, 0.05, 0.10, 0.15, 0.20],
            start_frame=0, end_frame=4,
        )


def test_dropped_frame_rejected():
    with pytest.raises(ValueError, match="Frame count mismatch"):
        validate_timestamps([0, 0.05, 0.11], [0, 0.05], start_frame=0, end_frame=2)


def test_nonmonotonic_timestamp_rejected():
    with pytest.raises(ValueError, match="Non-increasing"):
        validate_timestamps([0, 0.05, 0.11], [0, 0.05, 0.05],
                            start_frame=0, end_frame=2)


def test_segment_offset():
    result = validate_timestamps(
        [0, 0.05, 0.10, 0.16, 0.23],
        [0, 0.06, 0.13],
        start_frame=2, end_frame=4,
    )
    assert result.frames == 3


def test_small_duplicate_timestamp_repaired():
    fixed, corrections = repair_timestamp_collisions(
        [0.0, 0.05, 0.05, 0.11], tolerance_seconds=0.001)
    assert fixed[2] > fixed[1]
    assert len(corrections) == 1
    assert fixed[2] - 0.05 <= 0.001


def test_large_backward_jump_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        repair_timestamp_collisions(
            [0.0, 0.05, 0.03], tolerance_seconds=0.001)


def test_repeated_duplicates_cannot_accumulate_unbounded_drift():
    with pytest.raises(ValueError, match="exceeds"):
        repair_timestamp_collisions(
            [0.0] * 5, tolerance_seconds=0.000002)
