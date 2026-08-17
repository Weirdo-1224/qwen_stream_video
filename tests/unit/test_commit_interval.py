from qwen_stream_video.video import (
    VideoMetadata,
    VideoWindow,
    build_video_windows,
    evidence_intersects_commit_interval,
)


def test_overlapping_windows_have_non_overlapping_commit_intervals() -> None:
    metadata = VideoMetadata(path="demo.mp4", fps=1, frame_count=12, duration_seconds=12, width=1, height=1)
    windows = build_video_windows(metadata, 6, 3)
    assert [(w.start_seconds, w.commit_start_seconds, w.end_seconds) for w in windows] == [
        (0.0, 0.0, 6.0), (3.0, 6.0, 9.0), (6.0, 9.0, 12.0)
    ]


def test_evidence_commit_intersection() -> None:
    window = VideoWindow(global_index=1, run_index=1, start_seconds=3, commit_start_seconds=6, end_seconds=9)
    frames = [
        {"sample_index": 0, "frame_index": 0, "timestamp_seconds": 4.0},
        {"sample_index": 1, "frame_index": 1, "timestamp_seconds": 7.0},
    ]
    from qwen_stream_video.video import SampledFrame

    sampled = [SampledFrame(run_index=1, global_index=1, image=None, **frame) for frame in frames]
    assert evidence_intersects_commit_interval([0], sampled, window) is False
    assert evidence_intersects_commit_interval([1], sampled, window) is True
