#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import av
from fractions import Fraction
from timestamp_validation import read_presentation_times, validate_timestamps


def collect_bodypart_triplets(df: pd.DataFrame, bodypart: str):
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
        raise RuntimeError("Expected a DeepLabCut CSV with a 3-row header.")
    scorers = sorted({c[0] for c in df.columns if c[1] == bodypart})
    triplets = []
    for s in scorers:
        xcol = (s, bodypart, "x")
        ycol = (s, bodypart, "y")
        lcol = (s, bodypart, "likelihood")
        if xcol in df.columns and ycol in df.columns and lcol in df.columns:
            triplets.append((xcol, ycol, lcol))
    if not triplets:
        bodyparts = sorted({c[1] for c in df.columns})
        raise KeyError(f"No (x,y,likelihood) for '{bodypart}'. Available: {bodyparts}")
    return triplets


def maxlik_and_coords(df: pd.DataFrame, bodypart: str):
    triplets = collect_bodypart_triplets(df, bodypart)
    L = np.column_stack([df[l].to_numpy(dtype=float) for (_, _, l) in triplets])
    X = np.column_stack([df[x].to_numpy(dtype=float) for (x, _, _) in triplets])
    Y = np.column_stack([df[y].to_numpy(dtype=float) for (_, y, _) in triplets])
    # Treat NaNs as very small for argmax
    L_filled = np.where(np.isnan(L), -1.0, L)
    idx = np.argmax(L_filled, axis=1)
    lik_max = L[np.arange(L.shape[0]), idx]
    cx = X[np.arange(X.shape[0]), idx]
    cy = Y[np.arange(Y.shape[0]), idx]
    return lik_max, cx, cy


def boolean_runs(mask: np.ndarray):
    runs = []
    if mask.size == 0:
        return runs
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    for s, e in zip(starts, ends):
        runs.append((int(s), int(e)))
    return runs


def merge_runs_with_gaps(runs, include_gap: int):
    """Merge adjacent runs if the gap between them is <= include_gap frames."""
    if not runs or include_gap <= 0:
        return runs[:]
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1
        if gap <= include_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def main():
    p = argparse.ArgumentParser(
        description="Split video where DLC condition holds; merge short gaps; crop to coords or draw ROI."
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bodypart", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument("--include-gap", type=int, default=0,
                   help="Merge runs separated by <= this many low-likelihood frames.")
    p.add_argument("--crop", action="store_true", help="Output cropped segments centered on coords.")
    p.add_argument("--box", type=int, default=128, help="Square crop size in pixels.")
    p.add_argument("--debug-roi", action="store_true",
                   help="Draw ROI rectangle and center dot on full frame (overrides --crop).")
    p.add_argument("--timestamp-tolerance-ms", type=float, default=1.0,
                   help="Maximum allowed source/output relative timestamp difference in milliseconds.")
    args = p.parse_args()

    # Validation
    if args.min_frames <= 0:
        raise ValueError("--min-frames must be > 0")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0,1]")
    if args.box <= 0:
        raise ValueError("--box must be > 0")
    if args.include_gap < 0:
        raise ValueError("--include-gap must be >= 0")
    if not np.isfinite(args.timestamp_tolerance_ms) or args.timestamp_tolerance_ms < 0:
        raise ValueError("--timestamp-tolerance-ms must be finite and >= 0")

    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load DLC CSV and compute per-frame best coords
    df = pd.read_csv(args.csv, header=[0, 1, 2])
    lik_max, cx, cy = maxlik_and_coords(df, args.bodypart)

    # Base condition: above threshold AND finite coordinates
    base_mask = (lik_max >= float(args.threshold)) & np.isfinite(cx) & np.isfinite(cy)
    # Initial true runs
    runs = boolean_runs(base_mask)
    # Merge short low-likelihood gaps
    runs = merge_runs_with_gaps(runs, int(args.include_gap))
    # Enforce min length AFTER merging (gap frames count toward length)
    runs = [(s, e) for (s, e) in runs if (e - s + 1) >= int(args.min_frames)]
    if not runs:
        print("No segments met the condition; nothing to write.")
        return

    # Decode presentation-order frames and preserve their source PTS intervals.
    source_times = read_presentation_times(args.video)
    if len(source_times) != len(df):
        raise ValueError(
            f"DLC/video frame mismatch: {len(df)} CSV rows vs "
            f"{len(source_times)} decoded video frames"
        )
    if any(b <= a for a, b in zip(source_times, source_times[1:])):
        raise ValueError("Source video timestamps must be strictly increasing")

    manifest = [
        "segment_index\tstart_frame\tend_frame\tfile\t"
        "source_start_time_s\tsource_end_time_s\tmax_timestamp_error_s"
    ]
    frame_manifest = [
        "segment_index\tsegment_frame\tsource_frame\t"
        "source_time_s\tsegment_time_s"
    ]
    time_base = Fraction(1, 1000000)
    stem = args.video.stem
    written = 0

    with av.open(str(args.video)) as source:
        stream = source.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        do_crop = args.crop and not args.debug_roi
        crop_size = min(max(8, args.box), width, height)
        if do_crop:
            # yuv420p requires even dimensions.
            crop_size -= crop_size % 2
            if crop_size < 2:
                raise ValueError("Video too small for yuv420p cropping")
        elif width % 2 or height % 2:
            raise ValueError("Full-frame H.264 output requires even dimensions")
        out_size = (crop_size, crop_size) if do_crop else (width, height)

        frame_iterator = enumerate(source.decode(video=0))
        current_idx = -1
        for seg_idx, (start, end) in enumerate(runs, 1):
            out_path = outdir / f"{stem}_seg{seg_idx:03d}.mp4"
            segment_origin = source_times[start]
            last_cx = last_cy = None
            output_pts = []
            with av.open(str(out_path), mode="w") as target:
                encoder = target.add_stream("libx264", rate=20)
                encoder.width, encoder.height = out_size
                encoder.pix_fmt = "yuv420p"
                encoder.codec_context.time_base = time_base
                encoder.time_base = time_base
                encoder.options = {"crf": "18", "preset": "medium", "bf": "0"}

                for frame_idx in range(start, end + 1):
                    while current_idx < frame_idx:
                        try:
                            current_idx, frame = next(frame_iterator)
                        except StopIteration:
                            raise ValueError(
                                f"Video ended before source frame {frame_idx}"
                            ) from None
                    image = frame.to_ndarray(format="bgr24")
                    x, y = cx[frame_idx], cy[frame_idx]
                    if np.isfinite(x) and np.isfinite(y):
                        last_cx, last_cy = float(x), float(y)
                    if last_cx is None:
                        last_cx, last_cy = width / 2.0, height / 2.0

                    left = clamp_int(
                        int(round(last_cx)) - crop_size // 2,
                        0, width - crop_size
                    )
                    top = clamp_int(
                        int(round(last_cy)) - crop_size // 2,
                        0, height - crop_size
                    )
                    if do_crop:
                        image = image[top:top + crop_size, left:left + crop_size]
                    elif args.debug_roi:
                        cv2.rectangle(
                            image, (left, top),
                            (left + crop_size, top + crop_size), (0, 255, 0), 2
                        )
                        cv2.circle(
                            image, (round(last_cx), round(last_cy)),
                            3, (0, 0, 255), -1
                        )
                        cv2.putText(
                            image, f"f={frame_idx}", (left, max(0, top - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA
                        )

                    pts = round((source_times[frame_idx] - segment_origin) * 1000000)
                    if output_pts and pts <= output_pts[-1]:
                        raise ValueError(
                            f"Timestamp precision collision at frame {frame_idx}"
                        )
                    output_pts.append(pts)
                    encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
                    encoded.pts = pts
                    encoded.time_base = time_base
                    for packet in encoder.encode(encoded):
                        target.mux(packet)
                    frame_manifest.append(
                        f"{seg_idx}\t{frame_idx - start}\t{frame_idx}\t"
                        f"{source_times[frame_idx]:.9f}\t{pts / 1e6:.9f}"
                    )
                for packet in encoder.encode():
                    target.mux(packet)

            # Decode the actual encoded output: encoder metadata alone is not proof.
            actual_times = read_presentation_times(out_path)
            validation = validate_timestamps(
                source_times, actual_times, start_frame=start, end_frame=end,
                tolerance_seconds=args.timestamp_tolerance_ms / 1000.0
            )
            manifest.append(
                f"{seg_idx}\t{start}\t{end}\t{out_path}\t"
                f"{source_times[start]:.9f}\t{source_times[end]:.9f}\t"
                f"{validation.max_error_seconds:.9f}"
            )
            written += 1
            print(f"Validated segment {seg_idx}: "
                  f"{validation.frames} frames, max error "
                  f"{validation.max_error_seconds * 1000:.3f} ms")

    (outdir / "frame_timestamps.tsv").write_text(
        "\n".join(frame_manifest) + "\n", encoding="utf-8"
    )
    manifest_path = outdir / "segments.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Wrote {written} validated segment(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
