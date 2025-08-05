#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


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

    # Video IO
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else len(df)
    max_idx = min(n_video, len(df)) - 1
    runs = [(max(0, s), min(e, max_idx)) for s, e in runs if s <= max_idx]

    # Output shape and mode
    crop_size = max(8, min(int(args.box), width, height))
    do_crop = args.crop and not args.debug_roi
    out_size = (crop_size, crop_size) if do_crop else (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stem = Path(args.video).stem
    manifest = ["segment_index\tstart_frame\tend_frame\tfile"]

    run_i = 0
    writer = None
    current_end = -1
    frame_idx = 0
    next_start = runs[run_i][0] if run_i < len(runs) else None

    # Per-segment last-good center for gap frames or NaNs
    last_cx = None
    last_cy = None

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx > max_idx:
            break

        # Advance to next run if past current
        while next_start is not None and frame_idx > runs[run_i][1]:
            if writer is not None:
                writer.release()
                writer = None
            run_i += 1
            next_start = runs[run_i][0] if run_i < len(runs) else None
            last_cx = last_cy = None  # reset per segment

        # Start new segment
        if next_start is not None and frame_idx == next_start:
            seg_idx = run_i + 1
            out_path = outdir / f"{stem}_seg{seg_idx:03d}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, out_size)
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not open output for writing: {out_path}")
            current_end = runs[run_i][1]
            manifest.append(f"{seg_idx}\t{frame_idx}\t{current_end}\t{out_path}")

        if writer is not None:
            # Use coords even inside included gaps.
            x = cx[frame_idx]
            y = cy[frame_idx]
            if np.isfinite(x) and np.isfinite(y):
                last_cx, last_cy = float(x), float(y)
            # Fallback in gap frames with NaN:
            if last_cx is None or last_cy is None:
                last_cx = width / 2.0
                last_cy = height / 2.0

            half = crop_size // 2
            left = int(round(last_cx)) - half
            top = int(round(last_cy)) - half
            left = clamp_int(left, 0, max(0, width - crop_size))
            top = clamp_int(top, 0, max(0, height - crop_size))

            if do_crop:
                roi = frame[top: top + crop_size, left: left + crop_size]
                if roi.shape[0] != crop_size or roi.shape[1] != crop_size:
                    roi = cv2.copyMakeBorder(
                        roi,
                        top=0, bottom=crop_size - roi.shape[0],
                        left=0, right=crop_size - roi.shape[1],
                        borderType=cv2.BORDER_REPLICATE,
                    )
                writer.write(roi)
            else:
                if args.debug_roi:
                    cv2.rectangle(frame, (left, top), (left + crop_size, top + crop_size), (0, 255, 0), 2)
                    cv2.circle(frame, (int(round(last_cx)), int(round(last_cy))), 3, (0, 0, 255), -1)
                    # Lightweight label
                    cv2.putText(
                        frame, f"f={frame_idx}",
                        (left, max(0, top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )
                writer.write(frame)

            if frame_idx == current_end:
                writer.release()
                writer = None
                last_cx = last_cy = None

        frame_idx += 1

    if writer is not None:
        writer.release()
    cap.release()

    manifest_path = outdir / "segments.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Wrote {len(runs)} segment(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
