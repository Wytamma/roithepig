#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def list_segment_files(seg_dir: Path, exts=(".mp4", ".mov", ".mkv", ".avi")) -> List[Path]:
    files = sorted([p for p in seg_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No segment files with extensions {exts} found in {seg_dir}")
    return files


def parse_cell(cell: str) -> Tuple[int, int]:
    cell = cell.lower()
    if "x" in cell:
        w, h = cell.split("x")
        return int(w), int(h)
    s = int(cell)
    return s, s


def compute_grid(n: int, cols_opt: int | None) -> Tuple[int, int]:
    if cols_opt and cols_opt > 0:
        cols = cols_opt
        rows = math.ceil(n / cols)
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    return rows, cols


def letterbox(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Create a synchronized grid of segment videos using OpenCV.")
    ap.add_argument("--segments", type=Path, required=True, help="Directory containing segment clips.")
    ap.add_argument("--out", type=Path, required=True, help="Output MP4 path.")
    ap.add_argument("--cell", type=str, default="320x320", help='Cell WxH or single int, default 320x320.')
    ap.add_argument("--fps", type=float, default=30.0, help="Output FPS.")
    ap.add_argument("--cols", type=int, default=0, help="Number of columns; 0 = auto square-ish.")
    args = ap.parse_args()

    seg_dir = args.segments.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = parse_cell(args.cell)
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("--cell must be positive")
    out_fps = float(args.fps)
    if out_fps <= 0:
        raise ValueError("--fps must be > 0")

    files = list_segment_files(seg_dir)
    rows, cols = compute_grid(len(files), args.cols if args.cols > 0 else None)

    # Open all captures and gather stream info
    caps = []
    infos = []
    max_duration = 0.0
    for f in files:
        cap = cv2.VideoCapture(str(f))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {f}")
        fps_i = cap.get(cv2.CAP_PROP_FPS) or out_fps
        if fps_i <= 1e-6:
            fps_i = out_fps
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        duration = (nframes / fps_i) if (fps_i > 0 and nframes > 0) else 0.0
        max_duration = max(max_duration, duration)
        caps.append(cap)
        infos.append({
            "fps": fps_i,
            "nframes": nframes,
            "frame_idx": 0,
            "ended": False,
            "last_frame": None,
            "path": f,
        })

    # If any duration is unknown (0), fall back to max by frame count heuristic
    if max_duration <= 0:
        # best-effort: take maximum nframes/fps across sources with valid stats
        for inf in infos:
            if inf["fps"] > 0 and inf["nframes"] > 0:
                max_duration = max(max_duration, inf["nframes"] / inf["fps"])
        if max_duration <= 0:
            # If still unknown, bail out conservatively after 10 seconds
            max_duration = 10.0

    # Output writer
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (grid_w, grid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    # Timeline: generate frames on a common clock; freeze each tile at last frame once its source ends
    total_out_frames = int(math.ceil(max_duration * out_fps))

    for k in range(total_out_frames):
        t = k / out_fps
        tiles = []

        for i, cap in enumerate(caps):
            inf = infos[i]
            fps_i = inf["fps"]
            if not inf["ended"]:
                # Which source frame should we be at for timestamp t?
                target_idx = int(round(t * fps_i))
                # Read forward until we reach target_idx or end
                while inf["frame_idx"] <= target_idx and not inf["ended"]:
                    ok, frame = cap.read()
                    if not ok:
                        inf["ended"] = True
                        break
                    inf["last_frame"] = frame
                    inf["frame_idx"] += 1
                # If we overshot by rounding, we will just keep last_frame
            # Use last_frame (frozen if ended or waiting)
            if inf["last_frame"] is None:
                # If we have nothing yet, create a black tile
                tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            else:
                tile = letterbox(inf["last_frame"], cell_w, cell_h)
            tiles.append(tile)

        # Compose grid
        rows_imgs = []
        for r in range(rows):
            row_tiles = tiles[r*cols:(r+1)*cols]
            # If last row is incomplete, fill missing tiles with black
            if len(row_tiles) < cols:
                row_tiles += [np.zeros((cell_h, cell_w, 3), dtype=np.uint8) for _ in range(cols - len(row_tiles))]
            rows_imgs.append(np.hstack(row_tiles))
        grid = np.vstack(rows_imgs)

        writer.write(grid)

    # Clean up
    writer.release()
    for cap in caps:
        cap.release()

    print(f"Wrote grid: {out_path}")


if __name__ == "__main__":
    main()
