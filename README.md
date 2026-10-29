# roithepig

[![PyPI - Version](https://img.shields.io/pypi/v/roithepig.svg)](https://pypi.org/project/roithepig)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/roithepig.svg)](https://pypi.org/project/roithepig)

-----

RoiThePig is a small video-processing pipeline that uses DeepLabCut to detect a body part (e.g., a pig’s ear), crops the region-of-interest around it, splits the video into segments where detections are confident, and renders a grid preview of all segments. It’s packaged as a single command backed by a Snakemake workflow.

- Detect with DeepLabCut (config/model bundled)
- Optional downscaling for faster inference, then rescales keypoints back
- Split video into segments using likelihood threshold, min length, and gap merging
- Crop a square ROI around the body part (or draw a debug ROI on full frames)
- Create a tiled segment grid video for quick QC

**Table of Contents**

- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI Options](#cli-options)
- [Outputs](#outputs)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Requirements

- Python 3.8–3.12
- ffmpeg available on PATH
- Conda/Mamba recommended. The workflow defines a DeepLabCut environment at `src/roithepig/workflow/envs/deeplabcut.yaml` and will create/use it automatically when the pipeline runs.

## Installation

```console
pip install roithepig
```

## Quickstart

Process a video and produce segments plus a segment grid:

```console
roithepig run --video /path/to/video.mp4
```

Results are written to `output/<VIDEO_STEM>/`.

## CLI Options

The command reads its interface from `snk.yaml`. Common options:

- `--video PATH` (required): Input video file
- `--bodypart TEXT` (default: "L ear base"): Body part to track/crop
- `--threshold FLOAT` (default: 0.6): Minimum DLC likelihood to keep frames
- `--downscale-factor FLOAT` (default: 1.0): Downscale before DLC for speed
- `--min-frames INT` (default: 30): Minimum segment length (frames)
- `--include-gap INT` (default: 1): Merge runs separated by ≤ this many low-likelihood frames
- `--cropping "x1,x2,y1,y2"`: Optional pre-cropping passed to DLC
- `--box INT` (default: 50): ROI box size (pixels)
- `--batch-size INT` (default: 2): DLC inference batch size
- `--debug` (flag): Draw ROI rectangle/center instead of cropping
- `--output-dir PATH` (default: `output`): Where to write results

For full help:

```console
roithepig --help
roithepig run --help
```

## Outputs

Given an input `/videos/pig.mp4`, the pipeline writes to `output/pig/`:

- `pig.ds.mp4` — downscaled video used for DLC (may be a symlink/copy if factor=1.0)
- `pig.csv` — DeepLabCut keypoints rescaled to original resolution
- `segments/` — cropped (or debug-annotated) segment clips and `segments.tsv`
- `segment_grid.mp4` — tiled grid preview of all segments

## Examples

- Faster DLC via downscaling, stricter segmenting, larger crop box:

```console
roithepig run \
  --video tests/pig.mp4 \
  --bodypart "L ear base" \
  --threshold 0.7 \
  --min-frames 45 \
  --include-gap 2 \
  --box 128 \
  --downscale-factor 0.5
```

## Troubleshooting

- Missing ffmpeg: install via your package manager (e.g., Homebrew on macOS).
- DeepLabCut env creation: ensure you have Conda/Mamba configured; the workflow will create/use the `DEEPLABCUT` env defined in `envs/deeplabcut.yaml` automatically.
- CPU vs GPU: the shipped environment works on CPU; GPU is optional and depends on your system CUDA stack.

## Development

- Run tests (via Hatch):

```console
hatch run test
```

- Lint types:

```console
hatch run types:check
```

## License

`roithepig` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


### Variable-frame-rate recordings and timestamp validation

ROI segments preserve the source video's decoded presentation-time intervals instead of using a fixed OpenCV output FPS. This is important for recordings whose acquisition rate varies (for example, 16–20 FPS). Segments use timestamps relative to their first frame, and `segments/frame_timestamps.tsv` retains the original decoded timestamps with their source frame indices. `segments/segments.tsv` records each segment's source range and maximum measured relative timestamp error.

The `crop_and_split` step decodes each segment after writing it and checks that all expected frames are present, the timestamp order is strictly increasing, and every relative presentation timestamp differs from the source by at most `--timestamp-tolerance-ms` (default 1 ms). The corresponding CLI configuration key is `timestamp_tolerance_ms`. A violation raises an error and fails the Snakemake job. DLC CSV row counts must match the cleaned video's decoded frame count; mismatches fail rather than silently truncating.

This check compares against the cleaned video passed into segmentation. It cannot recover camera acquisition times that were lost before or during recording or previous transcoding. The composite `segment_grid.mp4` is a fixed-FPS visualization, not a timing-preserving scientific output. The source and cleaned video timestamps should be checked separately if absolute acquisition timing is required.


#### Timestamp collisions in cleaned recordings

Small duplicate or backward presentation timestamps in decoded source frames are repaired only when the required adjustment is at most `timestamp_tolerance_ms` (default 1 ms) for every affected frame. The original source times are retained in `segments/frame_timestamps.tsv` and each correction is recorded in `segments/timestamp_corrections.tsv`. Larger anomalies fail the workflow: do not increase the tolerance without inspecting the original acquisition file. Repairs make encoding timestamps representable but **cannot recover missing acquisition timestamps**. A microsecond encoder time base is used in the FFmpeg cleaning and downscaling steps to reduce accidental timestamp collisions. The segmented video timing is validated against the *repaired* cleaned-video timestamps, not original camera clock measurements. RGB extraction and iPPG analysis remain separate from RoiThePig.
