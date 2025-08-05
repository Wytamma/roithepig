#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="input_csv",  required=True)
    p.add_argument("--out", dest="output_csv", required=True)
    p.add_argument("--sx", type=float, required=True, help="Scale factor for x (e.g., 2.0 if downscaled by 0.5)")
    p.add_argument("--sy", type=float, required=True, help="Scale factor for y (e.g., 2.0 if downscaled by 0.5)")
    p.add_argument("--cropping", type=str, default=None, help="Cropping coordinates in the format 'x1,x2,y1,y2'.")
    args = p.parse_args()

    try:
        # DLC CSVs have a 3-row header: (scorer, bodypart, coord)
        df = pd.read_csv(args.input_csv, header=[0, 1, 2])
    except Exception as e:
        print(f"Failed to read DLC CSV with 3-line header: {e}", file=sys.stderr)
        raise

    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
        raise ValueError("Expected DLC CSV with 3-level header (scorer, bodypart, coord).")

    # Collect the MultiIndex columns whose 3rd level is 'x' or 'y' (case-insensitive)
    x_cols = [col for col in df.columns if isinstance(col, tuple) and str(col[2]).lower() == "x"]
    y_cols = [col for col in df.columns if isinstance(col, tuple) and str(col[2]).lower() == "y"]
    x1,y1 = 0, 0
    if args.cropping:
        try:
            x1, _, y1, _ = map(int, args.cropping.split(','))
        except ValueError:
            raise ValueError("Invalid cropping format. Use 'x1,x2,y1,y2'.")
    print(df)
    # Scale x and y only. Likelihood remains unchanged.
    if x_cols:
        # Coerce to numeric first to avoid issues if pandas read them as object
        df[x_cols] = df[x_cols] + x1  # Add cropping offset if specified
        df[x_cols] = df[x_cols].apply(pd.to_numeric, errors="coerce") * float(args.sx)
    if y_cols:
        df[y_cols] = df[y_cols] + y1  # Add cropping offset if specified
        df[y_cols] = df[y_cols].apply(pd.to_numeric, errors="coerce") * float(args.sy)
    print(df)
    # Preserve the exact header structure
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
