#!/usr/bin/env python3
"""INT8-quantize an ncnn model for a faster / smaller edge model.

Requires the ncnn tools `ncnn2table` and `ncnn2int8` (ship in the ncnn release
zips, e.g. ncnn-YYYYMMDD-ubuntu-2404.zip -> bin/). Point at them with --tools or
put them on PATH.

    python tools/quantize_int8.py \
        --model models/yolo26n_ncnn_320x512 \
        --calib calib/list.txt \
        --shape 512 320

Writes model_int8.ncnn.param / model_int8.ncnn.bin into the model dir, which
run.py picks up when TRACKER_INT8=1.

Calibration images must already be at the model input size (use the same
letterboxing as inference) and are read in RGB with pixel/255 normalisation,
matching run.py's preprocessing.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_tool(name, tools_dir):
    if tools_dir:
        p = Path(tools_dir) / name
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"error: {name} not found (use --tools DIR pointing at ncnn bin/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir with model.ncnn.param/.bin")
    ap.add_argument("--calib", required=True, help="newline-separated calibration image list")
    ap.add_argument("--shape", nargs=2, type=int, default=[512, 320], metavar=("W", "H"))
    ap.add_argument("--tools", default=None, help="dir containing ncnn2table/ncnn2int8")
    ap.add_argument("--method", default="kl", choices=["kl", "aciq", "eq"])
    args = ap.parse_args()

    mdir = Path(args.model)
    param, binp = mdir / "model.ncnn.param", mdir / "model.ncnn.bin"
    if not param.exists() or not binp.exists():
        sys.exit(f"error: {param} or {binp} missing")

    table = mdir / "model_int8.table"
    out_param = mdir / "model_int8.ncnn.param"
    out_bin = mdir / "model_int8.ncnn.bin"
    norm = 1.0 / 255.0

    ncnn2table = find_tool("ncnn2table", args.tools)
    ncnn2int8 = find_tool("ncnn2int8", args.tools)

    subprocess.run([
        ncnn2table, str(param), str(binp), args.calib, str(table),
        "mean=[0,0,0]", f"norm=[{norm},{norm},{norm}]",
        f"shape=[{args.shape[0]},{args.shape[1]}]", "pixel=RGB",
        f"method={args.method}", f"thread={os.cpu_count() or 4}",
    ], check=True)
    subprocess.run([ncnn2int8, str(param), str(binp), str(out_param), str(out_bin), str(table)], check=True)
    print(f"[int8] {out_param}  {out_bin}  (table {table})")


if __name__ == "__main__":
    main()
