#!/usr/bin/env python3
"""Build an INT8 calibration image set matching inference exactly.

Crops the configured ROI from each frame and letterboxes it to the model input,
so the quantizer sees the same distribution the model runs on (calibrating on
whole frames gives poor INT8 accuracy for ROI inference).

    python tools/make_calib.py --frames /tmp/frames --out calib --shape 320 320 \
        --roi 0.004,0.337 0.300,0.011 0.546,0.016 0.083,0.992

Writes <out>/list.txt for use with tools/quantize_int8.py --calib.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of full frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", nargs=2, type=int, required=True, metavar=("W", "H"))
    ap.add_argument("--roi", nargs="+", required=True,
                    help="normalized x,y points: x0,y0 x1,y1 [...]")
    ap.add_argument("--pad", type=int, default=114)
    args = ap.parse_args()

    W, H = args.shape
    roi = [[float(v) for v in p.split(",")] for p in args.roi]
    os.makedirs(args.out, exist_ok=True)

    files = []
    for j, path in enumerate(sorted(glob.glob(os.path.join(args.frames, "*")))):
        f = cv2.imread(path)
        if f is None:
            continue
        fh, fw = f.shape[:2]
        xs = [int(x * fw) for x, _ in roi]
        ys = [int(y * fh) for _, y in roi]
        crop = f[min(ys):max(ys), min(xs):max(xs)]
        sc = min(W / crop.shape[1], H / crop.shape[0])
        nw, nh = int(round(crop.shape[1] * sc)), int(round(crop.shape[0] * sc))
        canvas = np.full((H, W, 3), args.pad, np.uint8)
        pw, ph = (W - nw) // 2, (H - nh) // 2
        canvas[ph:ph + nh, pw:pw + nw] = cv2.resize(crop, (nw, nh))
        out = os.path.join(args.out, f"c_{j:03d}.jpg")
        cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(out)

    with open(os.path.join(args.out, "list.txt"), "w") as fh:
        fh.write("\n".join(files) + "\n")
    print(f"[calib] {len(files)} ROI crops -> {args.out}/list.txt")


if __name__ == "__main__":
    main()
