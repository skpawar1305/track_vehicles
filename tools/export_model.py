#!/usr/bin/env python3
"""Export a YOLO model to ncnn at a chosen input size.

Square (matches Ultralytics default):

    python tools/export_model.py --imgsz 640

Aspect-matched (avoids padding the network input with gray; HxW, both %32):

    python tools/export_model.py --imgsz 384x640

Output lands in models/<name or yolo11n_ncnn_<imgsz>>/ with model.ncnn.param,
model.ncnn.bin and metadata.yaml.
"""
import argparse
import shutil
from pathlib import Path


def parse_size(s: str):
    s = str(s).lower()
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    n = int(s)
    return n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolo11n.pt")
    ap.add_argument("--imgsz", default="384x640", help="NN (square) or HxW")
    ap.add_argument("--name", default=None, help="output dir name under models/")
    args = ap.parse_args()

    h, w = parse_size(args.imgsz)
    if h % 32 or w % 32:
        ap.error(f"imgsz {h}x{w}: both dims must be multiples of 32 (YOLO stride)")

    import subprocess
    import tempfile
    from ultralytics import YOLO

    model = YOLO(args.weights)
    stem = Path(args.weights).stem
    dest = Path("models") / (args.name or f"{stem}_ncnn_{args.imgsz}")
    dest.mkdir(parents=True, exist_ok=True)

    # 1. ONNX at the requested input size.
    onnx_path = Path(model.export(format="onnx", imgsz=[h, w], opset=12))

    # 2. ncnn via pnnx. Some models (e.g. end-to-end YOLO26) emit a
    #    `pnnx.Expression` layer this ncnn build can't load, so verify and fall
    #    back to Ultralytics' own ncnn export, which handles them.
    work = Path(tempfile.mkdtemp())
    shutil.copy(onnx_path, work / "model.onnx")
    subprocess.run(["pnnx", "model.onnx", f"inputshape=[1,3,{h},{w}]"], cwd=work, check=True)
    param_text = (work / "model.ncnn.param").read_text()

    if "pnnx.Expression" in param_text or "torch." in param_text:
        print("[export] pnnx output has unsupported layers; using Ultralytics ncnn export")
        exported = Path(YOLO(args.weights).export(format="ncnn", imgsz=[h, w]))
        shutil.copy(exported / "model.ncnn.param", dest / "model.ncnn.param")
        shutil.copy(exported / "model.ncnn.bin", dest / "model.ncnn.bin")
    else:
        shutil.copy(work / "model.ncnn.param", dest / "model.ncnn.param")
        shutil.copy(work / "model.ncnn.bin", dest / "model.ncnn.bin")

    (dest / "INPUT_SHAPE").write_text(f"{h} {w}\n")
    print(f"[export] imgsz={h}x{w} -> {dest} (param, bin, INPUT_SHAPE)")


if __name__ == "__main__":
    main()
