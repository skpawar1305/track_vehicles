# Dumper-truck INT8 calibration set

260 background images for INT8 quantization of the edge detector: real
**dump trucks, tippers and articulated haulers** pulled from Wikimedia Commons
(freely licensed: CC0 / Public domain / CC BY / CC BY-SA — see `ATTRIBUTIONS.md`).
These are general/public photos, **not** site captures.

Each image is letterboxed to the model input size (`320x320`, gray `114`
padding), matching `run.py`'s preprocessing and the production model shape.
`list.txt` is the manifest for the quantizer (paths relative to the repo root,
so run it from the repo root).

## Regenerate

    python calib/dumper_trucks/build_calib.py --shape 320 320

Downloads from Wikimedia Commons into `_src/` and letterboxes into this folder.
Record the source/license in `ATTRIBUTIONS.md` when refreshing.

## Use

    python tools/quantize_int8.py \
        --model models/yolo26n_ncnn_320x320 \
        --calib calib/dumper_trucks/list.txt \
        --shape 320 320 \
        --tools <ncnn bin dir with ncnn2table/ncnn2int8>
