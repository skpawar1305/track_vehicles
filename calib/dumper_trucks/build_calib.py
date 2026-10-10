#!/usr/bin/env python3
"""Build the dumper-truck INT8 calibration set.

Downloads dump-truck / tipper / articulated-hauler photos from Wikimedia Commons
(freely licensed; see ATTRIBUTIONS.md) and letterboxes each to the model input
size, writing the images + list.txt into this folder for tools/quantize_int8.py.

    python calib/dumper_trucks/build_calib.py --shape 320 320

Images must already be at the model input size for the quantizer, so pick the
shape that matches the model being quantized.
"""
import argparse
import glob
import json
import os
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

UA = "tracker-calib/1.0 (https://github.com/skpawar1305/track_vehicles)"
API = "https://commons.wikimedia.org/w/api.php"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_src")
QUERIES = ["dump truck", "dumper truck", "tipper truck", "articulated hauler",
           "mining dump truck", "quarry truck", "haul truck", "dump trailer"]
CAP = 260


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def fetch():
    os.makedirs(SRC, exist_ok=True)
    titles, seen = [], set()
    for q in QUERIES:
        off = 0
        while off < 300:
            u = (f"{API}?action=query&list=search&srsearch={urllib.parse.quote(q)}"
                 f"&srnamespace=6&srlimit=100&sroffset={off}&format=json")
            try:
                d = _get(u)
            except Exception as e:
                print("search err", q, e); break
            res = d.get("query", {}).get("search", [])
            for x in res:
                if x["title"] not in seen:
                    seen.add(x["title"]); titles.append(x["title"])
            if "continue" not in d or not res:
                break
            off = d["continue"]["sroffset"]; time.sleep(0.3)
    dl = []
    for i in range(0, len(titles), 50):
        u = (f"{API}?action=query&titles={urllib.parse.quote('|'.join(titles[i:i+50]))}"
             f"&prop=imageinfo&iiprop=url|extmetadata|mime&iiurlwidth=1280&format=json")
        try:
            d = _get(u)
        except Exception as e:
            print("info err", e); continue
        for page in d.get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo")
            if not ii:
                continue
            info = ii[0]
            if "jpeg" not in info.get("mime", "") and "png" not in info.get("mime", ""):
                continue
            dl.append((page["title"], info.get("thumburl") or info.get("url"),
                       info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")))
        time.sleep(0.3)
    manifest = []
    for i, (title, url, lic) in enumerate(dl[:CAP]):
        ext = ".png" if ".png" in url.lower().split("?")[0] else ".jpg"
        dst = os.path.join(SRC, f"{i:04d}{ext}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r, open(dst, "wb") as f:
                f.write(r.read())
            manifest.append({"file": os.path.basename(dst), "title": title, "license": lic})
        except Exception as e:
            print("dl err", i, e)
        time.sleep(0.05)
    json.dump(manifest, open(os.path.join(SRC, "manifest.json"), "w"), indent=1)
    print("fetched", len(manifest), "images ->", SRC)


def letterbox(shape):
    H, W = shape
    src = sorted(glob.glob(os.path.join(SRC, "*.jpg")) + glob.glob(os.path.join(SRC, "*.png")))
    for old in glob.glob(os.path.join(HERE, "*.jpg")):
        os.remove(old)
    listing = []
    for i, p in enumerate(src):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(W / w, H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        c = np.full((H, W, 3), 114, dtype=np.uint8)
        pw, ph = (W - nw) // 2, (H - nh) // 2
        c[ph:ph + nh, pw:pw + nw] = r
        dst = os.path.join(HERE, f"{i:04d}.jpg")
        cv2.imwrite(dst, c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        listing.append(os.path.relpath(dst, os.path.dirname(os.path.dirname(HERE))))
    open(os.path.join(HERE, "list.txt"), "w").write("\n".join(listing) + "\n")
    print(f"letterboxed {len(listing)} images to {H}x{W}; wrote list.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=2, type=int, default=[320, 320], metavar=("W", "H"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download or not os.path.isdir(SRC):
        fetch()
    letterbox((args.shape[1], args.shape[0]))
