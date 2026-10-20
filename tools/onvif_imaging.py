#!/usr/bin/env python3
"""Inspect and change a camera's ONVIF imaging settings (WS-UsernameToken).

Built for the PT-NC120D3-WNM(D2) night profile. The camera is ONVIF, but its
RTSP Digest is SHA-256-only, so the vendor UI is inconvenient; this talks the
ONVIF Imaging service directly.

Safe by default: with no ``--apply`` it prints the current settings and the
planned changes, but sends no Set request. The Set request re-sends the
camera's own ``<tt:ImagingSettings>`` XML with only the targeted tags
substituted, so fields the camera requires (Exposure, WhiteBalance,
NoiseReduction extension, ...) are preserved rather than dropped.

Credentials come from --password / --user, the TRACKER_CAM_PASSWORD env var, or
a config.json ``stream_url`` (pass --config; it also auto-reads ./config.json
and /opt/tracker/config.json). Never hard-coded.

On this site ``config.json``'s ``stream_url`` is the NVR (192.168.225.150), while
the counting camera's imaging service is on 192.168.225.71: pass
``--host 192.168.225.71 --config /opt/tracker/config.json`` (the password is
still read from the config).

    # read current settings + supported options
    python tools/onvif_imaging.py --config /opt/tracker/config.json

    # headlight night profile: WDR on at level 50 (dry-run)
    python tools/onvif_imaging.py --config /opt/tracker/config.json \
        --wdr on --wdr-level 50

    # apply it
    python tools/onvif_imaging.py --config /opt/tracker/config.json \
        --wdr on --wdr-level 50 --apply
"""
import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onvif_media import _wsse, MEDIA_NS  # noqa: E402

IMAGING_NS = "http://www.onvif.org/ver20/imaging/wsdl"


def _flat(text):
    return re.sub(r"\s+", " ", text).strip()


def _ok(resp):
    return "Response" in resp and "Fault" not in resp


def _fault(resp):
    m = re.search(r"<[^>]*Reason>\s*<[^>]*Text[^>]*>([^<]+)", resp)
    return m.group(1) if m else _flat(resp)[:240]


def _local(tag):
    return tag.split("}")[-1]


def _to_obj(e):
    kids = list(e)
    if not kids:
        return (e.text or "").strip() or None
    out = {}
    for c in kids:
        k, v = _local(c.tag), _to_obj(c)
        if k in out:
            if not isinstance(out[k], list):
                out[k] = [out[k]]
            out[k].append(v)
        else:
            out[k] = v
    return out


class OnvifImaging:
    def __init__(self, host, user, password, timeout=12.0):
        self.host = host
        self.user = user
        self.password = password
        self.timeout = timeout
        self.device_url = f"https://{host}/onvif/device_service"
        self.ctx = ssl._create_unverified_context()

    def _post(self, url, inner):
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
            f'xmlns:trt="{MEDIA_NS}" '
            'xmlns:tt="http://www.onvif.org/ver10/schema" '
            f'xmlns:timg="{IMAGING_NS}">'
            f'{_wsse(self.user, self.password)}<soap:Body>{inner}</soap:Body></soap:Envelope>')
        req = urllib.request.Request(
            url, data=body.encode(),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        try:
            return urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout).read().decode()
        except urllib.error.HTTPError as e:
            return e.read().decode()
        except (urllib.error.URLError, OSError) as e:
            return f"<Error>{e}</Error>"

    def discover(self):
        x = self._post(self.device_url,
                       "<tds:GetServices><tds:IncludeCapability>false</tds:IncludeCapability></tds:GetServices>")
        img_url = media_url = None
        for block in re.findall(r"<tds:Service>.*?</tds:Service>", x, re.S):
            ns = re.search(r"<tds:Namespace>([^<]+)</tds:Namespace>", block)
            xa = re.search(r"<tds:XAddr>([^<]+)</tds:XAddr>", block)
            if not ns or not xa:
                continue
            if "imaging" in ns.group(1):
                img_url = xa.group(1).strip()
            if ns.group(1) == MEDIA_NS:
                media_url = xa.group(1).strip()
        return (img_url or f"https://{self.host}/onvif/Imaging",
                media_url or f"https://{self.host}/onvif/Media")

    def video_source_token(self, media_url, explicit=None):
        if explicit:
            return explicit
        x = self._post(media_url, "<trt:GetVideoSources/>")
        m = re.search(r'<trt:VideoSources[^>]*token="([^"]+)"', x)
        if not m:
            raise SystemExit(f"no VideoSource token in GetVideoSources: {_flat(x)[:240]}")
        return m.group(1)

    def get(self, img_url, token):
        x = self._post(
            img_url,
            f"<timg:GetImagingSettings><timg:VideoSourceToken>{token}</timg:VideoSourceToken>"
            "</timg:GetImagingSettings>")
        m = re.search(r"<(?:\w+:)?ImagingSettings>(.*?)</(?:\w+:)?ImagingSettings>", x, re.S)
        if not m:
            raise SystemExit(f"GetImagingSettings failed: {_fault(x)}")
        return m.group(1)

    def options(self, img_url, token):
        return self._post(
            img_url,
            f"<timg:GetOptions><timg:VideoSourceToken>{token}</timg:VideoSourceToken>"
            "</timg:GetOptions>")

    def set(self, img_url, token, block):
        return self._post(
            img_url,
            f"<timg:SetImagingSettings><timg:VideoSourceToken>{token}</timg:VideoSourceToken>"
            f"<timg:ImagingSettings>{block}</timg:ImagingSettings></timg:SetImagingSettings>")


def parse_settings(block):
    """Parse the inner ImagingSettings XML into a JSON-able dict."""
    root = ET.fromstring(
        '<root xmlns:tt="http://www.onvif.org/ver10/schema">'
        f"{block}</root>")
    return {_local(c.tag): _to_obj(c) for c in root}


def _replace_scalar(block, tag, value):
    pat = rf"<tt:{tag}>[^<]*</tt:{tag}>"
    if re.search(pat, block):
        return re.sub(pat, f"<tt:{tag}>{value}</tt:{tag}>", block, count=1)
    return block


def _replace_block(block, tag, xml):
    pat = rf"<tt:{tag}>.*?</tt:{tag}>"
    if re.search(pat, block, re.S):
        return re.sub(pat, xml, block, count=1)
    return block


def _insert_before(block, before_tag, xml):
    pat = rf"<tt:{before_tag}>"
    m = re.search(pat, block)
    if m:
        return block[:m.start()] + xml + block[m.start():]
    return block + xml


def apply_changes(block, wdr=None, wdr_level=None, blc=None, ircut=None,
                  brightness=None, contrast=None, saturation=None,
                  sharpness=None, max_gain=None):
    """Replace only the given fields; return (new_block, changes)."""
    new, changes = block, []
    if wdr is not None or wdr_level is not None:
        cur_mode = re.search(r"<tt:WideDynamicRange>.*?<tt:Mode>([^<]*)</tt:Mode>", block, re.S)
        cur_level = re.search(r"<tt:WideDynamicRange>.*?<tt:Level>([^<]*)</tt:Level>", block, re.S)
        mode = (wdr or (cur_mode.group(1) if cur_mode else "OFF")).upper()
        level = wdr_level if wdr_level is not None else (cur_level.group(1) if cur_level else "0")
        xml = (f"<tt:WideDynamicRange><tt:Mode>{mode}</tt:Mode>"
               f"<tt:Level>{level}</tt:Level></tt:WideDynamicRange>")
        new = _replace_block(new, "WideDynamicRange", xml)
        changes.append(f"WideDynamicRange={mode}/{level}")
    if blc is not None:
        xml = f"<tt:BacklightCompensation><tt:Mode>{blc.upper()}</tt:Mode></tt:BacklightCompensation>"
        new = _replace_block(new, "BacklightCompensation", xml)
        changes.append(f"BacklightCompensation={blc.upper()}")
    if ircut is not None:
        new = _replace_scalar(new, "IrCutFilter", ircut.upper())
        changes.append(f"IrCutFilter={ircut.upper()}")
    for tag, value in (("Brightness", brightness), ("ColorSaturation", saturation),
                       ("Contrast", contrast), ("Sharpness", sharpness)):
        if value is not None:
            new = _replace_scalar(new, tag, value)
            changes.append(f"{tag}={value}")
    if max_gain is not None:
        def sub(m):
            return re.sub(r"<tt:MaxGain>[^<]*</tt:MaxGain>",
                          f"<tt:MaxGain>{max_gain}</tt:MaxGain>", m.group(0), count=1)
        new = re.sub(r"<tt:Exposure>.*?</tt:Exposure>", sub, new, count=1)
        changes.append(f"Exposure/MaxGain={max_gain}")
    return new, changes


def _creds_from_config(path):
    try:
        with open(path) as f:
            url = json.load(f).get("stream_url", "")
    except (OSError, ValueError):
        return None
    if not url.startswith("rtsp://"):
        return None
    u = urlsplit(url)
    return u.hostname, unquote(u.username or "admin"), unquote(u.password or "")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default=os.environ.get("TRACKER_CAM_PASSWORD", ""))
    ap.add_argument("--config", help="config.json to read stream_url from")
    ap.add_argument("--token", help="video source token (auto-detected by default)")
    ap.add_argument("--wdr", choices=["on", "off"], help="WideDynamicRange mode")
    ap.add_argument("--wdr-level", type=int, help="WideDynamicRange level (0-100)")
    ap.add_argument("--blc", choices=["on", "off"], help="BacklightCompensation mode")
    ap.add_argument("--ircut", choices=["auto", "on", "off"], help="IrCutFilter mode")
    ap.add_argument("--brightness", type=int)
    ap.add_argument("--contrast", type=int)
    ap.add_argument("--saturation", type=int)
    ap.add_argument("--sharpness", type=int)
    ap.add_argument("--max-gain", type=int)
    ap.add_argument("--get", action="store_true", help="only print current settings/options")
    ap.add_argument("--apply", action="store_true", help="send SetImagingSettings")
    args = ap.parse_args()

    host, user, password = args.host, args.user, args.password
    if not host or not password:
        for path in filter(None, [args.config, "config.json", "/opt/tracker/config.json"]):
            creds = _creds_from_config(path)
            if creds:
                host = host or creds[0]
                user = args.user if args.user != "admin" else creds[1]
                password = password or creds[2]
                print(f"[imaging] credentials from {path}")
                break
    if not host or not password:
        ap.error("provide --host/--password, --config, or set TRACKER_CAM_PASSWORD")

    cam = OnvifImaging(host, user, password)
    img_url, media_url = cam.discover()
    print(f"[imaging] {host}  imaging={img_url}")
    token = cam.video_source_token(media_url, args.token)
    block = cam.get(img_url, token)
    settings = parse_settings(block)
    print("[imaging] current:")
    print(json.dumps(settings, indent=2, sort_keys=True))

    opts = cam.options(img_url, token)
    if "ImagingOptions" in opts:
        root = ET.fromstring(opts)
        for e in root.iter():
            if _local(e.tag) == "ImagingOptions":
                o = _to_obj(e)
                print("[imaging] options:")
                for k in ("IrCutFilter", "Exposure", "WideDynamicRange",
                          "BacklightCompensation", "Brightness", "ColorSaturation",
                          "Contrast", "Sharpness"):
                    if k in o:
                        print(f"    {k}: {json.dumps(o[k], sort_keys=True)}")
                break
    if args.get:
        return

    new, changes = apply_changes(
        block, wdr=args.wdr, wdr_level=args.wdr_level, blc=args.blc,
        ircut=args.ircut, brightness=args.brightness, contrast=args.contrast,
        saturation=args.saturation, sharpness=args.sharpness, max_gain=args.max_gain)
    if not changes:
        print("[imaging] no changes requested")
        return
    print("[imaging] planned: " + ", ".join(changes))
    if not args.apply:
        print("[imaging] dry-run; pass --apply to send")
        return

    resp = cam.set(img_url, token, new)
    if not _ok(resp):
        raise SystemExit(f"[imaging] SetImagingSettings failed: {_fault(resp)}")
    print("[imaging] applied; read-back:")
    print(json.dumps(parse_settings(cam.get(img_url, token)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
