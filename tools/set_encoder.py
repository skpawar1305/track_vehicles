#!/usr/bin/env python3
"""Inspect and change a camera's ONVIF video encoder (WS-UsernameToken).

Built for the PT-NC120D3-WNM(D2): it is ONVIF but its RTSP Digest is SHA-256
only, so the vendor UI is inconvenient. This talks ONVIF directly to read the
encoder configuration + options and (with ``--apply``) rewrite resolution,
bitrate, GOP, frame-rate and quality.

The counting substream (``.../Streaming/Channels/102``) is selected by matching
the profile whose ``GetStreamUri`` contains ``102`` (override with
``--profile-token`` / ``--encoder-token``).

Safe by default: with no ``--apply`` it only prints what it *would* change. The
Set request re-sends the camera's own configuration XML with only the targeted
tags substituted, so fields the camera requires (multicast, session timeout,
quality, ...) are preserved rather than dropped.

Credentials come from --password / --user, the TRACKER_CAM_PASSWORD env var, or
a config.json ``stream_url`` (pass --config; it also auto-reads ./config.json
and /opt/tracker/config.json). Never hard-coded.

    # inspect the 102 profile: current res/bitrate/GOP + supported resolutions
    python tools/set_encoder.py --config /opt/tracker/config.json
    # sharpen the substream at lower decode cost: 640x320, 2 Mbps, GOP 25
    python tools/set_encoder.py --config /opt/tracker/config.json \\
        --width 640 --height 320 --bitrate 2048 --gop 25 --apply
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import secrets
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit, unquote

MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"


def _wsse(user, password):
    nonce = secrets.token_bytes(16)
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return (
        '<soap:Header><wsse:Security '
        'xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" '
        'xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
        f'<wsse:UsernameToken><wsse:Username>{user}</wsse:Username>'
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f'{digest}</wsse:Password>'
        '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f'{base64.b64encode(nonce).decode()}</wsse:Nonce>'
        f'<wsu:Created>{created}</wsu:Created></wsse:UsernameToken></wsse:Security></soap:Header>')


def _flat(text):
    return re.sub(r"\s+", " ", text).strip()


def _ok(resp):
    return "Response" in resp and "Fault" not in resp


def _fault(resp):
    m = re.search(r"<[^>]*Reason>\s*<[^>]*Text[^>]*>([^<]+)", resp)
    return m.group(1) if m else _flat(resp)[:200]


class OnvifEncoder:
    def __init__(self, host, user, password, timeout=12.0):
        self.host = host
        self.user = user
        self.password = password
        self.timeout = timeout
        self.device_url = f"https://{host}/onvif/device_service"
        self.ctx = ssl._create_unverified_context()

    def _post(self, url, inner, auth=True):
        header = _wsse(self.user, self.password) if auth else ""
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
            f'xmlns:trt="{MEDIA_NS}" '
            'xmlns:tt="http://www.onvif.org/ver10/schema">'
            f'{header}<soap:Body>{inner}</soap:Body></soap:Envelope>')
        req = urllib.request.Request(
            url, data=body.encode(),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        try:
            return urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout).read().decode()
        except urllib.error.HTTPError as e:
            return e.read().decode()
        except (urllib.error.URLError, OSError) as e:
            return f"<Error>{e}</Error>"

    def discover_media(self):
        x = self._post(self.device_url,
                       "<tds:GetServices><tds:IncludeCapability>false</tds:IncludeCapability></tds:GetServices>")
        for block in re.findall(r"<tds:Service>.*?</tds:Service>", x, re.S):
            if MEDIA_NS in block:
                m = re.search(r"<tds:XAddr>([^<]+)</tds:XAddr>", block)
                if m:
                    return m.group(1).strip()
        return f"https://{self.host}/onvif/media_service"

    def get_profiles(self, media_url):
        """Return [{token,name,encoder_token}] plus any fault text."""
        x = self._post(media_url, "<trt:GetProfiles/>")
        if "Fault" in x and "GetProfilesResponse" not in x:
            return [], _flat(x)[:200]
        out = []
        for m in re.finditer(r'<trt:Profiles[^>]*token="([^"]+)"[^>]*>(.*?)</trt:Profiles>', x, re.S):
            token, body = m.group(1), m.group(2)
            name = re.search(r"<tt:Name>([^<]+)</tt:Name>", body)
            enc = re.search(r'<tt:VideoEncoderConfiguration[^>]*token="([^"]+)"', body)
            out.append({"token": token,
                        "name": name.group(1) if name else token,
                        "encoder_token": enc.group(1) if enc else None})
        return out, None

    def get_stream_uri(self, media_url, profile_token):
        inner = ("<trt:GetStreamUri>"
                 "<trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>"
                 "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>"
                 f"<trt:ProfileToken>{profile_token}</trt:ProfileToken></trt:GetStreamUri>")
        x = self._post(media_url, inner)
        m = re.search(r"<tt:Uri>([^<]+)</tt:Uri>", x)
        return m.group(1).strip() if m else None

    def get_encoder_configs(self, media_url):
        x = self._post(media_url, "<trt:GetVideoEncoderConfigurations/>")
        if "Fault" in x and "GetVideoEncoderConfigurationsResponse" not in x:
            return [], _flat(x)[:200]
        out = []
        for m in re.finditer(r'<trt:Configurations\b[^>]*token="([^"]+)"[^>]*>(.*?)</trt:Configurations>',
                             x, re.S):
            out.append({"token": m.group(1), "block": m.group(2)})
        return out, None

    def get_encoder_config(self, media_url, token):
        inner = ("<trt:GetVideoEncoderConfiguration>"
                 f"<trt:ConfigurationToken>{token}</trt:ConfigurationToken>"
                 "</trt:GetVideoEncoderConfiguration>")
        x = self._post(media_url, inner)
        m = re.search(r"<trt:Configuration\b[^>]*>(.*?)</trt:Configuration>", x, re.S)
        return m.group(1) if m else None

    def get_options(self, media_url, token=None, profile_token=None):
        inner = "<trt:GetVideoEncoderConfigurationOptions>"
        if profile_token:
            inner += f"<trt:ProfileToken>{profile_token}</trt:ProfileToken>"
        if token:
            inner += f"<trt:ConfigurationToken>{token}</trt:ConfigurationToken>"
        inner += "</trt:GetVideoEncoderConfigurationOptions>"
        x = self._post(media_url, inner)
        if "Fault" in x and "OptionsResponse" not in x:
            return None, _flat(x)[:200]
        return x, None

    def set_config(self, media_url, token, block):
        inner = ("<trt:SetVideoEncoderConfiguration>"
                 f'<trt:Configuration token="{token}">{block}</trt:Configuration>'
                 "<trt:ForcePersistence>true</trt:ForcePersistence>"
                 "</trt:SetVideoEncoderConfiguration>")
        return self._post(media_url, inner)


# ── pure helpers (unit-tested by --selftest) ──────────────────────────
def parse_encoder(block):
    """Pull the human-relevant fields out of an encoder config block."""
    def g(tag, s=block):
        m = re.search(rf"<tt:{tag}>([^<]+)</tt:{tag}>", s)
        return m.group(1).strip() if m else None
    res = re.search(r"<tt:Width>(\d+)</tt:Width>\s*<tt:Height>(\d+)</tt:Height>", block)
    return {
        "encoding": g("Encoding"),
        "name": g("Name"),
        "resolution": (f"{res.group(1)}x{res.group(2)}" if res else "?"),
        "quality": g("Quality"),
        "fps_limit": g("FrameRateLimit"),
        "encoding_interval": g("EncodingInterval"),
        "bitrate_limit": g("BitrateLimit"),
        "gov_length": g("GovLength"),
        "h264_profile": g("H264Profile"),
    }


def parse_options(x):
    """Supported resolutions / ranges from a GetVideoEncoderConfigurationOptions."""
    res = [f"{w}x{h}" for w, h in
           re.findall(r"<tt:Width>(\d+)</tt:Width>\s*<tt:Height>(\d+)</tt:Height>", x)]
    def rng(tag):
        m = re.search(rf"<tt:{tag}>\s*<tt:Min>([^<]+)</tt:Min>\s*<tt:Max>([^<]+)</tt:Max>",
                      x, re.S)
        return f"{m.group(1)}..{m.group(2)}" if m else None
    return {"resolutions": sorted(set(res)),
            "bitrate_range": rng("BitrateRange"),
            "fps_range": rng("FrameRateRange"),
            "gov_range": rng("GovLengthRange")}


def apply_changes(block, width=None, height=None, bitrate=None, gop=None,
                  fps=None, quality=None):
    """Return (new_block, changes, missing) with only the given tags replaced.

    Every other field of the camera's own XML is preserved. A tag that is absent
    from the block is reported in ``missing`` (the camera may not expose it) and
    left untouched rather than guessed.
    """
    new = block
    changes, missing = {}, []
    def sub(tag, value, where=block):
        nonlocal new
        pat = rf"<tt:{tag}>[^<]*</tt:{tag}>"
        if re.search(pat, new):
            new = re.sub(pat, f"<tt:{tag}>{value}</tt:{tag}>", new, count=1)
            changes[tag] = value
        else:
            missing.append(tag)
    if width is not None:
        sub("Width", int(width))
    if height is not None:
        sub("Height", int(height))
    if bitrate is not None:
        sub("BitrateLimit", int(bitrate))
    if gop is not None:
        sub("GovLength", int(gop))
    if fps is not None:
        sub("FrameRateLimit", int(fps))
    if quality is not None:
        sub("Quality", int(quality))
    return new, changes, missing


def _selftest():
    sample = (
        "<tt:Name>Sub</tt:Name><tt:UseCount>1</tt:UseCount>"
        "<tt:Encoding>H264</tt:Encoding>"
        "<tt:Resolution><tt:Width>640</tt:Width><tt:Height>480</tt:Height></tt:Resolution>"
        "<tt:Quality>4</tt:Quality>"
        "<tt:RateControl><tt:FrameRateLimit>25</tt:FrameRateLimit>"
        "<tt:EncodingInterval>1</tt:EncodingInterval>"
        "<tt:BitrateLimit>1024</tt:BitrateLimit></tt:RateControl>"
        "<tt:H264><tt:GovLength>50</tt:GovLength>"
        "<tt:H264Profile>Main</tt:H264Profile></tt:H264>"
        "<tt:Multicast><tt:Address><tt:Type>IPv4</tt:Type></tt:Address></tt:Multicast>")
    p = parse_encoder(sample)
    assert p["resolution"] == "640x480", p
    assert p["bitrate_limit"] == "1024" and p["gov_length"] == "50", p
    new, changes, missing = apply_changes(
        sample, width=640, height=320, bitrate=2048, gop=25)
    p2 = parse_encoder(new)
    assert p2["resolution"] == "640x320", p2
    assert p2["bitrate_limit"] == "2048" and p2["gov_length"] == "25", p2
    assert p2["quality"] == "4" and p2["fps_limit"] == "25", p2        # preserved
    assert "<tt:Multicast>" in new, "unrelated fields must survive"
    assert not missing and set(changes) == {"Width", "Height", "BitrateLimit", "GovLength"}, changes
    _, _, missing2 = apply_changes(sample, gop=None, bitrate=None)
    assert missing2 == [], missing2
    # a tag the camera doesn't expose must be reported, not invented
    _, _, missing3 = apply_changes("<tt:Encoding>H264</tt:Encoding>", bitrate=2048)
    assert missing3 == ["BitrateLimit"], missing3
    print("selftest OK")


def _creds_from_config(path, user_default="admin"):
    try:
        with open(path) as f:
            url = json.load(f).get("stream_url", "")
    except (OSError, ValueError):
        return None
    if not url.startswith("rtsp://"):
        return None
    u = urlsplit(url)
    return u.hostname, unquote(u.username or user_default), unquote(u.password or "")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default=os.environ.get("TRACKER_CAM_PASSWORD", ""))
    ap.add_argument("--config", help="config.json to read stream_url from")
    ap.add_argument("--match", default="102",
                    help="substring of the counting profile's stream URI (default 102)")
    ap.add_argument("--profile-token", help="use this media profile explicitly")
    ap.add_argument("--encoder-token", help="use this encoder config token explicitly")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--bitrate", type=int, help="BitrateLimit in kbps")
    ap.add_argument("--gop", type=int, help="GovLength (frames between keyframes)")
    ap.add_argument("--fps", type=int, help="FrameRateLimit")
    ap.add_argument("--quality", type=int)
    ap.add_argument("--apply", action="store_true",
                    help="actually write the change (default: dry-run)")
    ap.add_argument("--selftest", action="store_true", help="run offline assertions and exit")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    host, user, password = args.host, args.user, args.password
    if not host or not password:
        for path in filter(None, [args.config, "config.json", "/opt/tracker/config.json"]):
            creds = _creds_from_config(path)
            if creds:
                host = host or creds[0]
                user = args.user if args.user != "admin" else creds[1]
                password = password or creds[2]
                print(f"[enc] credentials from {path}")
                break
    if not host or not password:
        ap.error("provide --host/--password, --config, or set TRACKER_CAM_PASSWORD")

    cam = OnvifEncoder(host, user, password)
    media = cam.discover_media()
    print(f"[enc] {host}  media={media}")

    # Resolve the target encoder token.
    if args.encoder_token:
        token, profile_token = args.encoder_token, args.profile_token
        print(f"[enc] encoder token: {token} (explicit)")
    elif args.profile_token:
        profiles, err = cam.get_profiles(media)
        if err:
            print("[enc] GetProfiles:", err)
        p = next((p for p in profiles if p["token"] == args.profile_token), None)
        if not p or not p["encoder_token"]:
            ap.error(f"profile {args.profile_token!r} has no encoder config")
        token, profile_token = p["encoder_token"], p["token"]
        print(f"[enc] encoder token: {token} (profile {profile_token})")
    else:
        profiles, err = cam.get_profiles(media)
        if err:
            print("[enc] GetProfiles:", err)
        token = profile_token = None
        for p in profiles:
            uri = cam.get_stream_uri(media, p["token"]) or ""
            if args.match and args.match in uri:
                token, profile_token = p["encoder_token"], p["token"]
                print(f"[enc] encoder token: {token} "
                      f"(profile {p['name']}/{profile_token}, uri matches {args.match!r})")
                break
        if not token:
            listing = "; ".join(f"{p['name']}({p['token']})->{p['encoder_token']}" for p in profiles)
            ap.error(f"no profile matches {args.match!r}; profiles: {listing}")

    # Current config + options.
    block = cam.get_encoder_config(media, token)
    if not block:
        ap.error(f"GetVideoEncoderConfiguration returned nothing for token {token!r}")
    cur = parse_encoder(block)
    print(f"[enc] current: {cur['encoding']} {cur['resolution']} "
          f"bitrate<={cur['bitrate_limit']}kbps GOP={cur['gov_length']} "
          f"fps<={cur['fps_limit']} quality={cur['quality']} {cur['h264_profile'] or ''}".rstrip())

    xopt, err = cam.get_options(media, token=token, profile_token=profile_token)
    if err:
        print("[enc] options:", err)
    else:
        o = parse_options(xopt)
        print(f"[enc] supported resolutions: {', '.join(o['resolutions']) or '?'}")
        print(f"[enc] ranges: bitrate {o['bitrate_range']}  fps {o['fps_range']}  gop {o['gov_range']}")

    if not any([args.width, args.height, args.bitrate, args.gop, args.fps, args.quality]):
        print("[enc] nothing to change (pass --width/--height/--bitrate/--gop/--fps/--quality)")
        return

    new_block, changes, missing = apply_changes(
        block, width=args.width, height=args.height, bitrate=args.bitrate,
        gop=args.gop, fps=args.fps, quality=args.quality)
    if missing:
        print(f"[enc] WARNING: camera config has no {', '.join(missing)} tag(s) — left unchanged")
    after = parse_encoder(new_block)
    print(f"[enc] target: {after['encoding']} {after['resolution']} "
          f"bitrate<={after['bitrate_limit']}kbps GOP={after['gov_length']} "
          f"fps<={after['fps_limit']} quality={after['quality']}")

    if not args.apply:
        print("[enc] dry-run (add --apply to write)")
        return
    resp = cam.set_config(media, token, new_block)
    if not _ok(resp):
        ap.error(f"SetVideoEncoderConfiguration failed: {_fault(resp)}")
    print("[enc] SetVideoEncoderConfiguration ok")
    block2 = cam.get_encoder_config(media, token)
    if block2:
        now = parse_encoder(block2)
        print(f"[enc] now: {now['encoding']} {now['resolution']} "
              f"bitrate<={now['bitrate_limit']}kbps GOP={now['gov_length']} "
              f"fps<={now['fps_limit']} quality={now['quality']}")


if __name__ == "__main__":
    main()
