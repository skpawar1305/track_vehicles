#!/usr/bin/env python3
"""Inspect a camera's ONVIF media profiles, encoder settings and stream URIs.

Use this to explain stream problems (e.g. "the substream looks blurry/blocky on
motion"): it prints each profile's resolution, frame-rate limit, bitrate limit
and H.264 GOP, plus the RTSP URI. A low ``BitrateLimit`` or a long ``GovLength``
is the usual reason a 640x360 substream blurs when something moves.

The PT-NC120D3-WNM(D2) is ONVIF and its RTSP Digest is SHA-256-only, so the
vendor web UI is not always convenient; this talks ONVIF directly.

Credentials come from --password / --user or the TRACKER_CAM_PASSWORD env var,
or from the ``stream_url`` of a config.json (pass --config, or it auto-reads
``./config.json`` / ``/opt/tracker/config.json``) — never hard-coded.

    # from an RTSP URL (derives host/user/password), or:
    python tools/onvif_media.py --host 192.168.225.71 --user admin --password ...
    python tools/onvif_media.py --config /opt/tracker/config.json

Uses WS-UsernameToken (PasswordDigest); the camera's TLS cert is self-signed, so
verification is disabled on purpose.
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


def _mask(url):
    return re.sub(r"//([^:]+):[^@]+@", r"//\1:***@", url)


class OnvifMedia:
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
        x = self._post(media_url, "<trt:GetProfiles/>")
        if "Fault" in x and "GetProfilesResponse" not in x:
            return [], _flat(x)[:200]
        profiles = []
        for block in re.findall(r'<trt:Profiles[^>]*token="([^"]+)"[^>]*>(.*?)</trt:Profiles>', x, re.S):
            token, body = block
            name = re.search(r"<tt:Name>([^<]+)</tt:Name>", body)
            enc = re.search(r"<tt:VideoEncoderConfiguration[^>]*>(.*?)</tt:VideoEncoderConfiguration>",
                            body, re.S)
            info = {"token": token, "name": name.group(1) if name else token}
            if enc:
                e = enc.group(1)
                res = re.search(r"<tt:Width>(\d+)</tt:Width>\s*<tt:Height>(\d+)</tt:Height>", e)
                enc_kind = re.search(r"<tt:Encoding>([^<]+)</tt:Encoding>", e)
                info["encoding"] = enc_kind.group(1) if enc_kind else "?"
                info["resolution"] = (f"{res.group(1)}x{res.group(2)}" if res else "?")
            profiles.append(info)
        return profiles, None

    def get_encoders(self, media_url):
        x = self._post(media_url, "<trt:GetVideoEncoderConfigurations/>")
        if "Fault" in x and "GetVideoEncoderConfigurationsResponse" not in x:
            return [], _flat(x)[:200]
        out = []
        for block in re.findall(r"<trt:Configurations[^>]*>(.*?)</trt:Configurations>", x, re.S):
            tok = re.search(r'<trt:Configurations[^>]*token="([^"]+)"', x)
            def g(tag, s=block):
                m = re.search(rf"<tt:{tag}>([^<]+)</tt:{tag}>", s)
                return m.group(1).strip() if m else None
            res = re.search(r"<tt:Width>(\d+)</tt:Width>\s*<tt:Height>(\d+)</tt:Height>", block)
            out.append({
                "name": g("Name"),
                "encoding": g("Encoding"),
                "resolution": (f"{res.group(1)}x{res.group(2)}" if res else "?"),
                "quality": g("Quality"),
                "fps_limit": g("FrameRateLimit"),
                "bitrate_limit": g("BitrateLimit"),
                "gov_length": g("GovLength"),
                "h264_profile": g("H264Profile"),
            })
        return out, None

    def get_stream_uri(self, media_url, token):
        inner = ("<trt:GetStreamUri>"
                 "<trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>"
                 "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>"
                 f"<trt:ProfileToken>{token}</trt:ProfileToken></trt:GetStreamUri>")
        x = self._post(media_url, inner)
        m = re.search(r"<tt:Uri>([^<]+)</tt:Uri>", x)
        return m.group(1).strip() if m else None


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
    args = ap.parse_args()

    host, user, password = args.host, args.user, args.password
    if not host or not password:
        for path in filter(None, [args.config, "config.json", "/opt/tracker/config.json"]):
            creds = _creds_from_config(path)
            if creds:
                host = host or creds[0]
                user = args.user if args.user != "admin" else creds[1]
                password = password or creds[2]
                print(f"[onvif] credentials from {path}")
                break
    if not host or not password:
        ap.error("provide --host/--password, --config, or set TRACKER_CAM_PASSWORD")

    cam = OnvifMedia(host, user, password)
    media_url = cam.discover_media()
    print(f"[onvif] {host}  media={media_url}")

    profiles, err = cam.get_profiles(media_url)
    if err:
        print("[onvif] GetProfiles:", err)
    encs, err = cam.get_encoders(media_url)
    if err:
        print("[onvif] GetVideoEncoderConfigurations:", err)
    print("[onvif] encoder configs:")
    for e in encs:
        print(f"    {e['name']}: {e['encoding']} {e['resolution']} "
              f"fps<={e['fps_limit']} bitrate<={e['bitrate_limit']}kbps "
              f"GOP={e['gov_length']} quality={e['quality']} {e['h264_profile'] or ''}".rstrip())
    print("[onvif] profiles:")
    for p in profiles:
        uri = cam.get_stream_uri(media_url, p["token"])
        print(f"    {p['name']} ({p['token']}) {p.get('encoding','?')} {p.get('resolution','?')}")
        if uri:
            print(f"        {_mask(uri)}")


if __name__ == "__main__":
    main()
