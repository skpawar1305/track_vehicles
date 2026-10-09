#!/usr/bin/env python3
"""Read or set a camera's clock over ONVIF (Manual or NTP).

The PT-NC120D3-WNM(D2) is ONVIF and its RTSP Digest is SHA-256-only, so it is
easiest to manage its clock here rather than through a vendor web UI. It ships
with a *Manual* clock that can be badly off (the unit we had was 5h30m behind:
local time mistaken for UTC). Prefer NTP so it self-corrects.

Credentials come from --password or the TRACKER_CAM_PASSWORD environment
variable, never hard-coded.

    # show the current clock
    python tools/set_camera_time.py --host 192.168.225.71 --user admin
    # set to this host's UTC, Manual
    python tools/set_camera_time.py --host 192.168.225.71 --user admin --set-manual
    # switch to NTP (self-correcting)
    python tools/set_camera_time.py --host 192.168.225.71 --user admin --set-ntp pool.ntp.org

Uses WS-UsernameToken (PasswordDigest) auth; the camera's TLS cert is
self-signed, so verification is disabled on purpose.
"""
import argparse
import base64
import datetime
import hashlib
import os
import re
import secrets
import ssl
import urllib.error
import urllib.request


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


class OnvifClock:
    def __init__(self, host, user, password, timeout=12.0):
        self.user = user
        self.password = password
        self.timeout = timeout
        self.url = f"https://{host}/onvif/device_service"
        self.ctx = ssl._create_unverified_context()

    def _post(self, inner, auth=True):
        header = _wsse(self.user, self.password) if auth else ""
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
            'xmlns:tt="http://www.onvif.org/ver10/schema">'
            f'{header}<soap:Body>{inner}</soap:Body></soap:Envelope>')
        req = urllib.request.Request(
            self.url, data=body.encode(),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        try:
            return urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout).read().decode()
        except urllib.error.HTTPError as e:
            return e.read().decode()

    def get_time(self):
        x = self._post("<tds:GetSystemDateAndTime/>", auth=False)

        def g(tag):
            m = re.search(rf"<tt:{tag}>(\d+)</tt:{tag}>", x)
            if not m:
                raise RuntimeError("unexpected GetSystemDateAndTime response: " + _flat(x)[:200])
            return int(m.group(1))

        utc = datetime.datetime(g("Year"), g("Month"), g("Day"),
                                g("Hour"), g("Minute"), g("Second"),
                                tzinfo=datetime.timezone.utc)
        mode = re.search(r"<tt:DateTimeType>([^<]+)", x)
        tz = re.search(r"<tt:TZ>([^<]+)</tt:TZ>", x)
        return utc, (mode.group(1) if mode else "?"), (tz.group(1) if tz else "?")

    def get_ntp(self):
        x = self._post("<tds:GetNTP/>")
        dns = re.search(r"<tt:DNSname>([^<]+)</tt:DNSname>", x)
        ip = re.search(r"<tt:IPv4Address>([^<]+)</tt:IPv4Address>", x)
        dhcp = re.search(r"<tt:FromDHCP>(true|false)</tt:FromDHCP>", x)
        return {"from_dhcp": dhcp.group(1) if dhcp else "?",
                "server": (dns or ip).group(1) if (dns or ip) else None}

    def set_ntp_server(self, server):
        is_ip = bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", server))
        kind = "IPv4" if is_ip else "DNS"
        field = "IPv4Address" if is_ip else "DNSname"
        inner = ("<tds:SetNTP><tds:FromDHCP>false</tds:FromDHCP><tds:NTPManual>"
                 f"<tt:Type>{kind}</tt:Type><tt:{field}>{server}</tt:{field}>"
                 "</tds:NTPManual></tds:SetNTP>")
        return self._post(inner)

    def set_datetime(self, datetime_type, tz):
        n = datetime.datetime.now(datetime.timezone.utc)
        inner = (
            "<tds:SetSystemDateAndTime>"
            f"<tds:DateTimeType>{datetime_type}</tds:DateTimeType>"
            "<tds:DaylightSavings>false</tds:DaylightSavings>"
            f"<tds:TimeZone><tt:TZ>{tz}</tt:TZ></tds:TimeZone>"
            "<tds:UTCDateTime><tt:Time>"
            f"<tt:Hour>{n.hour}</tt:Hour><tt:Minute>{n.minute}</tt:Minute><tt:Second>{n.second}</tt:Second>"
            "</tt:Time><tt:Date>"
            f"<tt:Year>{n.year}</tt:Year><tt:Month>{n.month}</tt:Month><tt:Day>{n.day}</tt:Day>"
            "</tt:Date></tds:UTCDateTime></tds:SetSystemDateAndTime>")
        return self._post(inner)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="camera IP/hostname")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default=os.environ.get("TRACKER_CAM_PASSWORD", ""))
    ap.add_argument("--tz", default="IST-5:30:00", help="ONVIF/POSIX timezone string")
    ap.add_argument("--set-manual", action="store_true",
                    help="set the clock to this host's UTC (Manual mode)")
    ap.add_argument("--set-ntp", metavar="SERVER",
                    help="switch to NTP mode and set the server (DNS name or IPv4)")
    args = ap.parse_args()

    if not args.password:
        ap.error("provide --password or set TRACKER_CAM_PASSWORD")

    cam = OnvifClock(args.host, args.user, args.password)
    utc, mode, tz = cam.get_time()
    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"[cam] {args.host} mode={mode} tz={tz} utc={utc:%Y-%m-%d %H:%M:%S}")
    print(f"[cam] host utc={now:%Y-%m-%d %H:%M:%S} drift={int((now - utc).total_seconds())}s")

    if args.set_ntp:
        r = cam.set_ntp_server(args.set_ntp)
        print("[cam] SetNTP", "ok" if _ok(r) else _flat(r)[:200])
        r = cam.set_datetime("NTP", args.tz)
        print("[cam] SetSystemDateAndTime(NTP)", "ok" if _ok(r) else _flat(r)[:200])
        print("[cam] GetNTP", cam.get_ntp())
    elif args.set_manual:
        r = cam.set_datetime("Manual", args.tz)
        print("[cam] SetSystemDateAndTime(Manual)", "ok" if _ok(r) else _flat(r)[:200])

    if args.set_manual or args.set_ntp:
        utc, mode, tz = cam.get_time()
        print(f"[cam] now mode={mode} tz={tz} utc={utc:%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
