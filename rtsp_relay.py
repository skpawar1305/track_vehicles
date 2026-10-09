"""RTSP capture shim for cameras whose RTSP Digest uses SHA-256.

OpenCV's bundled FFmpeg negotiates RTSP Digest with MD5 only, so it cannot
authenticate against a SHA-256-only camera (e.g. the PT-NC120D3-WNM(D2), which
challenges with ``algorithm="SHA-256"``). This module performs the RTSP
OPTIONS/DESCRIBE/SETUP/PLAY handshake itself using SHA-256, asks the camera to
send RTP to a loopback UDP port, writes a small SDP that points there, and
keeps the session alive. OpenCV then decodes the SDP like any local stream.

The reader must set ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` to include
``protocol_whitelist;file,udp,rtp`` so FFmpeg may open the ``rtp://`` source
referenced by the SDP.
"""
import hashlib
import os
import re
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit, unquote


class RtspRelay:
    """Bridges a SHA-256 (or MD5) RTSP digest camera to a local UDP SDP."""

    def __init__(self, url, sdp_path="/tmp/tracker_rtsp_relay.sdp",
                 local_port=5004, keepalive=15.0, auth="sha256"):
        u = urlsplit(url)
        self.host = u.hostname
        self.port = u.port or 554
        self.user = unquote(u.username or "")
        self.password = unquote(u.password or "")
        self.base = f"rtsp://{self.host}:{self.port}{u.path}"
        if u.query:
            self.base += "?" + u.query
        self.sdp_path = sdp_path
        self.local_port = local_port
        self.keepalive = keepalive
        self.auth = auth
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._sock = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtsp-relay", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.ready.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def _digest(self, data):
        algo = hashlib.sha256 if self.auth == "sha256" else hashlib.md5
        return algo(data.encode()).hexdigest()

    def _auth(self, uri, params, method):
        nonce = params.get("nonce", "")
        realm = params.get("realm", "")
        ha1 = self._digest(f"{self.user}:{realm}:{self.password}")
        ha2 = self._digest(f"{method}:{uri}")
        response = self._digest(f"{ha1}:{nonce}:{ha2}")
        header = (f'Digest username="{self.user}", realm="{realm}", '
                  f'nonce="{nonce}", uri="{uri}", response="{response}"')
        if params.get("algorithm"):
            header += f', algorithm={params["algorithm"]}'
        if params.get("opaque"):
            header += f', opaque="{params["opaque"]}"'
        return header

    @staticmethod
    def _challenge(response):
        text = response.decode("latin1", "replace")
        m = re.search(r"(?im)^WWW-Authenticate:\s*Digest\s+([^\r\n]+)", text)
        if not m:
            return None
        return {k: v.strip('"') for k, v in
                re.findall(r'(\w+)=("[^"]*"|[^,\s]+)', m.group(1))}

    @staticmethod
    def _read(sock, timeout=6.0):
        sock.settimeout(timeout)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        m = re.search(rb"(?i)Content-Length:\s*(\d+)", head)
        if m:
            need = int(m.group(1))
            while len(rest) < need:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                rest += chunk
            data = head + b"\r\n\r\n" + rest
        return data

    def _request(self, method, uri, cseq, params=None, extra=""):
        header = f"{method} {uri} RTSP/1.0\r\nCSeq: {cseq}\r\nUser-Agent: tracker-relay\r\n"
        if params:
            header += f"Authorization: {self._auth(uri, params, method)}\r\n"
        header += extra + "\r\n"
        self._sock.sendall(header.encode())
        return self._read(self._sock)

    def _establish(self):
        cseq = 1
        self._sock = socket.create_connection((self.host, self.port), timeout=5)
        self._request("OPTIONS", self.base, cseq)
        cseq += 1
        resp = self._request("DESCRIBE", self.base, cseq)
        cseq += 1
        params = self._challenge(resp)
        head = resp.decode("latin1", "replace").splitlines()[0] if resp else ""
        if params is None and "200" not in head:
            raise RuntimeError(f"DESCRIBE failed: {head}")
        if params is not None:
            resp = self._request("DESCRIBE", self.base, cseq, params)
            cseq += 1
        if "200" not in resp.decode("latin1", "replace").splitlines()[0]:
            raise RuntimeError("DESCRIBE unauthorized")
        text = resp.decode("latin1", "replace")
        sdp = text.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in text else ""
        track = ""
        rtpmap, fmtp = [], []
        for line in sdp.splitlines():
            if line.startswith("a=control:") and "trackID" in line:
                c = line.split(":", 1)[1].strip()
                track = c if c.startswith("rtsp://") else f"{self.base.rstrip('/')}/{c}"
            elif line.startswith("a=rtpmap:"):
                rtpmap.append(line)
            elif line.startswith("a=fmtp:"):
                fmtp.append(line)
        if not track:
            raise RuntimeError("no video track in SDP")
        resp = self._request("SETUP", track, cseq, params,
                             extra=(f"Transport: RTP/AVP/UDP;unicast;"
                                    f"client_port={self.local_port}-{self.local_port + 1}\r\n"))
        cseq += 1
        m = re.search(r"(?im)^Session:\s*([^;\r\n]+)", resp.decode("latin1", "replace"))
        session = m.group(1).strip() if m else ""
        resp = self._request("PLAY", self.base, cseq, params,
                             extra=f"Session: {session}\r\n")
        if "200" not in resp.decode("latin1", "replace").splitlines()[0]:
            raise RuntimeError("PLAY failed")
        tmp = self.sdp_path + ".tmp"
        lines = ["v=0", "o=- 0 0 IN IP4 127.0.0.1", "s=tracker-relay",
                 "c=IN IP4 0.0.0.0", "t=0 0",
                 f"m=video {self.local_port} RTP/AVP 96"]
        lines += rtpmap + fmtp
        with open(tmp, "w") as f:
            f.write("\r\n".join(lines) + "\r\n")
        os.replace(tmp, self.sdp_path)
        self.ready.set()
        return params, session, cseq

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                params, session, cseq = self._establish()
                print(f"[relay] streaming -> udp/{self.local_port} "
                      f"(sdp={self.sdp_path})", flush=True)
                backoff = 1.0
                while not self._stop.is_set():
                    if self._stop.wait(self.keepalive):
                        break
                    cseq += 1
                    resp = self._request("GET_PARAMETER", self.base, cseq, params,
                                         extra=f"Session: {session}\r\n")
                    line = resp.decode("latin1", "replace").splitlines()[0] if resp else ""
                    if "200" not in line:
                        raise RuntimeError(f"keepalive: {line}")
            except Exception as e:
                if self._stop.is_set():
                    break
                self.ready.clear()
                print(f"[relay] {e}; reconnecting in {backoff:.0f}s", flush=True)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
        self.ready.clear()
