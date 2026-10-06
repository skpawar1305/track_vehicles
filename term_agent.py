#!/usr/bin/env python3
"""Reverse WebSocket terminal agent for the vehicle tracker.

The Pi sits behind NAT and only opens outbound connections, so this dials the
dashboard and bridges the socket to a local PTY shell. Open
https://tracker.drnanoinc.com/terminal in a browser to use it (the Bun broker
on the VPS pairs the browser with this agent).

Protocol (all frames relayed verbatim by the broker):
  * agent -> browser : binary, raw PTY bytes
  * browser -> agent : text; a leading \\x01 marks a JSON resize control frame,
                       anything else is keystrokes

Env:
  TRACKER_TERM_URL    default wss://tracker.drnanoinc.com/api/term
  TRACKER_TOKEN       bearer token shared with the server (required)
  TRACKER_TERM_SHELL  default /bin/bash
  TRACKER_TERM_COLS   default 80
  TRACKER_TERM_ROWS   default 24
"""
import fcntl
import json
import os
import pty
import select
import subprocess
import sys
import termios
import threading
import time
import struct

import websocket  # websocket-client

CTRL = b"\x01"


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        print(f"[term] FATAL: {name} is not set", file=sys.stderr)
        sys.exit(2)
    return val


class Session:
    """A PTY shell bridged to one WebSocket."""

    def __init__(self, ws, shell, cols, rows):
        self.ws = ws
        self.shell = shell
        self.alive = True
        self.master, slave = pty.openpty()
        argv = [shell, "-l"] if os.path.basename(shell) in ("bash", "sh") else [shell]
        self.proc = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            preexec_fn=os.setsid,
            close_fds=True,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        os.close(slave)
        self.resize(cols, rows)
        print(f"[term] shell pid={self.proc.pid} ({shell})", file=sys.stderr)
        threading.Thread(target=self._pump, daemon=True).start()

    def resize(self, cols, rows):
        try:
            fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                        struct.pack("HHHH", int(rows), int(cols), 0, 0))
        except OSError:
            pass

    def write(self, data):
        if not self.alive:
            return
        try:
            os.write(self.master, data)
        except OSError:
            self.alive = False

    def _pump(self):
        """PTY -> WebSocket until the shell exits or the socket dies."""
        while self.alive:
            try:
                r, _, _ = select.select([self.master], [], [], 0.5)
            except (OSError, ValueError):
                break
            if not r:
                if self.proc.poll() is not None:
                    break
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                break  # EIO once the slave side is gone
            if not data:
                break
            try:
                self.ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                break
        self.alive = False
        try:
            self.ws.close()
        except Exception:
            pass

    def close(self):
        if not self.alive and self.proc.poll() is not None:
            pass
        self.alive = False
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass


def run_once(url, token, shell, cols, rows):
    session = {}

    def on_open(ws):
        try:
            session["s"] = Session(ws, shell, cols, rows)
            ws.send(b"Connected to tracker-pi.\r\n",
                    opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"[term] shell spawn failed: {e}", file=sys.stderr)
            ws.close()

    def on_message(ws, msg):
        s = session.get("s")
        if not s:
            return
        data = msg.encode() if isinstance(msg, str) else msg
        if data[:1] == CTRL:
            try:
                c = json.loads(data[1:].decode())
                s.resize(int(c.get("c", cols)), int(c.get("r", rows)))
            except (ValueError, KeyError):
                pass
            return
        s.write(data)

    def on_close(ws, *a):
        s = session.pop("s", None)
        if s:
            s.close()

    def on_error(ws, err):
        print(f"[term] socket error: {err}", file=sys.stderr)

    app = websocket.WebSocketApp(
        url,
        header={"Authorization": f"Bearer {token}"},
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error,
    )
    app.run_forever(ping_interval=30, ping_timeout=15)


def main():
    url = env("TRACKER_TERM_URL", "wss://tracker.drnanoinc.com/api/term")
    token = env("TRACKER_TOKEN", required=True)
    shell = env("TRACKER_TERM_SHELL", "/bin/bash")
    cols = int(env("TRACKER_TERM_COLS", "80"))
    rows = int(env("TRACKER_TERM_ROWS", "24"))
    url += ("&" if "?" in url else "?") + "role=agent"
    print(f"[term] dialing {url}", file=sys.stderr)

    while True:
        try:
            run_once(url, token, shell, cols, rows)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"[term] {e}", file=sys.stderr)
        print("[term] disconnected; retrying in 3s", file=sys.stderr)
        time.sleep(3)


if __name__ == "__main__":
    main()
