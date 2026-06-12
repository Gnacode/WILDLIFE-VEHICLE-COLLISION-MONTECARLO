#!/usr/bin/env python3
"""
wvc_web.py — local control console for wvc.py
=============================================

A small, dependency-free web interface to the WVC simulation toolkit. It runs a
local HTTP server that:

  * lets you tune every parameter in a form and pick a subcommand;
  * runs `python wvc.py <command> ...` as a subprocess;
  * streams the live terminal output to the page as it happens (server-sent
    events) — a real "command window" that shows the run working;
  * when the run finishes, auto-loads the produced CSV into a sortable table and
    draws charts, and shows any matplotlib PNG figures the run wrote.

It uses only the Python standard library (no Flask) and draws its own charts in
plain canvas (no CDN), so it works fully offline — including on a NAS.

Usage:
    python wvc_web.py                      # serve on http://127.0.0.1:8753
    python wvc_web.py --port 9000
    python wvc_web.py --workdir ./runs     # default working directory for runs
    python wvc_web.py --host 0.0.0.0       # expose on the network (see note)

Place this file next to wvc.py. Open the printed URL in a browser.

Security note: the server executes wvc.py with the options you submit. It binds
to 127.0.0.1 (localhost only) by default. Only use --host 0.0.0.0 on a trusted
network.
"""

from __future__ import annotations
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
WVC = HERE / "wvc.py"

ALLOWED_COMMANDS = {"simulate", "data", "generate-data", "stats", "figures",
                    "ablation", "pipeline"}

DEFAULT_WORKDIR = Path(".").resolve()

JOBS: dict[str, "Job"] = {}
JOBS_LOCK = threading.Lock()


# ============================================================
#  Job: one subprocess run, with a line queue for streaming
# ============================================================

class Job:
    def __init__(self, command: str, args: list[str], workdir: Path):
        self.id = uuid.uuid4().hex[:12]
        self.command = command
        self.args = args
        self.workdir = workdir
        self.cmdline = [sys.executable, "-u", str(WVC),
                        "--workdir", str(workdir), command, *args]
        self.q: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.status = "running"
        self.returncode: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._before = self._snapshot()
        self.new_files: list[str] = []

    def _snapshot(self) -> dict[str, float]:
        snap = {}
        try:
            for p in self.workdir.iterdir():
                if p.is_file():
                    try:
                        snap[p.name] = p.stat().st_mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return snap

    def _diff(self) -> list[str]:
        out = []
        try:
            for p in self.workdir.iterdir():
                if not p.is_file():
                    continue
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if p.name not in self._before or m > self._before[p.name] + 1e-6:
                    if p.suffix.lower() in (".csv", ".png", ".json", ".txt"):
                        out.append(p.name)
        except OSError:
            pass
        # newest first
        out.sort(key=lambda n: (self.workdir / n).stat().st_mtime, reverse=True)
        return out

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # Announce the exact command line first so the terminal shows what ran.
        self.q.put(("line", "$ " + " ".join(
            a if " " not in a else f'"{a}"' for a in self.cmdline)))
        self.q.put(("line", ""))
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            self.proc = subprocess.Popen(
                self.cmdline, cwd=str(self.workdir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=env)
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.q.put(("line", line.rstrip("\n")))
            self.proc.wait()
            self.returncode = self.proc.returncode
            self.status = "done" if self.returncode == 0 else "error"
        except Exception as exc:  # pragma: no cover - defensive
            self.q.put(("line", f"[console] failed to run: {exc}"))
            self.status = "error"
            self.returncode = -1
        finally:
            self.finished = time.time()
            self.new_files = self._diff()
            self.q.put(("end", {
                "status": self.status,
                "returncode": self.returncode,
                "files": self.new_files,
                "elapsed": round((self.finished or time.time()) - self.started, 1),
            }))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            return True
        return False


# ============================================================
#  HTTP handler
# ============================================================

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "wvc-web/1.0"

    # quieter logging
    def log_message(self, fmt, *args):
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # ---- helpers ----
    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"))

    # ---- routing ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path == "/stream":
            self._stream(parse_qs(u.query))
        elif u.path == "/file":
            self._file(parse_qs(u.query))
        elif u.path == "/api/files":
            self._list_files(parse_qs(u.query))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/run":
            self._run()
        elif u.path == "/stop":
            self._stop(parse_qs(u.query))
        else:
            self._json(404, {"error": "not found"})

    # ---- endpoints ----
    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _run(self):
        try:
            body = self._read_body()
        except Exception as exc:
            return self._json(400, {"error": f"bad request body: {exc}"})

        command = body.get("command")
        args = body.get("args", [])
        workdir = body.get("workdir") or str(DEFAULT_WORKDIR)

        if command not in ALLOWED_COMMANDS:
            return self._json(400, {"error": f"unknown command: {command!r}"})
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return self._json(400, {"error": "args must be a list of strings"})

        wd = Path(workdir).expanduser().resolve()
        if not wd.is_dir():
            return self._json(400, {"error": f"working directory not found: {wd}"})

        job = Job(command, args, wd)
        with JOBS_LOCK:
            JOBS[job.id] = job
        job.start()
        self._json(200, {"job_id": job.id, "cmdline": job.cmdline})

    def _stop(self, qs):
        jid = (qs.get("job") or [""])[0]
        job = JOBS.get(jid)
        if not job:
            return self._json(404, {"error": "no such job"})
        self._json(200, {"stopped": job.stop()})

    def _stream(self, qs):
        jid = (qs.get("job") or [""])[0]
        job = JOBS.get(jid)
        if not job:
            return self._json(404, {"error": "no such job"})

        # Close the TCP connection when the stream ends (rather than keep-alive),
        # so clients that read to EOF — curl, simple fetch — see a clean finish.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(kind, data):
            payload = json.dumps({"t": kind, "d": data})
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            while True:
                try:
                    kind, data = job.q.get(timeout=15)
                except queue.Empty:
                    # heartbeat keeps proxies / the connection alive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                emit(kind, data)
                if kind == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away / closed the tab

    def _safe_path(self, wd: Path, name: str) -> Path | None:
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None
        p = (wd / name).resolve()
        try:
            p.relative_to(wd)
        except ValueError:
            return None
        return p if p.is_file() else None

    def _file(self, qs):
        wd = Path((qs.get("workdir") or [str(DEFAULT_WORKDIR)])[0]).expanduser().resolve()
        name = (qs.get("name") or [""])[0]
        p = self._safe_path(wd, name)
        if not p:
            return self._json(404, {"error": "file not found"})
        ext = p.suffix.lower()
        ctype = {".csv": "text/csv; charset=utf-8",
                 ".png": "image/png",
                 ".json": "application/json",
                 ".txt": "text/plain; charset=utf-8"}.get(ext, "application/octet-stream")
        self._send(200, p.read_bytes(), ctype,
                   extra={"Cache-Control": "no-store"})

    def _list_files(self, qs):
        wd = Path((qs.get("workdir") or [str(DEFAULT_WORKDIR)])[0]).expanduser().resolve()
        items = []
        if wd.is_dir():
            for p in wd.iterdir():
                if p.is_file() and p.suffix.lower() in (".csv", ".png", ".json", ".txt"):
                    items.append({"name": p.name,
                                  "size": p.stat().st_size,
                                  "mtime": p.stat().st_mtime})
        items.sort(key=lambda x: x["mtime"], reverse=True)
        self._json(200, {"workdir": str(wd), "files": items})


# ============================================================
#  Frontend (single embedded page)
# ============================================================

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WVC Console</title>
<style>
  /* ---- identity: the three mode colours are the simulator's own (paper figures) ---- */
  :root{
    --bg:#0e1116; --panel:#151a21; --panel-2:#1b212b; --line:#28303c;
    --ink:#e6edf3; --ink-dim:#9aa7b4; --ink-faint:#5d6b79;
    --control:#8d8a82; --detection:#2bb38a; --aware:#e0922e;
    --run:#2bb38a; --stop:#e0573a; --warn:#e0922e;
    --term-bg:#0a0d11; --term-ink:#c6f0dd; --grid:#161c24;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--bg); color:var(--ink); font-family:var(--sans);
    font-size:14px; line-height:1.45;
    display:flex; flex-direction:column; height:100vh; overflow:hidden;
  }
  /* ---- top command bus ---- */
  header{
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line);
  }
  .brand{display:flex; align-items:baseline; gap:10px; margin-right:6px}
  .brand b{font-weight:650; letter-spacing:.02em}
  .brand .sub{color:var(--ink-faint); font-size:11px; font-family:var(--mono);
    text-transform:uppercase; letter-spacing:.14em}
  .bus{display:flex; align-items:center; gap:10px; flex-wrap:wrap; flex:1}
  label.fld{display:flex; flex-direction:column; gap:3px; font-size:11px;
    color:var(--ink-dim); text-transform:uppercase; letter-spacing:.08em}
  input,select{
    background:var(--panel-2); color:var(--ink); border:1px solid var(--line);
    border-radius:6px; padding:6px 8px; font:inherit; font-size:13px;
  }
  input[type=number]{font-family:var(--mono); width:88px}
  input:focus,select:focus{outline:2px solid var(--detection); outline-offset:0}
  .wd{font-family:var(--mono); min-width:220px}
  button{
    font:inherit; font-weight:600; border:1px solid var(--line);
    background:var(--panel-2); color:var(--ink); border-radius:6px;
    padding:7px 16px; cursor:pointer; letter-spacing:.02em;
  }
  button:hover{border-color:var(--ink-faint)}
  button:focus-visible{outline:2px solid var(--detection); outline-offset:2px}
  #run{background:var(--run); border-color:var(--run); color:#06231a}
  #run:hover{filter:brightness(1.08)}
  #run:disabled{opacity:.45; cursor:not-allowed; filter:none}
  #stop{background:transparent; border-color:var(--stop); color:var(--stop)}
  #stop:disabled{opacity:.35; cursor:not-allowed}
  .led{width:9px;height:9px;border-radius:50%;background:var(--ink-faint);
    box-shadow:0 0 0 0 transparent; transition:background .2s}
  .led.run{background:var(--warn); animation:pulse 1.1s infinite}
  .led.ok{background:var(--detection)}
  .led.err{background:var(--stop)}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(224,146,46,.5)}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0 transparent}}
  @media (prefers-reduced-motion: reduce){.led.run{animation:none}}

  /* ---- main split ---- */
  main{display:flex; flex:1; min-height:0}
  aside{
    width:312px; flex:none; background:var(--panel); border-right:1px solid var(--line);
    overflow:auto; padding:14px 16px 28px;
  }
  aside h2{font-size:11px; text-transform:uppercase; letter-spacing:.14em;
    color:var(--ink-faint); margin:4px 0 12px; font-weight:600}
  .params{display:flex; flex-direction:column; gap:14px}
  .p{display:flex; flex-direction:column; gap:5px}
  .p > .row{display:flex; align-items:center; justify-content:space-between; gap:8px}
  .p label.k{font-size:12px; color:var(--ink); font-weight:550}
  .p .help{font-size:11px; color:var(--ink-faint); line-height:1.35}
  .p input[type=number],.p input[type=text],.p select{width:100%; font-size:13px}
  .p input[type=text]{font-family:var(--mono)}
  .checks{display:flex; gap:14px; flex-wrap:wrap}
  .chk{display:flex; align-items:center; gap:6px; font-size:13px; color:var(--ink)}
  .chk input{width:auto}
  .chk.mode-control b{color:var(--control)}
  .chk.mode-detection b{color:var(--detection)}
  .chk.mode-aware b{color:var(--aware)}
  .toggle{display:flex; align-items:center; gap:8px; font-size:13px}
  .toggle input{width:auto}

  /* ---- right column ---- */
  section.work{flex:1; display:flex; flex-direction:column; min-width:0}
  .term-wrap{flex:none; height:42%; display:flex; flex-direction:column;
    border-bottom:1px solid var(--line)}
  .barrow{display:flex; align-items:center; gap:10px; padding:8px 14px;
    background:var(--panel); border-bottom:1px solid var(--line)}
  .barrow .title{font-size:11px; text-transform:uppercase; letter-spacing:.14em;
    color:var(--ink-faint)}
  .barrow .meta{margin-left:auto; font-family:var(--mono); font-size:12px;
    color:var(--ink-dim)}
  #term{
    flex:1; margin:0; overflow:auto; padding:12px 14px;
    background:var(--term-bg); color:var(--term-ink);
    font-family:var(--mono); font-size:12.5px; line-height:1.5;
    white-space:pre-wrap; word-break:break-word;
    background-image:linear-gradient(var(--grid) 1px,transparent 1px);
    background-size:100% 22px; background-position:0 12px;
  }
  #term .err{color:#ff9c83}
  #term .cmd{color:var(--aware)}
  #term .dim{color:var(--ink-faint)}

  .results{flex:1; display:flex; flex-direction:column; min-height:0}
  .tabs{display:flex; gap:2px; padding:0 10px; background:var(--panel);
    border-bottom:1px solid var(--line)}
  .tab{padding:9px 14px; font-size:12px; letter-spacing:.04em; color:var(--ink-dim);
    border:none; background:none; border-bottom:2px solid transparent; cursor:pointer}
  .tab.active{color:var(--ink); border-bottom-color:var(--detection)}
  .tab .count{color:var(--ink-faint); font-family:var(--mono)}
  .panes{flex:1; overflow:auto; position:relative}
  .pane{display:none; padding:16px}
  .pane.active{display:block}
  .empty{color:var(--ink-faint); font-size:13px; padding:24px 4px; max-width:54ch}

  /* charts */
  .chart-card{background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:14px 16px 10px; margin-bottom:16px}
  .chart-card h3{margin:0 0 2px; font-size:13px; font-weight:600}
  .chart-card .cap{margin:0 0 10px; font-size:11px; color:var(--ink-faint)}
  canvas{width:100%; height:auto; display:block}
  .legend{display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; font-size:12px}
  .legend span{display:inline-flex; align-items:center; gap:6px; color:var(--ink-dim)}
  .swatch{width:12px;height:3px;border-radius:2px;display:inline-block}

  /* table */
  .tbl-tools{display:flex; align-items:center; gap:12px; margin-bottom:10px;
    font-size:12px; color:var(--ink-dim)}
  table{border-collapse:collapse; width:100%; font-family:var(--mono); font-size:12px}
  thead th{position:sticky; top:0; background:var(--panel-2); color:var(--ink-dim);
    text-align:right; padding:6px 9px; border-bottom:1px solid var(--line);
    cursor:pointer; white-space:nowrap; font-weight:600}
  thead th:first-child{text-align:left}
  thead th.sorted::after{content:" \25be"; color:var(--detection)}
  thead th.sorted.asc::after{content:" \25b4"}
  tbody td{text-align:right; padding:5px 9px; border-bottom:1px solid var(--grid);
    white-space:nowrap}
  tbody td:first-child{text-align:left}
  tbody tr:hover{background:var(--panel-2)}
  td.mode-control{color:var(--control)} td.mode-detection{color:var(--detection)}
  td.mode-aware{color:var(--aware)}
  .summary{margin-bottom:18px}
  .summary table{font-size:12px}
  .summary caption{text-align:left; color:var(--ink-faint); font-size:11px;
    text-transform:uppercase; letter-spacing:.1em; padding-bottom:7px}

  /* files */
  .files{display:flex; flex-direction:column; gap:6px}
  .frow{display:flex; align-items:center; gap:12px; padding:9px 12px;
    background:var(--panel); border:1px solid var(--line); border-radius:8px}
  .frow .fname{font-family:var(--mono); font-size:13px; cursor:pointer}
  .frow .fname:hover{color:var(--detection)}
  .frow .badge{font-family:var(--mono); font-size:10px; padding:2px 7px; border-radius:20px;
    text-transform:uppercase; letter-spacing:.08em; border:1px solid var(--line); color:var(--ink-dim)}
  .frow .fsize{margin-left:auto; font-family:var(--mono); font-size:12px; color:var(--ink-faint)}
  .frow a{color:var(--ink-dim); font-size:12px; text-decoration:none}
  .frow a:hover{color:var(--ink)}
  .gallery img{max-width:100%; border:1px solid var(--line); border-radius:8px;
    background:#fff; margin-top:12px}
  .fig-title{font-family:var(--mono); font-size:12px; color:var(--ink-dim); margin:14px 0 4px}
</style>
</head>
<body>
<header>
  <div class="brand"><b>WVC&nbsp;Console</b><span class="sub">radar · magnetometer · awareness</span></div>
  <div class="bus">
    <label class="fld">Working directory
      <input id="workdir" class="wd" type="text" spellcheck="false" placeholder="."></label>
    <label class="fld">Command
      <select id="command">
        <option value="simulate">simulate</option>
        <option value="ablation">ablation</option>
        <option value="data">data</option>
        <option value="stats">stats</option>
        <option value="figures">figures</option>
        <option value="pipeline">pipeline</option>
      </select></label>
    <span class="led" id="led" title="idle"></span>
    <button id="run">Run</button>
    <button id="stop" disabled>Stop</button>
  </div>
</header>

<main>
  <aside>
    <h2 id="paramTitle">simulate · parameters</h2>
    <div class="params" id="params"></div>
  </aside>

  <section class="work">
    <div class="term-wrap">
      <div class="barrow">
        <span class="title">Terminal</span>
        <span class="meta" id="termMeta">idle</span>
      </div>
      <pre id="term"><span class="dim">Ready. Tune parameters and press Run.</span></pre>
    </div>

    <div class="results">
      <div class="tabs">
        <button class="tab active" data-pane="charts">Charts</button>
        <button class="tab" data-pane="table">Table</button>
        <button class="tab" data-pane="files">Files <span class="count" id="fileCount"></span></button>
      </div>
      <div class="panes">
        <div class="pane active" id="pane-charts">
          <div class="empty">Charts appear here after a run that produces a results CSV
            (<code>simulate</code> or a sweep). They are drawn from the per-trial data.</div>
        </div>
        <div class="pane" id="pane-table">
          <div class="empty">The results table appears here after a run. Click a column to sort.</div>
        </div>
        <div class="pane" id="pane-files">
          <div class="empty">Files written by the run are listed here. Click a CSV to load it
            into the table and charts; click a PNG to view it.</div>
        </div>
      </div>
    </div>
  </section>
</main>

<script>
//==============================================================
//  Parameter specs — one declarative list per command
//==============================================================
const MODES = ["control","detection","aware"];
const SWEEPS = ["(none)","spacing","range","size","detection_rate","rate","caution","cruise","reaction"];

// field types: int float text bool select modes values stages
const SPECS = {
  simulate: [
    {k:"trials", flag:"--trials", type:"int", def:10, label:"Trials per mode"},
    {k:"hours", flag:"--hours", type:"float", def:2.0, label:"Hours per trial"},
    {k:"modes", flag:"--modes", type:"modes", def:["control","detection","aware"], label:"Modes"},
    {k:"jobs", flag:"--jobs", type:"int", def:1, label:"Worker processes", help:"1 = serial; higher uses more CPU cores"},
    {k:"rate", flag:"--rate", type:"float", def:15, label:"Animal arrival λ (/hr)"},
    {k:"cruise", flag:"--cruise", type:"float", def:100, label:"Cruise speed (km/h)"},
    {k:"caution", flag:"--caution", type:"float", def:30, label:"Caution speed (km/h)"},
    {k:"reaction", flag:"--reaction", type:"float", def:1.5, label:"Driver reaction (s)"},
    {k:"radar_spacing", flag:"--radar-spacing", type:"float", def:15, label:"Radar spacing (m)"},
    {k:"radar_range", flag:"--radar-range", type:"float", def:15, label:"Radar range (m)"},
    {k:"size_scale", flag:"--size-scale", type:"float", def:1.0, label:"Animal size scale"},
    {k:"vehicle_model", flag:"--vehicle-model", type:"select", def:"perfect", options:["perfect","magnetometer"], label:"Vehicle model"},
    {k:"sweep", flag:"--sweep", type:"select", def:"(none)", options:SWEEPS, label:"Sweep parameter", help:"Vary one parameter across values instead of a single point"},
    {k:"values", flag:"--values", type:"values", def:"", label:"Sweep values", help:"Space-separated; overrides the preset list"},
    {k:"csv", flag:"--csv", type:"text", def:"wvc_results.csv", label:"Output CSV"},
    {k:"plot", flag:"--plot", type:"bool", def:true, label:"Also write matplotlib PNG"},
  ],
  ablation: [
    {k:"part", flag:"--part", type:"select", def:"all", options:["all","1","2"], label:"Part", help:"1 = ablation, 2 = geometry sweep"},
    {k:"trials", flag:"--trials", type:"int", def:10, label:"Trials per cell"},
    {k:"hours", flag:"--hours", type:"float", def:2.0, label:"Hours per trial"},
    {k:"jobs", flag:"--jobs", type:"int", def:1, label:"Worker processes"},
  ],
  data: [
    {k:"jobs", flag:"--jobs", type:"int", def:4, label:"Worker processes"},
    {k:"fresh", flag:"--fresh", type:"bool", def:false, label:"Fresh (ignore checkpoint)"},
    {k:"skip_sweeps", flag:"--skip-sweeps", type:"bool", def:false, label:"Headline only"},
    {k:"headline_trials", flag:"--headline-trials", type:"int", def:"", label:"Headline trials"},
    {k:"headline_hours", flag:"--headline-hours", type:"float", def:"", label:"Headline hours"},
    {k:"sweep_trials", flag:"--sweep-trials", type:"int", def:"", label:"Sweep trials"},
    {k:"sweep_hours", flag:"--sweep-hours", type:"float", def:"", label:"Sweep hours"},
  ],
  stats: [
    {k:"bootstrap_n", flag:"--bootstrap-n", type:"int", def:10000, label:"Bootstrap replicates"},
    {k:"seed", flag:"--seed", type:"int", def:42, label:"Bootstrap seed"},
  ],
  figures: [
    {k:"only", flag:"--only", type:"values", def:"", label:"Only figures", help:"e.g. 3 4 S1 S11 (blank = all)"},
    {k:"dpi", flag:"--dpi", type:"int", def:300, label:"DPI"},
  ],
  pipeline: [
    {k:"stages", flag:"--stages", type:"stages", def:["data","stats","figures"], label:"Stages"},
    {k:"jobs", flag:"--jobs", type:"int", def:4, label:"Worker processes"},
    {k:"fresh", flag:"--fresh", type:"bool", def:false, label:"Fresh (ignore checkpoint)"},
    {k:"skip_sweeps", flag:"--skip-sweeps", type:"bool", def:false, label:"Headline only"},
    {k:"headline_trials", flag:"--headline-trials", type:"int", def:"", label:"Headline trials"},
    {k:"headline_hours", flag:"--headline-hours", type:"float", def:"", label:"Headline hours"},
    {k:"sweep_trials", flag:"--sweep-trials", type:"int", def:"", label:"Sweep trials"},
    {k:"sweep_hours", flag:"--sweep-hours", type:"float", def:"", label:"Sweep hours"},
    {k:"bootstrap_n", flag:"--bootstrap-n", type:"int", def:2000, label:"Bootstrap replicates"},
    {k:"only", flag:"--only", type:"values", def:"", label:"Only figures"},
    {k:"dpi", flag:"--dpi", type:"int", def:300, label:"DPI"},
  ],
};
const STAGES = ["data","stats","figures"];

//==============================================================
//  DOM helpers
//==============================================================
const $ = s => document.querySelector(s);
const el = (t,attrs={},...kids)=>{const e=document.createElement(t);
  for(const[k,v]of Object.entries(attrs)){if(k==="class")e.className=v;else if(k==="html")e.innerHTML=v;else e.setAttribute(k,v);}
  kids.flat().forEach(k=>e.appendChild(typeof k==="string"?document.createTextNode(k):k));return e;};

const state = {fields:{}, command:"simulate", job:null, es:null, lastCsv:null};

//==============================================================
//  Build the parameter form for a command
//==============================================================
function renderParams(cmd){
  state.command = cmd;
  state.fields = {};
  $("#paramTitle").textContent = cmd + " · parameters";
  const host = $("#params"); host.innerHTML="";
  for(const f of SPECS[cmd]){
    const wrap = el("div",{class:"p"});
    if(f.type==="bool"){
      const id="f_"+f.k;
      const cb=el("input",{type:"checkbox",id}); cb.checked=!!f.def;
      wrap.appendChild(el("label",{class:"toggle"}, cb, f.label));
      state.fields[f.k]={spec:f, get:()=>cb.checked};
    } else if(f.type==="modes" || f.type==="stages"){
      wrap.appendChild(el("label",{class:"k"}, f.label));
      const list = f.type==="modes"?MODES:STAGES;
      const box = el("div",{class:"checks"});
      const boxes={};
      list.forEach(m=>{
        const cb=el("input",{type:"checkbox"}); cb.checked=f.def.includes(m); boxes[m]=cb;
        const cls = f.type==="modes" ? "chk mode-"+m : "chk";
        box.appendChild(el("label",{class:cls}, cb, el("b",{},m)));
      });
      wrap.appendChild(box);
      state.fields[f.k]={spec:f, get:()=>list.filter(m=>boxes[m].checked)};
    } else if(f.type==="select"){
      wrap.appendChild(el("label",{class:"k"}, f.label));
      const sel=el("select",{});
      f.options.forEach(o=>{const op=el("option",{value:o},o); if(o===f.def)op.selected=true; sel.appendChild(op);});
      wrap.appendChild(sel);
      state.fields[f.k]={spec:f, get:()=>sel.value};
    } else {
      wrap.appendChild(el("label",{class:"k"}, f.label));
      const isNum = (f.type==="int"||f.type==="float");
      const inp=el("input",{type:isNum?"number":"text"});
      if(isNum && f.type==="float") inp.step="any";
      if(f.def!=="" && f.def!=null) inp.value=f.def;
      if(f.type==="text"||f.type==="values") inp.spellcheck=false;
      wrap.appendChild(inp);
      state.fields[f.k]={spec:f, get:()=>inp.value.trim()};
    }
    if(f.help) wrap.appendChild(el("div",{class:"help"}, f.help));
    host.appendChild(wrap);
  }
}

//==============================================================
//  Assemble the wvc.py argument list from the form
//==============================================================
function buildArgs(){
  const args=[];
  for(const f of SPECS[state.command]){
    const v = state.fields[f.k].get();
    if(f.type==="bool"){ if(v) args.push(f.flag); }
    else if(f.type==="modes"||f.type==="stages"){ if(v.length){ args.push(f.flag, ...v); } }
    else if(f.type==="values"){ if(v){ args.push(f.flag, ...v.split(/\s+/)); } }
    else if(f.type==="select"){ if(v && v!=="(none)") args.push(f.flag, v); }
    else { if(v!=="" && v!=null) args.push(f.flag, String(v)); }
  }
  return args;
}

// expected primary CSV name, so we can auto-load it after the run
function expectedCsv(){
  if(state.command!=="simulate") return null;
  const csv = state.fields.csv.get() || "wvc_results.csv";
  const sweep = state.fields.sweep.get();
  if(sweep && sweep!=="(none)") return csv.replace(/\.csv$/, "_sweep_"+sweep+".csv");
  return csv;
}

//==============================================================
//  Run / stream
//==============================================================
const term = $("#term");
function termClear(){ term.innerHTML=""; }
function termLine(text, cls){
  const span=el("span", cls?{class:cls}:{});
  span.textContent = text + "\n";
  term.appendChild(span);
  term.scrollTop = term.scrollHeight;
}
function setLed(s){ const l=$("#led"); l.className="led"+(s?" "+s:""); l.title=s||"idle"; }

async function run(){
  const command = $("#command").value;
  const args = buildArgs();
  const workdir = $("#workdir").value.trim() || ".";
  $("#run").disabled=true; $("#stop").disabled=false; setLed("run");
  termClear();
  $("#termMeta").textContent = "starting…";
  let res;
  try{
    res = await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({command,args,workdir})}).then(r=>r.json());
  }catch(e){ termLine("[console] cannot reach server: "+e,"err"); finish("error"); return; }
  if(res.error){ termLine("[console] "+res.error,"err"); finish("error"); return; }
  state.job = res.job_id;
  $("#termMeta").textContent = "running · "+command;

  const es = new EventSource("/stream?job="+res.job_id);
  state.es = es;
  es.onmessage = (ev)=>{
    const m = JSON.parse(ev.data);
    if(m.t==="line"){
      const line=m.d;
      let cls=null;
      if(line.startsWith("$ ")) cls="cmd";
      else if(/error|fatal|traceback|exception/i.test(line)) cls="err";
      termLine(line, cls);
    } else if(m.t==="end"){
      es.close(); state.es=null;
      onDone(m.d, workdir);
    }
  };
  es.onerror = ()=>{ /* stream closed; end event handles status */ };
}

function finish(status){
  $("#run").disabled=false; $("#stop").disabled=true;
  setLed(status==="done"?"ok":status==="error"?"err":"");
}

async function onDone(info, workdir){
  finish(info.status);
  $("#termMeta").textContent =
    (info.status==="done"?"done":"exited "+info.returncode)+" · "+info.elapsed+"s";
  termLine("", null);
  termLine("— "+(info.status==="done"?"completed":"failed")+" in "+info.elapsed+"s (exit "+info.returncode+") —",
           info.status==="done"?"dim":"err");

  // Files tab
  renderFiles(info.files||[], workdir);

  // Auto-load primary CSV (simulate) or, for other commands, the newest CSV produced
  let csv = expectedCsv();
  if(!(info.files||[]).includes(csv)){
    csv = (info.files||[]).find(f=>f.endsWith(".csv")) || null;
  }
  if(csv){ await loadCsv(csv, workdir); }

  // Auto-show first PNG if present
  const png = (info.files||[]).find(f=>f.endsWith(".png"));
  if(png){ showFigures((info.files||[]).filter(f=>f.endsWith(".png")), workdir); }
}

async function stop(){
  if(!state.job) return;
  await fetch("/stop?job="+state.job,{method:"POST"});
  termLine("[console] stop requested","dim");
}

//==============================================================
//  CSV parsing + table + charts
//==============================================================
function parseCsv(text){
  const lines = text.replace(/\r/g,"").split("\n").filter(l=>l.length);
  if(!lines.length) return {cols:[], rows:[]};
  const split = l=>{ // minimal CSV: handles simple quoted fields
    const out=[]; let cur="",q=false;
    for(let i=0;i<l.length;i++){const c=l[i];
      if(q){ if(c==='"'){ if(l[i+1]==='"'){cur+='"';i++;} else q=false;} else cur+=c;}
      else { if(c==='"')q=true; else if(c===","){out.push(cur);cur="";} else cur+=c; }
    } out.push(cur); return out;
  };
  const cols = split(lines[0]);
  const rows = lines.slice(1).map(l=>{const v=split(l); const o={};
    cols.forEach((c,i)=>{const x=v[i]; const n=Number(x); o[c]=(x!==""&&!isNaN(n))?n:x;}); return o;});
  return {cols, rows};
}

async function loadCsv(name, workdir){
  state.lastCsv = {name, workdir};
  let text;
  try{ text = await fetch("/file?name="+encodeURIComponent(name)+"&workdir="+encodeURIComponent(workdir)).then(r=>r.text()); }
  catch(e){ return; }
  const data = parseCsv(text);
  renderTable(name, data);
  renderCharts(name, data);
}

// detect sweep column: first col is not "mode" but second is
function sweepColumn(cols){
  return (cols[0] && cols[0]!=="mode" && cols[1]==="mode") ? cols[0] : null;
}

const MODE_COLOR = {control:"#8d8a82", detection:"#2bb38a", aware:"#e0922e"};

function renderTable(name, data){
  const pane=$("#pane-table"); pane.innerHTML="";
  if(!data.rows.length){ pane.innerHTML='<div class="empty">No rows in '+name+'.</div>'; return; }
  pane.appendChild(buildSummary(data));
  pane.appendChild(el("div",{class:"tbl-tools"},
    el("span",{},name+" · "+data.rows.length+" rows · "+data.cols.length+" columns"),
    el("span",{html:'click a header to sort'})));
  const tbl=el("table"); const thead=el("thead"); const trh=el("tr");
  data.cols.forEach(c=>trh.appendChild(el("th",{"data-col":c}, c)));
  thead.appendChild(trh); tbl.appendChild(thead);
  const tbody=el("tbody");
  const fmt = v => typeof v==="number" ? (Number.isInteger(v)?v:(+v).toFixed(4)) : v;
  let sortCol=null, asc=false;
  const draw = rows=>{
    tbody.innerHTML="";
    rows.forEach(r=>{const tr=el("tr");
      data.cols.forEach(c=>{const td=el("td",{}, String(fmt(r[c])));
        if(c==="mode" && MODE_COLOR[r[c]]) td.className="mode-"+r[c]; tr.appendChild(td);});
      tbody.appendChild(tr);});
  };
  draw(data.rows);
  trh.querySelectorAll("th").forEach(th=>{
    th.onclick=()=>{const c=th.dataset.col; asc = sortCol===c ? !asc : true; sortCol=c;
      const rows=[...data.rows].sort((a,b)=>{const x=a[c],y=b[c];
        return (x<y?-1:x>y?1:0)*(asc?1:-1);});
      trh.querySelectorAll("th").forEach(h=>h.className="");
      th.className="sorted"+(asc?" asc":""); draw(rows);};
  });
  tbl.appendChild(tbody); pane.appendChild(tbl);
}

// per-mode mean ± sd of the headline metrics
function buildSummary(data){
  const sweep = sweepColumn(data.cols);
  const wrap=el("div",{class:"summary"});
  const metrics=[["col_rate","Collision rate %",100],["det_rate","Detection rate %",100],
                 ["collisions","Collisions/trial",1],["road_entries","Road entries",1]]
                 .filter(m=>data.cols.includes(m[0]));
  const modes=[...new Set(data.rows.map(r=>r.mode))];
  const cap = sweep ? "Means across all sweep points (per mode)" : "Per-mode summary (mean ± σ)";
  const tbl=el("table"); tbl.appendChild(el("caption",{},cap));
  const head=el("tr"); head.appendChild(el("th",{},"mode"));
  metrics.forEach(m=>head.appendChild(el("th",{}, m[1])));
  const thead=el("thead"); thead.appendChild(head); tbl.appendChild(thead);
  const body=el("tbody");
  modes.forEach(mo=>{
    const tr=el("tr"); const td0=el("td",{}, mo); if(MODE_COLOR[mo])td0.className="mode-"+mo; tr.appendChild(td0);
    metrics.forEach(m=>{
      const vals=data.rows.filter(r=>r.mode===mo).map(r=>+r[m[0]]*m[2]).filter(v=>!isNaN(v));
      const mean=vals.reduce((a,b)=>a+b,0)/(vals.length||1);
      const sd=Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(vals.length-1,1));
      tr.appendChild(el("td",{}, mean.toFixed(2)+" ± "+sd.toFixed(2)));
    });
    body.appendChild(tr);
  });
  tbl.appendChild(body); wrap.appendChild(tbl); return wrap;
}

//==============================================================
//  Charts (plain canvas, no libraries)
//==============================================================
function renderCharts(name, data){
  const pane=$("#pane-charts"); pane.innerHTML="";
  if(!data.rows.length){ pane.innerHTML='<div class="empty">No data to plot.</div>'; return; }
  const sweep = sweepColumn(data.cols);
  if(sweep && data.cols.includes("col_rate")){
    pane.appendChild(sweepChart(data, sweep, "col_rate", "Collision rate vs "+sweep,
      "Mean collision rate per road entry (%), ±1 SEM band, by mode."));
  } else if(data.cols.includes("col_rate")){
    pane.appendChild(barChart(data, "col_rate", "Collision rate by mode",
      "Mean collision rate per road entry (%), error bars ±1 σ across trials."));
    if(data.cols.includes("det_rate"))
      pane.appendChild(barChart(data, "det_rate", "Detection rate by mode",
        "Mean fraction of animals detected (%), error bars ±1 σ across trials."));
  } else {
    pane.appendChild(el("div",{class:"empty"},
      "No collision-rate column in "+name+" — nothing to chart. See the Table and Files tabs."));
  }
}

function modeStats(data, metric){
  const modes=[...new Set(data.rows.map(r=>r.mode))]
    .sort((a,b)=>MODES.indexOf(a)-MODES.indexOf(b));
  return modes.map(mo=>{
    const v=data.rows.filter(r=>r.mode===mo).map(r=>+r[metric]*100).filter(x=>!isNaN(x));
    const mean=v.reduce((a,b)=>a+b,0)/(v.length||1);
    const sd=Math.sqrt(v.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(v.length-1,1));
    return {mode:mo, mean, sd};
  });
}

function hidpiCanvas(w,h){
  const c=el("canvas"); const dpr=window.devicePixelRatio||1;
  c.width=w*dpr; c.height=h*dpr; c.style.width="100%"; c.style.maxWidth=w+"px"; c.style.height="auto";
  const ctx=c.getContext("2d"); ctx.scale(dpr,dpr); return {c,ctx,w,h};
}
function card(title,cap,canvas,legendItems){
  const cd=el("div",{class:"chart-card"}, el("h3",{},title), el("p",{class:"cap"},cap), canvas);
  if(legendItems){const lg=el("div",{class:"legend"});
    legendItems.forEach(([lab,col])=>lg.appendChild(el("span",{},
      el("i",{class:"swatch"}), lab)));
    lg.querySelectorAll(".swatch").forEach((s,i)=>s.style.background=legendItems[i][1]);
    cd.appendChild(lg);}
  return cd;
}

function barChart(data, metric, title, cap){
  const stats=modeStats(data, metric);
  const W=560,H=300, pad={l:54,r:18,t:18,b:46};
  const {c,ctx,w,h}=hidpiCanvas(W,H);
  const max=Math.max(0.001,...stats.map(s=>s.mean+s.sd))*1.15;
  const x0=pad.l, x1=w-pad.r, y0=h-pad.b, y1=pad.t;
  const sy=v=>y0-(v/max)*(y0-y1);
  // grid + y axis
  ctx.strokeStyle="#28303c"; ctx.fillStyle="#9aa7b4"; ctx.lineWidth=1;
  ctx.font="11px ui-monospace,Menlo,monospace"; ctx.textBaseline="middle";
  const ticks=5;
  for(let i=0;i<=ticks;i++){const v=max*i/ticks, y=sy(v);
    ctx.globalAlpha=.5; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); ctx.globalAlpha=1;
    ctx.textAlign="right"; ctx.fillText(v.toFixed(1), x0-8, y);}
  const n=stats.length, bw=(x1-x0)/n*0.5;
  ctx.textAlign="center"; ctx.textBaseline="top";
  stats.forEach((s,i)=>{
    const cx=x0+(x1-x0)*(i+0.5)/n;
    const col=MODE_COLOR[s.mode]||"#6c7a89";
    ctx.fillStyle=col; ctx.fillRect(cx-bw/2, sy(s.mean), bw, y0-sy(s.mean));
    // error bar ±sd
    if(s.sd>0){ctx.strokeStyle="#e6edf3"; ctx.lineWidth=1.2;
      const yt=sy(s.mean+s.sd), yb=sy(Math.max(0,s.mean-s.sd));
      ctx.beginPath(); ctx.moveTo(cx,yt); ctx.lineTo(cx,yb);
      ctx.moveTo(cx-5,yt); ctx.lineTo(cx+5,yt);
      ctx.moveTo(cx-5,yb); ctx.lineTo(cx+5,yb); ctx.stroke();}
    ctx.fillStyle="#e6edf3"; ctx.fillText(s.mean.toFixed(1), cx, sy(s.mean)-16<y1?y1:sy(s.mean)-16);
    ctx.fillStyle="#9aa7b4"; ctx.fillText(s.mode, cx, y0+8);
  });
  // axis line
  ctx.strokeStyle="#28303c"; ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y0); ctx.stroke();
  return card(title,cap,c);
}

function sweepChart(data, sweep, metric, title, cap){
  const W=620,H=320, pad={l:56,r:18,t:18,b:46};
  const {c,ctx,w,h}=hidpiCanvas(W,H);
  const xs=[...new Set(data.rows.map(r=>+r[sweep]))].sort((a,b)=>a-b);
  const modes=[...new Set(data.rows.map(r=>r.mode))].sort((a,b)=>MODES.indexOf(a)-MODES.indexOf(b));
  const series=modes.map(mo=>({mode:mo, pts:xs.map(xv=>{
    const v=data.rows.filter(r=>r.mode===mo && +r[sweep]===xv).map(r=>+r[metric]*100).filter(x=>!isNaN(x));
    const mean=v.reduce((a,b)=>a+b,0)/(v.length||1);
    const sem=Math.sqrt(v.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(v.length-1,1))/Math.sqrt(v.length||1);
    return {x:xv, mean, sem};
  })}));
  const allY=series.flatMap(s=>s.pts.map(p=>p.mean+p.sem));
  const max=Math.max(0.001,...allY)*1.15;
  const x0=pad.l,x1=w-pad.r,y0=h-pad.b,y1=pad.t;
  const xmin=xs[0],xmax=xs[xs.length-1];
  const sx=v=> xmax===xmin? (x0+x1)/2 : x0+(v-xmin)/(xmax-xmin)*(x1-x0);
  const sy=v=>y0-(v/max)*(y0-y1);
  ctx.font="11px ui-monospace,Menlo,monospace";
  // grid + y ticks
  ctx.strokeStyle="#28303c"; ctx.fillStyle="#9aa7b4"; ctx.lineWidth=1; ctx.textBaseline="middle";
  for(let i=0;i<=5;i++){const v=max*i/5,y=sy(v);
    ctx.globalAlpha=.5; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); ctx.globalAlpha=1;
    ctx.textAlign="right"; ctx.fillText(v.toFixed(1), x0-8, y);}
  // x ticks
  ctx.textAlign="center"; ctx.textBaseline="top";
  xs.forEach(xv=>ctx.fillText(String(xv), sx(xv), y0+8));
  // series
  series.forEach(s=>{
    const col=MODE_COLOR[s.mode]||"#6c7a89";
    // SEM band
    ctx.fillStyle=col; ctx.globalAlpha=.16; ctx.beginPath();
    s.pts.forEach((p,i)=>{const X=sx(p.x),Y=sy(p.mean+p.sem); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    [...s.pts].reverse().forEach(p=>{ctx.lineTo(sx(p.x),sy(Math.max(0,p.mean-p.sem)));});
    ctx.closePath(); ctx.fill(); ctx.globalAlpha=1;
    // line
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
    s.pts.forEach((p,i)=>{const X=sx(p.x),Y=sy(p.mean); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}); ctx.stroke();
    // markers
    ctx.fillStyle=col; s.pts.forEach(p=>{ctx.beginPath();ctx.arc(sx(p.x),sy(p.mean),3,0,7);ctx.fill();});
  });
  ctx.strokeStyle="#28303c"; ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y0); ctx.stroke();
  ctx.fillStyle="#5d6b79"; ctx.textAlign="center"; ctx.fillText(sweep, (x0+x1)/2, y0+24);
  return card(title,cap,c, modes.map(m=>[m, MODE_COLOR[m]||"#6c7a89"]));
}

//==============================================================
//  Files
//==============================================================
function badge(ext){return ext.replace(".","");}
function renderFiles(files, workdir){
  const pane=$("#pane-files"); pane.innerHTML="";
  $("#fileCount").textContent = files.length?("("+files.length+")"):"";
  if(!files.length){ pane.innerHTML='<div class="empty">This run wrote no new files.</div>'; return; }
  const list=el("div",{class:"files"});
  files.forEach(f=>{
    const ext=f.slice(f.lastIndexOf("."));
    const url="/file?name="+encodeURIComponent(f)+"&workdir="+encodeURIComponent(workdir);
    const row=el("div",{class:"frow"});
    row.appendChild(el("span",{class:"badge"}, badge(ext)));
    const nm=el("span",{class:"fname"}, f);
    nm.onclick=()=>{ if(ext===".csv"){ setTab("table"); loadCsv(f, workdir); }
      else if(ext===".png"){ setTab("charts"); showFigures([f], workdir, true); }
      else window.open(url,"_blank"); };
    row.appendChild(nm);
    row.appendChild(el("a",{href:url,target:"_blank"},"open"));
    row.appendChild(el("a",{href:url,download:f},"download"));
    list.appendChild(row);
  });
  pane.appendChild(list);
}

function showFigures(pngs, workdir, replace){
  const pane=$("#pane-charts");
  let gal = pane.querySelector(".gallery");
  if(replace || !gal){ /* keep canvas charts above; add a gallery below */ }
  gal = el("div",{class:"gallery"});
  gal.appendChild(el("div",{class:"fig-title"},"matplotlib figures"));
  pngs.forEach(f=>{
    const url="/file?name="+encodeURIComponent(f)+"&workdir="+encodeURIComponent(workdir)+"&_t="+Date.now();
    gal.appendChild(el("div",{class:"fig-title"}, f));
    gal.appendChild(el("img",{src:url, alt:f}));
  });
  // remove previous gallery, append fresh
  const old=pane.querySelector(".gallery"); if(old) old.remove();
  pane.appendChild(gal);
}

//==============================================================
//  Tabs
//==============================================================
function setTab(name){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.pane===name));
  document.querySelectorAll(".pane").forEach(p=>p.classList.toggle("active",p.id==="pane-"+name));
}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>setTab(t.dataset.pane));

//==============================================================
//  Wire up
//==============================================================
$("#command").onchange = e=>renderParams(e.target.value);
$("#run").onclick = run;
$("#stop").onclick = stop;
document.addEventListener("keydown", e=>{
  if((e.metaKey||e.ctrlKey) && e.key==="Enter" && !$("#run").disabled){ run(); }
});
renderParams("simulate");
</script>
</body>
</html>
"""


# ============================================================
#  Server entry
# ============================================================

def main():
    global DEFAULT_WORKDIR
    ap = argparse.ArgumentParser(description="Local web console for wvc.py")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to expose)")
    ap.add_argument("--port", type=int, default=8753, help="port (default 8753)")
    ap.add_argument("--workdir", default=".",
                    help="default working directory for runs (default: current dir)")
    args = ap.parse_args()

    if not WVC.exists():
        sys.exit(f"error: wvc.py not found next to this script (looked in {HERE}). "
                 f"Place wvc_web.py in the same folder as wvc.py.")

    DEFAULT_WORKDIR = Path(args.workdir).expanduser().resolve()
    DEFAULT_WORKDIR.mkdir(parents=True, exist_ok=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host=='0.0.0.0' else args.host}:{args.port}"
    print(f"WVC Console serving at {url}")
    print(f"  wvc.py:   {WVC}")
    print(f"  workdir:  {DEFAULT_WORKDIR}")
    if args.host == "0.0.0.0":
        print("  (bound to 0.0.0.0 — reachable on your network; only do this on a trusted LAN)")
    print("  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()