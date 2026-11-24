"""Run a job script inside Fusion through Fusion's own MCP server (127.0.0.1:27180/mcp).

    python fusion_client.py <job.py> [json-args] [--timeout 600]

The job file is executed from disk by a tiny loader, so large scripts never travel through
the MCP JSON. The loader:
  * guards with a lock file, because fusion_mcp_execute runs every script twice;
  * passes ARGS (dict) and RESULT (path) into the job's namespace and calls job main(ARGS);
  * writes {"run_id", "ok", "result" | "error"} to RESULT.
The client waits for that file (matching run_id), since MCP calls time out long before slow
builds finish. Fusion is shared with other sessions: every job must find its own document by
attribute (see fusion_common.py), never trust app.activeDocument.
"""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

URL = "http://127.0.0.1:27180/mcp"
RUNS = Path(__file__).resolve().parents[2] / ".tmp" / "fusion_runs"  # intermediates, not source

LOADER = '''
import json, os, traceback
RUN_ID = {run_id!r}
LOCK = {lock!r}
RESULT = {result!r}
JOB = {job!r}
ARGS = json.loads({args!r})

def run(_context: str):
    if os.path.exists(LOCK):
        return
    with open(LOCK, "w") as f:
        f.write(RUN_ID)
    out = {{"run_id": RUN_ID}}
    try:
        ns = {{"__name__": "fusion_job", "__file__": JOB}}
        with open(JOB, encoding="utf-8") as f:
            exec(compile(f.read(), JOB, "exec"), ns)
        out["result"] = ns["main"](ARGS)
        out["ok"] = True
    except Exception:
        out["ok"] = False
        out["error"] = traceback.format_exc()
    with open(RESULT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, default=str, indent=1)
    os.replace(RESULT + ".tmp", RESULT)

run("")
'''


def _post(body, sid=None, timeout=60):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if sid:
        h["Mcp-Session-Id"] = sid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        sid = r.headers.get("Mcp-Session-Id") or sid
        txt = r.read().decode("utf-8", "replace")
    out = [json.loads(l[5:].strip()) for l in txt.splitlines() if l.startswith("data:")]
    if not out and txt.strip().startswith("{"):
        out.append(json.loads(txt))
    return sid, out


def run_job(job, args=None, timeout=600):
    """Execute `job` (path) in Fusion with `args` (dict); return the job's result or raise."""
    RUNS.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    lock, result = RUNS / f"{run_id}.lock", RUNS / f"{run_id}.json"
    script = LOADER.format(run_id=run_id, lock=str(lock), result=str(result),
                           job=str(Path(job).resolve()), args=json.dumps(args or {}))
    sid, _ = _post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "fsae-suspension", "version": "1"}}})
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    try:
        _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "fusion_mcp_execute",
                          "arguments": {"featureType": "script", "object": {"script": script}}}}, sid, timeout=120)
    except Exception as e:  # the call often times out while the job keeps running
        print(f"(MCP call returned early: {type(e).__name__}; waiting for the result file)", file=sys.stderr)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if result.exists():
            out = json.loads(result.read_text(encoding="utf-8"))
            if out.get("run_id") == run_id:
                lock.unlink(missing_ok=True)
                if not out.get("ok"):
                    raise RuntimeError("Fusion job failed:\n" + out.get("error", "?"))
                return out.get("result")
        time.sleep(1.0)
    raise TimeoutError(f"no result from Fusion after {timeout} s (run {run_id})")


if __name__ == "__main__":
    a = sys.argv[1:]
    timeout = 600
    if "--timeout" in a:
        i = a.index("--timeout")
        timeout = float(a[i + 1])
        del a[i:i + 2]
    res = run_job(a[0], json.loads(a[1]) if len(a) > 1 else {}, timeout)
    print(json.dumps(res, indent=1, default=str)[:20000])
