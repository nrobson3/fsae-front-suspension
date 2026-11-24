"""Requirements traceability matrix, generated from data/requirements.json and the analysis outputs.

Every value and status is read from a generated file (kinematics_summary.csv, vehicle_derived.json,
member_checks.csv, joint_checks.csv, misalignment.csv, interference.csv, mass_budget.csv,
upright_loads.json, bearings.json, bolt_checks.csv, upright_fea.json, upright_fatigue.json, fea_benchmarks.json,
fastener_checks.csv, handcalc_results.json), never typed by hand, so the matrix
cannot drift from the analysis. Every source file must carry the current hardpoint version, and every output
run_all writes must match its stamp (stamps.stale: the content of each input it was made from, data and code, and
of the output itself, back through the outputs it was made from, including the upright STEP); the CAD meshes must
pass common.load_mesh_index (their inputs, the upright STEP they were exported with, each file's hash). Otherwise
the matrix is not built. The upright FEA (an hour's run outside run_all) must also match the current fingerprint
of its inputs (fingerprint.fea_inputs_hash): if the STEP, the loads or the design data changed since, its items
read STALE instead of PASS or FAIL.

Statuses: PASS / FAIL / OPEN (verified in a later week) / DEVIATION (a FAIL accepted with a
recorded reason, see notes/requirements.md) / ACTION (a FAIL with a planned fix) / STALE (the FEA
result no longer matches its inputs: re-run scripts/fea/upright_fea.py) / RETIRED (the thing the requirement
was about no longer exists; the entry names the change record, "RC-nn: reason").
Run: python requirements_matrix.py   Writes notes/requirements_matrix.md
"""
import csv
import json
import math
import re

import fingerprint
import stamps
from common import DATA, NOTES, load_mesh_index, read_version


def _rows(path):
    with open(path, newline="") as f:
        lines = list(f)
    m = re.match(r"#\s*hardpoints[\s,]+(v[\d.]+)", lines[0]) if lines else None
    stamp = m.group(1) if m else None
    return stamp, list(csv.DictReader(l for l in lines if not l.startswith("#")))


def _cad_stamp(path):
    """The CAD fingerprint a CAD-derived output was computed from (its header's 'cad <hash>')."""
    with open(path) as f:
        m = re.search(r"cad ([0-9a-f]{16})", f.readline())
    return m.group(1) if m else None


def _matches(recorded, fingerprint_fn):
    """True if a recorded fingerprint equals today's; False if today's cannot be taken (a STEP with no DATA section)."""
    try:
        now = fingerprint_fn()
    except (OSError, ValueError):
        return False
    return recorded is not None and recorded == now


def load_sources(version):
    src = {}
    stamp, rows = _rows(DATA / "kinematics_summary.csv")
    src["kin"] = (stamp, {r["metric"]: r for r in rows})
    veh = json.loads((DATA / "vehicle_derived.json").read_text())
    src["veh"] = (veh.get("_hardpoints"), veh)
    src["chk"] = _rows(DATA / "member_checks.csv")
    src["joint"] = _rows(DATA / "joint_checks.csv")
    src["mis"] = _rows(DATA / "misalignment.csv")
    src["intf"] = _rows(DATA / "interference.csv")
    with open(DATA / "interference.csv") as f:
        m = re.search(r"poses (\d+)", f.readline())
    src["intf_poses"] = int(m.group(1)) if m else 0
    src["mass"] = _rows(DATA / "mass_budget.csv")
    src["fast"] = _rows(DATA / "fastener_checks.csv")
    with open(DATA / "bom_costed.csv") as f:
        head = f.readline()
        src["bom"] = list(csv.DictReader(l for l in f if not l.startswith("#")))
    from bom import bom_sha
    if f"bom.csv {bom_sha()}" not in head:
        raise RuntimeError("bom_costed.csv is not from today's bom.csv: run bom.py (run_all.py does)")
    upl = json.loads((DATA / "upright_loads.json").read_text())
    src["upl"] = (upl.get("_hardpoints"), upl)
    brg = json.loads((DATA / "bearings.json").read_text())
    src["brg"] = (brg.get("_hardpoints"), brg)
    src["bolt"] = _rows(DATA / "bolt_checks.csv")
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    src["hc"] = (hc.get("_hardpoints"), hc)
    src["hc_current"] = _matches(hc.get("_inputs_sha"), fingerprint.handcalc_inputs_hash)
    stale = [k for k, v in src.items() if isinstance(v, tuple) and v[0] != version]
    cad = fingerprint.cad_inputs_hash()
    stale += [f"{name} (CAD changed since)" for name in ("interference.csv", "mass_budget.csv", "fastener_checks.csv")
              if _cad_stamp(DATA / name) != cad]
    with open(DATA / "fastener_checks.csv") as f:
        m = re.search(r"inputs ([0-9a-f]{16})", f.readline())
    if not m or m.group(1) != fingerprint.fastener_inputs_hash():
        stale.append("fastener_checks.csv (hardware, tool sizes, poses or sweep code changed since)")
    try:
        load_mesh_index()                 # the meshes the CAD outputs came from: inputs, upright STEP, each file's hash
    except (RuntimeError, FileNotFoundError, ValueError) as e:      # (ValueError: a STEP with no DATA section)
        stale.append(f"CAD meshes: {e}")
    stale += stamps.stale()               # every run_all output against the content of what it was made from
    fea_path = DATA / "upright_fea.json"
    fea = json.loads(fea_path.read_text()) if fea_path.exists() else None
    src["fea_current"] = bool(fea) and fea.get("_hardpoints") == version and _matches(fea.get("_inputs_sha"), fingerprint.fea_inputs_hash)
    src["fea"] = fea
    fat_path = DATA / "upright_fatigue.json"
    fat = json.loads(fat_path.read_text()) if fat_path.exists() else None
    src["fat_current"] = bool(fat) and fat.get("_hardpoints") == version and _matches(fat.get("_inputs_sha"), fingerprint.fatigue_inputs_hash)
    src["fat"] = fat
    bench_path = DATA / "fea_benchmarks.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    src["bench_current"] = bool(bench) and bench.get("_inputs_sha") == fingerprint.bench_inputs_hash()
    src["bench"] = bench
    if stale:
        raise RuntimeError(f"outputs not current (hardpoints {version}): {stale}; run scripts/run_all.py")
    return src


def evaluate(spec, src):
    """-> (value text, status)"""
    unit = spec.get("unit", "")
    note = spec.get("note", "")
    if "open" in spec:
        return spec["open"], "OPEN"
    if "retired" in spec:
        if not re.match(r"RC-\d+: \S", spec["retired"]):
            raise RuntimeError(f"a retired requirement needs its change record and reason ('RC-nn: ...'): {spec['retired']!r}")
        return spec["retired"], "RETIRED"
    if "kin" in spec:
        r = src["kin"][1][spec["kin"]]
        val, status = float(r["value"]), r["status"] or "INFO"
        unit = r["unit"]
    elif "veh" in spec:
        val = float(src["veh"][1][spec["veh"]])
        status = None
    elif "chk_min" in spec or "joint_min" in spec or "mis_min" in spec or "bolt_min" in spec:
        key = next(k for k in ("chk", "joint", "mis", "bolt") if f"{k}_min" in spec)
        col = spec[f"{key}_min"]
        rows = src[key][1]
        name_col = {"chk": "member", "joint": "joint", "mis": "joint", "bolt": "joint"}[key]
        vals = [(float(r[col]), r[name_col]) for r in rows if r[col] not in ("", "inf", "nan")]
        val, who = min(vals)
        note = f"worst: {who}" + (f"; {note}" if note else "")
        status = None
    elif "intf_count" in spec:
        rows = src["intf"][1]
        val = float(sum(r["status"] == spec["intf_count"] for r in rows))
        status = None
        note = f"worst of {len(rows)} body pairs checked over {src['intf_poses']} poses" + (f"; {note}" if note else "")
    elif "upl" in spec:
        v = src["upl"][1][spec["upl"]]
        val, status = float(v["value"] if isinstance(v, dict) else v), None
        note = f"worst: {v['lc']}" if isinstance(v, dict) else note
    elif "hc" in spec:
        val = float(src["hc"][1][spec["hc"]])
        status = None if src.get("hc_current", True) else "STALE (re-run scripts/handcalcs.py)"
    elif "brg" in spec:
        v = src["brg"][1]
        for k in spec["brg"].split("."):
            v = v[k]
        val, status = float(v), None
    elif "fea" in spec or "fat" in spec or "bench" in spec:
        key = next(k for k in ("fea", "fat", "bench") if k in spec)
        script = {"fea": "upright_fea.py", "fat": "upright_fatigue.py", "bench": "benchmark_peaks.py"}[key]
        if src.get(key) is None:
            return f"not run (scripts/fea/{script})", "OPEN"
        val = float(src[key][spec[key]])
        status = None if src[f"{key}_current"] else f"STALE (re-run scripts/fea/{script})"
        note = note.replace("{crit}", str(src[key].get("critical_case", "")))
    elif "mass_part" in spec:
        rows = [r for r in src["mass"][1] if r["part"].startswith(spec["mass_part"])]
        if len(rows) != spec["parts"]:
            raise RuntimeError(f"mass_part {spec['mass_part']!r}: {len(rows)} rows in mass_budget.csv, expected {spec['parts']}")
        val, status = sum(float(r["mass_kg"]) for r in rows), None
        note = f"{len(rows)} parts" + (f"; {note}" if note else "")
    elif "fast_count" in spec:
        want = spec.get("check")
        want = [want] if isinstance(want, str) else want
        rows = [r for r in src["fast"][1] if want is None or r["check"] in want]
        missing = [w for w in (want or [None]) if not any(w is None or r["check"] == w for r in rows)]
        if not rows or missing:
            raise RuntimeError(f"fastener_checks.csv has no {missing or spec.get('check', '')} checks: a count of zero "
                               "FAILs would pass vacuously")
        val, status = float(sum(r["status"] == spec["fast_count"] for r in rows)), None
        note = f"{len(rows)} checks over the sweep's poses (notes/fasteners.md)" + (f"; {note}" if note else "")
    elif "bom_lines" in spec:                # BOM lines that must be present with a quantity
        have = {r["id"]: float(r["qty"]) for r in src["bom"]}
        missing = [i for i in spec["bom_lines"] if have.get(i, 0.0) <= 0]
        val, status = float(len(spec["bom_lines"]) - len(missing)), None
        note = (f"missing {missing}; " if missing else "") + note
    elif "fast_joints" in spec:              # distinct joints named like this with no FAIL (a WARN is reported)
        rows = [r for r in src["fast"][1] if spec["fast_joints"] in r["item"]]
        joints = {}
        for r in rows:
            name = r["item"].split(":")[0]
            joints[name] = joints.get(name, True) and r["status"] != "FAIL"
        warns = sum(r["status"] == "WARN" for r in rows)
        val, status = float(sum(joints.values())), None
        note = (f"{len(joints)} found in the fastener sweep" + (f", {warns} WARN rows (see notes/fasteners.md)" if warns else "")
                + (f"; {note}" if note else ""))
    elif "mass_unsprung" in spec:
        val = sum(float(r["mass_kg"]) * float(r["unsprung_share"]) for r in src["mass"][1])
        status = None
    else:
        val, status = float(spec["const"]), None
    if status is None:
        lo, hi = spec.get("lo", -math.inf), spec.get("hi", math.inf)
        status = "PASS" if lo - 1e-9 <= val <= hi + 1e-9 else "FAIL"
    if status == "FAIL" and "deviation" in spec:
        status = f"DEVIATION {spec['deviation']}"
    elif status == "FAIL" and "action" in spec:
        status = f"ACTION: {spec['action']}"
    txt = f"{val:.3g}" if abs(val) < 1000 else f"{val:.0f}"
    return f"{txt} {unit}".strip() + (f" ({note})" if note else ""), status


def is_current(path=None):
    """-> (True, "") if notes/requirements_matrix.md is what build() gives from today's outputs, else (False, why).
    Raises like load_sources when an output is stale."""
    path = path or NOTES / "requirements_matrix.md"
    if not path.exists():
        return False, "requirements_matrix.md does not exist"
    if path.read_text(encoding="utf-8") != build()[0]:
        return False, "requirements_matrix.md is not what today's outputs give: run requirements_matrix.py"
    return True, ""


def main():
    text, summary, version = build()
    (NOTES / "requirements_matrix.md").write_text(text, encoding="utf-8")
    print(f"requirements matrix ({version}): {summary}")


def build():
    """-> (the matrix page's text, the summary line, the hardpoint version), from today's outputs."""
    version = read_version(DATA / "hardpoints.csv")
    reg = json.loads((DATA / "requirements.json").read_text(encoding="utf-8"))
    src = load_sources(version)
    md = [f"# Requirements traceability matrix (hardpoints {version})", "",
          "Generated by `scripts/requirements_matrix.py` from `data/requirements.json` and the analysis outputs. "
          "Do not edit by hand: every value and status is read from a generated file.", ""]
    counts = {}
    for g in reg["groups"]:
        md += [f"## {g['name']}", "", "| ID | Requirement | Target | Source | Verified by | Result | Status |",
               "|---|---|---|---|---|---|---|"]
        for it in g["items"]:
            val, status = evaluate(it["value"], src)
            key = status.split()[0].rstrip(":")
            counts[key] = counts.get(key, 0) + 1
            md.append(f"| {it['id']} | {it['req']} | {it['target']} | {it['source']} | {it['verify']} | {val} | **{status}** |")
        md.append("")
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    md.insert(4, f"**Summary:** {summary}.")
    md.insert(5, "")
    return "\n".join(md) + "\n", summary, version


if __name__ == "__main__":
    main()
