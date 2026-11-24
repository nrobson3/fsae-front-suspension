"""Design FMEA and test plan (DVP&R), PLAN s17 step D.

data/dfmea.json holds the judgement (failure modes, ratings, controls, tests); this script checks it against the
generated outputs and renders notes/dfmea.md. Results in its text are not typed (load-case definitions and makers'
published figures are quoted): {req:ID} is the number in that
requirement's result in the matrix, {veh:key} and {hc:key} are values from vehicle_derived.json and
handcalc_results.json (an optional format after a second colon, e.g. {hc:s20_drop_bolt_fos:.2f}).
It refuses to write (fails closed) unless:
  - the matrix is current (what requirements_matrix.build() gives from today's outputs) and lists every requirement in
    requirements.json;
  - every placeholder resolves and no stray brace is left in any text;
  - every requirement whose status is not exactly PASS or RETIRED (a deviation, action, open, fail or stale item)
    is named in some row's or test's "closes", with what would close it;
  - every WARN row in data/fastener_checks.csv matches a row's "covers_warn" pattern, and that row has a
    "WARN: <pattern>" entry in its "closes";
  - every requirement, test, evidence file and placeholder the rows name exists; ratings are integers 1-10; an
    action never raises a rating or changes severity; a row with no action keeps its ratings; every row of
    severity 9 or 10 has an action.
Run after requirements_matrix.py:  python dfmea.py
"""
import csv
import json
import re

from common import DATA, NOTES, ROOT, read_version

ID_RE = re.compile(r"^\| ([A-Z]+-\d+[a-z]?) \|")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")     # the matrix prints with .3g: exponents included
PH_RE = re.compile(r"\{(req|veh|hc):([A-Za-z0-9_\-]+)(?::([^}]*))?\}")


def matrix(path=NOTES / "requirements_matrix.md"):
    """-> (hardpoint version, {id: (status, result text)})."""
    lines = path.read_text(encoding="utf-8").splitlines()
    m = re.search(r"\(hardpoints (v[\d.]+)\)", lines[0]) if lines else None
    out = {}
    for line in lines:
        mm = ID_RE.match(line)
        if mm:
            _, result, status = line.rstrip().rstrip("|").rsplit("|", 2)
            out[mm.group(1)] = (status.strip().strip("*").strip(), result.strip())
    return (m.group(1) if m else None), out


def warn_items(path=DATA / "fastener_checks.csv"):
    with open(path, encoding="utf-8") as f:
        f.readline()
        return sorted({r["item"] for r in csv.DictReader(f) if r["status"] == "WARN"})


def rpn(r):
    return r["S"] * r["O"] * r["D"]


def fill(text, mat, veh, hc):
    """Replace the placeholders; -> (text, [unresolved placeholders]). Any brace left after filling (another source,
    a dotted key, a stray space) counts as unresolved: the text uses braces for nothing else."""
    missing = []

    def sub(m):
        src, key, fmt = m.group(1), m.group(2), m.group(3)
        try:
            if src == "req":
                status, result = mat.get(key, ("", ""))
                n = NUM_RE.search(result)
                if not n or status.startswith(("RETIRED", "OPEN")):   # no result in the text (e.g. "RC-30: ...")
                    missing.append(m.group(0))
                    return m.group(0)
                return format(float(n.group(0)), fmt) if fmt else n.group(0)
            val = (veh if src == "veh" else hc).get(key)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                missing.append(m.group(0))
                return m.group(0)
            return format(val, fmt) if fmt else str(val) if isinstance(val, int) else format(val, ".3g")
        except ValueError:                                 # a bad format: name the placeholder
            missing.append(f"{m.group(0)} (bad format)")
            return m.group(0)
    out = PH_RE.sub(sub, text)
    rest = out
    for p in missing:                                  # already reported
        rest = rest.replace(p.removesuffix(" (bad format)"), "")
    stray = [j for j in (rest.find("{"), rest.find("}")) if j >= 0]
    if stray:
        missing.append(f"stray brace near '{rest[max(0, min(stray) - 25):min(stray) + 25]}'")
    return out, missing


def check(fm, req_ids, mat, warns, veh=None, hc=None):
    """-> list of problems (empty = valid). mat: {id: (status, result)} from the matrix."""
    bad = []
    tests = {t["id"]: t for t in fm["tests"]}
    ids = [r["id"] for r in fm["rows"]] + list(tests)
    if len(set(ids)) != len(ids):
        bad.append("repeated row or test ids")
    bad += [f"{q} is in requirements.json but not in the matrix: run requirements_matrix.py" for q in sorted(req_ids - set(mat))]
    for r in fm["rows"]:
        for k in ("S", "O", "D"):
            for v in (r[k], r["after"][k]):
                if not (isinstance(v, int) and 1 <= v <= 10):
                    bad.append(f"{r['id']}: {k} rating {v!r} is not an integer 1-10")
            if r["after"][k] > r[k]:
                bad.append(f"{r['id']}: the action raises {k}")
        if r["after"]["S"] != r["S"]:
            bad.append(f"{r['id']}: an action does not change severity (only a redesign of the effect does)")
        if not r["actions"] and any(r["after"][k] != r[k] for k in ("O", "D")):
            bad.append(f"{r['id']}: ratings lowered with no action")
        if r["S"] >= 9 and not r["actions"]:
            bad.append(f"{r['id']}: severity {r['S']} with no action")
        bad += [f"{r['id']}: unknown test {t}" for t in r["actions"] if t not in tests]
        bad += [f"{r['id']}: evidence {e} missing" for e in r["evidence"] if not (ROOT / e).exists()]
        for p in r.get("covers_warn", []):
            if not p.strip():
                bad.append(f"{r['id']}: an empty covers_warn pattern would cover every warning")
            if f"WARN: {p}" not in r["closes"]:
                bad.append(f"{r['id']}: covers '{p}' warnings but does not say what closes them")
    for x in fm["rows"] + fm["tests"]:
        bad += [f"{x['id']}: unknown requirement {q}" for q in x["requirements"] if q not in req_ids]
        bad += [f"{x['id']}: closes unknown requirement {q}" for q in x["closes"] if not q.startswith("WARN: ") and q not in req_ids]
        bad += [f"{x['id']}: says nothing about what closes {q}" for q, how in x["closes"].items() if not str(how).strip()]
        if veh is not None:
            for v in x.values():
                for s in ([v] if isinstance(v, str) else list(v.values()) if isinstance(v, dict) else []):
                    if isinstance(s, str):
                        bad += [f"{x['id']}: unresolved {p}" for p in fill(s, mat, veh, hc)[1]]
    closed = {q for x in fm["rows"] + fm["tests"] for q in x["closes"]}
    bad += [f"{q} is {s} in the matrix and no row or test says what closes it" for q, (s, _) in mat.items()
            if s not in ("PASS", "RETIRED") and q not in closed]
    pats = [p for r in fm["rows"] for p in r.get("covers_warn", [])]
    bad += [f"fastener WARN '{w}' is covered by no row" for w in warns if not any(p in w for p in pats)]
    return bad


def main():
    text, s = build()
    (NOTES / "dfmea.md").write_text(text, encoding="utf-8")
    print(f"dfmea ({s['version']}): {s['rows']} failure modes, {s['tests']} tests; highest RPN {s['rpn_max_before']} -> "
          f"{s['rpn_max_after']} after actions; {s['closing']} closing entries; {s['warns']} fastener warnings covered")
    return s


def is_current(path=None):
    """-> (True, "") if notes/dfmea.md is what build() gives today (so its coverage check passes), else (False, why)."""
    path = path or NOTES / "dfmea.md"
    if not path.exists():
        return False, "dfmea.md does not exist"
    if path.read_text(encoding="utf-8") != build()[0]:
        return False, "dfmea.md is not what today's dfmea.json and matrix give: run dfmea.py"
    return True, ""


def build():
    """-> (the DFMEA page's text, its counts); raises unless the checks in the module docstring pass."""
    ver = read_version(DATA / "hardpoints.csv")
    fm = json.loads((DATA / "dfmea.json").read_text(encoding="utf-8"))
    reqs = json.loads((DATA / "requirements.json").read_text(encoding="utf-8"))
    req_ids = {it["id"] for g in reqs["groups"] for it in g["items"]}
    mver, mat = matrix()
    if mver != ver:
        raise RuntimeError(f"requirements_matrix.md is for hardpoints {mver}, not {ver}: run requirements_matrix.py")
    import requirements_matrix
    ok, why = requirements_matrix.is_current()
    if not ok:
        raise RuntimeError(why)
    if set(mat) != req_ids:
        raise RuntimeError(f"the matrix's ids differ from requirements.json: {sorted(set(mat) ^ req_ids)}")
    veh = json.loads((DATA / "vehicle_derived.json").read_text())
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    import fingerprint
    if veh.get("_hardpoints") != ver or hc.get("_hardpoints") != ver or hc.get("_inputs_sha") != fingerprint.handcalc_inputs_hash():
        raise RuntimeError("vehicle_derived.json or handcalc_results.json is not current: run run_all.py")
    import fasteners
    clear, why = fasteners.sweep_clear()
    if not clear:
        raise RuntimeError(f"the fastener sweep is not current or not clear: {why}")
    warns = warn_items()
    bad = check(fm, req_ids, mat, warns, veh, hc)
    if bad:
        raise RuntimeError("dfmea.json does not cover the design: " + "; ".join(bad))
    f = lambda s: fill(s, mat, veh, hc)[0]                     # noqa: E731
    rows, tests = fm["rows"], fm["tests"]
    md = [f"# Design FMEA and test plan (hardpoints {ver})", "",
          "Generated by `scripts/dfmea.py` from `data/dfmea.json` (the ratings and controls); the numbers in the text "
          "come from the generated matrix and results. Do not edit by hand. The script refuses to write unless every "
          "requirement that is not passing, and every fastener-sweep warning, has a row or a test that says what would "
          "close it.", "",
          "RPN = severity x occurrence x detection, before and after the recommended actions (the after ratings "
          "assume the actions are done). Scales (anchor points; values between them sit between):", ""]
    for k, name in (("S", "Severity"), ("O", "Occurrence"), ("D", "Detection")):
        md.append(f"- **{name}:** " + "; ".join(f"{n} {t}" for n, t in sorted(fm["scales"][k].items(), key=lambda kv: -int(kv[0]))) + ".")
    md += ["", "## DFMEA", "",
           "| ID | Item / function | Failure mode | Effect | S | Cause | O | Prevention | Detection | D | RPN | Actions | "
           "After S/O/D | RPN after | Requirements |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-rpn(r), r["id"])):
        a = r["after"]
        md.append(f"| {r['id']} | **{f(r['item'])}**: {f(r['function'])} | {f(r['mode'])} | {f(r['effect'])} | {r['S']} | "
                  f"{f(r['cause'])} | {r['O']} | {f(r['prevention'])} | {f(r['detection'])} | {r['D']} | **{rpn(r)}** | "
                  f"{', '.join(r['actions']) or '-'}: {f(r['action_text'])} | {a['S']}/{a['O']}/{a['D']} | {rpn(a)} | "
                  f"{', '.join(r['requirements'])} |")
    md += ["", "Evidence for each row's controls:", ""]
    md += [f"- {r['id']}: " + ", ".join(f"`{e}`" for e in r["evidence"]) for r in rows]
    md += ["", "## Test plan (DVP&R)", "",
           "| ID | Test | Method | Requirements | Acceptance | Sample | When |", "|---|---|---|---|---|---|---|"]
    md += [f"| {t['id']} | {f(t['name'])} | {f(t['method'])} | {', '.join(t['requirements'])} | {f(t['acceptance'])} | "
           f"{f(t['sample'])} | {f(t['when'])} |" for t in tests]
    md += ["", "## What closes each open item", "",
           "Every requirement not passing in `notes/requirements_matrix.md`, and every warning in `notes/fasteners.md`:", "",
           "| Item | Status now | Closed by | How |", "|---|---|---|---|"]
    for x in rows + tests:
        for q, how in x["closes"].items():
            md.append(f"| {q} | {mat[q][0] if q in mat else 'interface (fastener sweep)'} | {x['id']} | {f(how)} |")
    before, after = max(rpn(r) for r in rows), max(rpn(r["after"]) for r in rows)
    return "\n".join(md) + "\n", {"version": ver, "rows": len(rows), "tests": len(tests), "rpn_max_before": before,
                                  "rpn_max_after": after, "closing": sum(len(x["closes"]) for x in rows + tests),
                                  "warns": len(warns)}


if __name__ == "__main__":
    main()
