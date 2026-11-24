"""The write-up (PLAN s17 step E): notes/writeup_template.md -> WRITEUP.md, every result read from a generated file
(the template quotes inputs, targets and history from the data files and the decision log).

Placeholders {source:key} or {source:key:format}; format is a Python format spec (default: a whole number as it is,
others .3g; for req the matrix's own rounding), or abs<spec> for a magnitude, or pct<n> for a fraction shown as a
percentage ("52.0 %").
  req    the first number in that requirement's result in notes/requirements_matrix.md
  tgt    that requirement's target text (data/requirements.json)
  mx     the matrix's status counts (PASS, DEVIATION, ACTION, RETIRED), dv_ids and actions (text)
  kin    data/kinematics_summary.csv by metric      lc    a load case's name (data/loadcases.csv)
  veh    data/vehicle_derived.json                  vin   data/vehicle.json (dotted paths)
  hc     data/handcalc_results.json                 fea   data/upright_fea.json (dotted; list items by index)
  fat    data/upright_fatigue.json                  lc4   data/fea_lc4_plot.json
  rim    data/rim_clearance.json                    dfmea rows, tests, rpn_before, rpn_after (data/dfmea.json)
  fast   fails, warns, checks (data/fastener_checks.csv)
  intf   poses, pairs, rim_gap, moving_min, intra_min (data/interference.csv)
  calc   camber_per_turn, camber_toe_per_turn, camber_height_per_turn, toe_per_turn, ride_per_mm: the setup table,
         from the kinematic solver (setup_sensitivities)
  trade  mig_min, mig_max, rec_min, rec_max, uca_y_max, per_10mm, version: the trade studies' frontier (trade_frontier)
  log    stack_worst_mm, stack_rss_mm: the clearance stack-up's totals in notes/requirements.md section 6 (a written
         analysis, quoted)
It fails closed on:
  - an unresolved placeholder, or any brace left after filling;
  - a requirement that is OPEN, FAIL or STALE;
  - any output the matrix's own freshness checks reject (hardpoint versions, the CAD and fastener fingerprints), or a
    stale FEA, fatigue, benchmark, hand-calc or LC4 result;
  - a matrix or DFMEA page that is not what today's outputs give (both are rebuilt and compared), or a matrix whose ids
    differ from requirements.json;
  - a rim check from other hardpoints, rim or steering stop, or one that no longer matches the sweep's pair;
  - a fact the template states in words that the outputs no longer support (template_facts: the governing case, which
    requirement each deviation and action is, the rim's tightest pose, the stack-up, the mesh trend, the warnings'
    kind, the line pressure's case, the trade studies' versions and their point at the design's width);
  - a placeholder on a RETIRED or OPEN requirement (its result text holds no result), or a bad format;
  - a linked file that does not exist with exactly that name.
Run: python writeup.py   Writes WRITEUP.md
"""
import csv
import json
import math
import re

import numpy as np

import dfmea
import fingerprint
import requirements_matrix
import rim_clearance
from common import DATA, NOTES, ROOT, load_hardpoints, read_version

PH = re.compile(r"\{([a-z0-9]+):([A-Za-z0-9_.\-]+)(?::([^}]*))?\}")
SETTLED = ("PASS", "RETIRED")       # a requirement in any other state than these, a deviation or an action stops the run
RIM_AGREE_MM = 0.5                  # the template says the analytic rim check and the sweep agree to within this
# what the template states in words: the run stops if the outputs no longer say the same (template_facts)
DEVIATIONS = {"K-08a": "DV-1", "S-06f": "DV-2", "S-07a": "DV-3", "M-02": "DV-4"}
ACTIONS = {"V-05"}                  # the helper spring: the open items name it as the only action
GOVERNING_CASE = "LC4"              # also the combined case the line pressure is quoted for
WARN_KINDS = ("Caliper bolt", "Rack bracket bolt")     # the open warnings: threads inside bought parts
# the trade studies' hardpoint versions whose frontier still holds for the design's: v1.7 moved only the damper's
# chassis eye, which the roll-centre frontier does not depend on
TRADE_VALID = {"v1.7": ("v1.6", "v1.7")}


def dotted(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def setup_sensitivities(hp, meta, mr, members, tyre_radius, d=0.1):
    """The setup table's about-values, from the kinematic solver with the rack centred: an adjustment lengthens the
    solver's own link lengths by a small d, and the wheel's change is scaled to one turn. On the car the spring perch,
    left alone, holds the lower wishbone (the pushrod and the spring carry the same load), so every pose is solved with
    the lower ball joint at its static height.
    Camber: both upper-wishbone chassis rod ends one turn longer (a pitch each). With the tie rod held that also moves
    toe (camber_toe_per_turn); camber_per_turn is the change once the toe is set back with the tie rod, and
    camber_height_per_turn how far the wheel centre then moves up or down relative to the body.
    Toe: the tie-rod tube is a turnbuckle (a left-hand rod end at its outer end), so one turn changes its length by
    two pitches. Ride height: the spring perch moves the damper's static length; the wheel moves 1 / motion ratio as
    much. The rod-end pitches are read from data/members.csv."""
    from scipy.optimize import brentq
    from suspension import LINKS, make_corner
    pitch = {r["member"]: float(r["rod_end_pitch_mm"]) for r in members}
    if pitch["UCA front leg"] != pitch["UCA rear leg"]:
        raise RuntimeError("the upper legs' rod ends have different pitches: the camber row assumes one")
    i_f, i_r, i_t = (LINKS.index(p) for p in (("UCA_O", "UCA_F"), ("UCA_O", "UCA_R"), ("TR_O", "TR_I")))
    z_lbj = float(hp["LCA_O"][2])

    def wheel(*grow):                       # (camber, toe [deg], wheel-centre height [mm]) with those links longer by d
        c = make_corner(hp, meta, tyre_radius)
        for i in grow:
            c.lengths[i] += d
        z = brentq(lambda zz: c.pose(zz, 0.0)["P"]["LCA_O"][2] - z_lbj, -5.0, 5.0, xtol=1e-10)
        P = c.pose(z, 0.0)["P"]
        s = (P["WA"] - P["WC"]) / np.linalg.norm(P["WA"] - P["WC"])
        return np.array([math.degrees(math.atan2(s[2], math.hypot(s[0], s[1]))), math.degrees(math.atan2(s[0], s[1])),
                         float(P["WC"][2])])
    w0 = wheel()
    legs = (wheel(i_f, i_r) - w0) / d * pitch["UCA front leg"]       # per turn of both upper rod ends
    tie = (wheel(i_t) - w0) / d                                       # per mm of tie rod
    reset = legs - legs[1] / tie[1] * tie                             # the same, with the toe set back by the tie rod
    return {"camber_per_turn": abs(reset[0]), "camber_toe_per_turn": abs(legs[1]),
            "camber_height_per_turn": abs(reset[2]), "toe_per_turn": abs(tie[1]) * 2 * pitch["Tie rod"],
            "ride_per_mm": 1 / mr}


def trade_frontier(camber_csv=DATA / "tradeoff_rc_camber.csv", width_csv=DATA / "tradeoff_rc_chassis_width.csv"):
    """The trade studies' best layouts (run on their own, see run_all.py). From the camber-recovery study (chassis
    bounds only): the range of the best roll-centre migration over its recovery targets, the recoveries reached and
    the widest upper pickup used. From the pickup-width study (rim clearance enforced): the migration bought per 10 mm
    of upper-pickup width, a least-squares slope over its points, and its point at the narrowest pickups. Also the
    hardpoint versions the two studies ran on (their header lines)."""
    hc, tr = _csv(camber_csv)
    hw, tw = _csv(width_csv)
    mig = [float(r["rc_migration_pm25_mm"]) for r in tr]
    rec = [float(r["recovery_pct"]) for r in tr]
    y = [float(r["UCA_in_y"]) for r in tw]
    m = [float(r["rc_migration_ground_pm25_mm"]) for r in tw]
    if len(tr) < 2 or len(set(y)) < 2:
        raise RuntimeError("the trade-study files are too short to quote: re-run the trade studies")
    stamps = [re.search(r"hardpoints (v[\d.]+)", h) for h in (hc, hw)]
    vers = sorted({s.group(1) if s else "unstamped" for s in stamps})
    i_n = int(np.argmin(y))
    return {"mig_min": min(mig), "mig_max": max(mig), "rec_min": min(rec), "rec_max": max(rec),
            "uca_y_max": max(float(r["UCA_in_y"]) for r in tr), "per_10mm": -10 * float(np.polyfit(y, m, 1)[0]),
            "version": " and ".join(vers), "versions": vers, "y_narrowest": y[i_n], "mig_narrowest": m[i_n]}


def _csv(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline()
        if not head.startswith("#"):
            f.seek(0)
            head = ""
        return head, list(csv.DictReader(l for l in f if not l.startswith("#")))


def sources():
    ver = read_version(DATA / "hardpoints.csv")
    msrc = requirements_matrix.load_sources(ver)          # raises unless every analysis output is current
    for key in ("fea_current", "fat_current", "bench_current", "hc_current"):
        if not msrc.get(key):
            raise RuntimeError(f"{key.split('_')[0]} results are stale: re-run them before the write-up")
    mver, mat = dfmea.matrix()
    if mver != ver:
        raise RuntimeError(f"requirements_matrix.md is for {mver}, not {ver}: run run_all.py")
    for ok, why in (requirements_matrix.is_current(), dfmea.is_current()):     # the pages it links, rebuilt and compared
        if not ok:
            raise RuntimeError(why)
    unsettled = {q: s for q, (s, _) in mat.items()
                 if s not in SETTLED and not s.startswith("DEVIATION") and not s.startswith("ACTION")}
    if unsettled:
        raise RuntimeError(f"requirements not settled (open, failed or stale): {unsettled}")
    reqs = json.loads((DATA / "requirements.json").read_text(encoding="utf-8"))
    req_ids = {it["id"] for g in reqs["groups"] for it in g["items"]}
    if set(mat) != req_ids:
        raise RuntimeError(f"the matrix's ids differ from requirements.json: {sorted(set(mat) ^ req_ids)}")
    wrong = {q: mat.get(q, ("absent", ""))[0] for q, dv in DEVIATIONS.items() if mat.get(q, ("absent", ""))[0] != f"DEVIATION {dv}"}
    wrong |= {q: s for q, (s, _) in mat.items() if s.startswith("DEVIATION") and q not in DEVIATIONS}
    if wrong:
        raise RuntimeError(f"the template names the deviations DV-1 to DV-4 by requirement, and the matrix now says {wrong}")
    summ =next(l for l in (NOTES / "requirements_matrix.md").read_text(encoding="utf-8").splitlines() if l.startswith("**Summary:**"))
    dvs = sorted({s.split()[1] for s, _ in mat.values() if s.startswith("DEVIATION")}, key=lambda d: int(d.split("-")[1]))
    acts = [f"{q}, {s.split(': ', 1)[1]}" for q, (s, _) in mat.items() if s.startswith("ACTION")]
    fea = json.loads((DATA / "upright_fea.json").read_text())
    lc4 = json.loads((DATA / "fea_lc4_plot.json").read_text())
    if lc4["_fea_inputs_sha"] != fingerprint.fea_inputs_hash():
        raise RuntimeError("the LC4 plot is not from the current FEA inputs: re-run scripts/fea/plot_lc4.py")
    if not fea["critical_case"] == lc4["case"] == GOVERNING_CASE:
        raise RuntimeError(f"the template says {GOVERNING_CASE} governs; the FEA says {fea['critical_case']}, the plot "
                           f"{lc4['case']}: update the template")
    vin = json.loads((DATA / "vehicle.json").read_text())
    rim = json.loads((DATA / "rim_clearance.json").read_text())
    poses = json.loads((DATA / "poses.json").read_text())
    if (rim["_hardpoints"] != ver or rim.get("_hardpoints_sha") != rim_clearance.hardpoints_sha()
            or abs(rim["rim_half_width_mm"] - vin["wheel"]["rim_width_in"] * 25.4 / 2) > 1e-9
            or abs(rim["rack_stop_mm"] - poses["rack_stop_mm"]) > 1e-9):
        raise RuntimeError("rim_clearance.json is not from today's hardpoints, rim and steering stop: run run_all.py")
    _, fr = _csv(DATA / "fastener_checks.csv")
    ihead, ir = _csv(DATA / "interference.csv")
    _, kin = _csv(DATA / "kinematics_summary.csv")
    _, lcs = _csv(DATA / "loadcases.csv")
    _, members = _csv(DATA / "members.csv")
    pair = lambda r, a, b: (a in r["a"] and b in r["b"]) or (a in r["b"] and b in r["a"])   # noqa: E731
    rim_gap = min(float(r["min_mm"]) for r in ir if pair(r, "Wheel (Braid", "Lower wishbone / legs"))
    if not rim["link"].startswith("lca_") or abs(rim["min_mm"] - rim_gap) > RIM_AGREE_MM:
        raise RuntimeError(f"the template compares the sweep's rim-to-lower-wishbone gap ({rim_gap:.2f} mm) with the "
                           f"analytic check's tightest link ({rim['link']}, {rim['min_mm']:.2f} mm) and says they agree "
                           f"within {RIM_AGREE_MM} mm: they no longer do")
    fm = json.loads((DATA / "dfmea.json").read_text(encoding="utf-8"))
    veh = json.loads((DATA / "vehicle_derived.json").read_text())
    hp, meta = load_hardpoints()
    stack = re.search(r"\*\*Worst case all together / root-sum-square\*\* \| \*\*([\d.]+) / ([\d.]+) mm\*\*",
                      (NOTES / "requirements.md").read_text(encoding="utf-8"))
    if not stack:
        raise RuntimeError("the clearance stack-up's total row is not in notes/requirements.md section 6")
    s = {
        "req": {q: r for q, (_, r) in mat.items()},
        "req_status": {q: st for q, (st, _) in mat.items()},
        "tgt": {it["id"]: it["target"] for g in reqs["groups"] for it in g["items"]},
        "mx": {**{k: int(v) for k, v in re.findall(r"([A-Z]+) (\d+)", summ)},
               "dv_ids": ", ".join(dvs[:-1]) + (" and " if len(dvs) > 1 else "") + dvs[-1] if dvs else "none",
               "actions": "; ".join(acts) if acts else "none"},
        "kin": {r["metric"]: float(r["value"]) for r in kin},
        "lc": {r["id"]: r["name"] for r in lcs},
        "veh": veh, "vin": vin, "hc": json.loads((DATA / "handcalc_results.json").read_text()),
        "fea": fea, "fat": json.loads((DATA / "upright_fatigue.json").read_text()), "lc4": lc4, "rim": rim,
        "dfmea": {"rows": len(fm["rows"]), "tests": len(fm["tests"]),
                  "rpn_before": max(dfmea.rpn(r) for r in fm["rows"]), "rpn_after": max(dfmea.rpn(r["after"]) for r in fm["rows"])},
        "fast": {"fails": sum(r["status"] == "FAIL" for r in fr), "warns": sum(r["status"] == "WARN" for r in fr), "checks": len(fr)},
        "intf": {"poses": int(re.search(r"poses (\d+)", ihead).group(1)), "pairs": len(ir),
                 "rim_gap": rim_gap,
                 "moving_min": min(float(r["min_mm"]) for r in ir if r["kind"] == "moving"),
                 "intra_min": min(float(r["min_mm"]) for r in ir if r["kind"] == "intra" and float(r["limit_mm"]) == 3.0)},
        "calc": setup_sensitivities(hp, meta, veh["motion_ratio_used"], members, vin["tyre"]["loaded_radius_mm"]),
        "trade": trade_frontier(),
        "log": {"stack_worst_mm": float(stack.group(1)), "stack_rss_mm": float(stack.group(2))},
    }
    bad = template_facts(s, ver, hp)
    if bad:
        raise RuntimeError("the template states, in words, things the outputs no longer support: " + "; ".join(bad))
    return s


def template_facts(s, ver, hp):
    """The facts the template states in words, checked against the outputs; -> the ones that no longer hold."""
    bad = []
    st = s["req_status"]
    acts = {q for q, v in st.items() if v.startswith("ACTION")}
    if acts != ACTIONS:
        bad.append(f"the actions are {sorted(acts)}; the open items name {sorted(ACTIONS)}")
    fea, lc4, rim, veh, kin, trade = s["fea"], s["lc4"], s["rim"], s["veh"], s["kin"], s["trade"]
    if lc4.get("level") != 1:
        bad.append("the LC4 plot's caption says level 1")
    if veh.get("brake_line_pressure_case") != GOVERNING_CASE:
        bad.append(f"the line pressure is quoted for the combined case {GOVERNING_CASE}, but peaks in "
                   f"{veh.get('brake_line_pressure_case')}")
    conv = [c["peaks_MPa"]["_global"] for c in fea["convergence"]]
    if not (all(a < b for a, b in zip(conv, conv[1:])) and fea["richardson"] is None):
        bad.append(f"the mesh paragraph says the peak is still creeping up, not asymptotic: {conv}")
    if not (rim["z_mm"] == 30.0 and abs(abs(rim["rack_mm"]) - rim["rack_stop_mm"]) < 1e-6 and rim.get("refined")):
        bad.append("the rim's tightest pose is quoted as full bump at full lock, refined")
    if not s["intf"]["rim_gap"] > s["log"]["stack_worst_mm"]:
        bad.append("the rim's gap no longer exceeds the worst-case stack-up")
    other = [w for w in dfmea.warn_items() if not any(k in w for k in WARN_KINDS)]
    if other:
        bad.append(f"the open warnings are described as threads inside bought parts; also: {other}")
    if ver not in TRADE_VALID or not set(trade["versions"]) <= set(TRADE_VALID[ver]):
        bad.append(f"the trade studies ran on {trade['versions']}; for {ver} the template accepts "
                   f"{TRADE_VALID.get(ver, (ver,))}: re-run them")
    if abs(trade["y_narrowest"] - float(hp["UCA_F"][1])) > 5.0 or trade["mig_narrowest"] < kin["rc_migration_heave"] - 0.5:
        bad.append("'at that width no geometry tried does better' needs the width study's narrowest point at the "
                   "design's pickups, and no better than the design")
    return bad


def fill(text, src):
    missing = []

    def sub(m):
        s, key, fmt = m.group(1), m.group(2), m.group(3) or ""
        try:
            if s == "req":
                if src.get("req_status", {}).get(key, "PASS").startswith(("RETIRED", "OPEN")):
                    raise KeyError(key)                      # its result text holds no result (e.g. "RC-30: ...")
                n = dfmea.NUM_RE.search(src["req"][key])
                if not n:
                    raise KeyError(key)
                if not fmt:
                    return n.group(0)                        # the matrix's own rounding
                val = float(n.group(0))
            else:
                val = dotted(src[s], key)
        except (KeyError, TypeError, IndexError, ValueError):
            missing.append(m.group(0))
            return m.group(0)
        if isinstance(val, str):
            return val
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            missing.append(m.group(0))
            return m.group(0)
        try:
            if fmt.startswith("abs"):                       # a magnitude: e.g. a compressive force
                val, fmt = abs(val), fmt[3:]
            if fmt.startswith("pct"):                       # a fraction as a percentage, house style "52.0 %"
                return f"{100 * val:.{int(fmt[3:] or 0)}f} %"
            if not fmt:                                     # a count stays a whole number; others 3 significant
                return str(val) if isinstance(val, int) else format(val, ".3g")
            return format(val, fmt)
        except ValueError:                                  # a bad format: name the placeholder
            missing.append(f"{m.group(0)} (bad format)")
            return m.group(0)
    return PH.sub(sub, text), missing


def stray_braces(text):
    """Lines with a brace left after filling: the template uses braces for nothing else, so any is a placeholder the
    pattern did not take (a space, a missing colon, an unclosed one)."""
    return [l.strip()[:80] for l in text.splitlines() if "{" in l or "}" in l]


def missing_links(text, root=ROOT):
    """Relative links (inline, with an optional <...> or "title", or reference style; not footnotes) whose file does
    not exist with exactly that name and case. A drive or rooted path counts as missing: it breaks off this machine."""
    targets = re.findall(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", text)
    targets += re.findall(r"^\s*\[(?!\^)[^\]]+\]:\s*<?([^\s>]+)>?", text, re.M)
    bad = []
    for t in targets:
        p = t.split("#", 1)[0]
        if not p or re.match(r"[a-z][a-z0-9+.\-]+:", p, re.I) and not re.match(r"[a-z]:", p, re.I):
            continue                                       # a fragment within the page, or a URL (not a drive letter)
        if re.match(r"([a-z]:|/|\\)", p, re.I) or "\\" in p:
            bad.append(t)
            continue
        here, ok = root, True
        for part in p.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                here = here.parent
                continue
            if not here.is_dir() or part not in {q.name for q in here.iterdir()}:
                ok = False
                break
            here = here / part
        if not ok:
            bad.append(t)
    return targets, bad


def main():
    tpl = (NOTES / "writeup_template.md").read_text(encoding="utf-8")
    out, missing = fill(tpl, sources())
    left = stray_braces(out)
    if missing or left:
        raise RuntimeError(f"unresolved placeholders: {sorted(set(missing))}; lines with a brace left: {left}")
    links, absent = missing_links(out)
    if absent:
        raise RuntimeError(f"the write-up links files that do not exist (with exactly that name): {absent}")
    head = "<!-- Generated by scripts/writeup.py from notes/writeup_template.md: edit the template, not this file. -->\n"
    (ROOT / "WRITEUP.md").write_text(head + out, encoding="utf-8")
    print(f"write-up: {len(out.split())} words, {len(links)} links, every result from a generated file")


if __name__ == "__main__":
    main()
