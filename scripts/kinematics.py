"""Kinematic sweeps (heave, roll, steer) for the front corner, scored against the targets in data/requirements.json (PLAN s3).

Run:  python kinematics.py [--hardpoints ../data/hardpoints.csv] [--no-plots]
Writes: data/kinematics_summary.csv, data/sweep_heave.csv, data/sweep_roll.csv,
        data/sweep_steer.csv, plots/k*.png, notes/kinematics_results.md
"""
import argparse
import csv
import math

import numpy as np
from scipy.optimize import brentq

import stamps
from common import (DATA, NOTES, PLOTS, INK, INK_2, MUTED, SERIES, FAIL, band, figure, load_hardpoints,
                    load_vehicle, read_version, style_axes)
from suspension import Corner, make_corner, front_view_ic, front_view_line, line_intersect_2d, mirror, point_velocity

HEAVE_EVAL = 25.0      # mm, window for bump steer, RC migration and MR progression
ROLL_EVAL = 2.0        # deg, window for RC migration in roll
ROLL_RECOVERY = (0.5, 2.0)  # deg, window over which camber recovery is judged
STOP_MARGIN = 0.5      # deg of outer-wheel angle between the hairpin lock and the rack steering stop (v1.5: 1 deg; v1.7: 0.5 deg,
                       # because the rim gap at full bump + lock falls about 2 mm per mm of rack and the stack-up needs it)

TARGETS = {  # metric: (lo, hi, unit, source)
    "static_camber":   (-1.5, -1.0, "deg", "PLAN s3"),
    "camber_recovery": (60.0, 80.0, "%", "PLAN s3"),
    "caster":          (3.0, 6.0, "deg", "PLAN s3"),
    "kpi":             (4.0, 8.0, "deg", "PLAN s3"),
    "scrub":           (0.0, 20.0, "mm", "PLAN s3"),
    "trail":           (10.0, 20.0, "mm", "PLAN s3"),
    "rc_height":       (20.0, 50.0, "mm", "PLAN s3"),
    "rc_migration":    (-15.0, 15.0, "mm", "PLAN s3"),
    "bump_steer":      (-0.1, 0.1, "deg", "PLAN s3"),
    "motion_ratio":    (0.9, 1.1, "-", "PLAN s3"),
    "mr_progression":  (0.0, 10.0, "%", "PLAN s3 'near linear or slightly rising'"),
    "ackermann":       (40.0, 60.0, "%", "PLAN s3"),
}


# ------------------------------------------------------------------ sweeps
def march(corner, values, fn):
    """Solve poses for `values` (must include 0) marching outward from 0 for continuity."""
    values = np.asarray(values, float)
    out = {}
    for side in (values[values >= 0], values[values <= 0][::-1]):
        warm = None
        for v in side:
            p = fn(v, warm)
            out[float(v)] = p
            warm = p
    return [out[float(v)] for v in values]


def enrich(corner, pose, veh):
    """Add velocity-derived metrics: motion ratio, IC, RC (heave), anti-dive."""
    w, v_c, c, mr, arb_rate = corner.velocity(pose)
    cp = pose["cp"]
    ic = front_view_ic(w, v_c, c, cp)
    pose["w"], pose["v_c"], pose["ic"], pose["mr"], pose["arb_rate"] = w, v_c, ic, mr, arb_rate
    pose["camber_rate"] = -math.degrees(w[0])            # deg per mm of bump (camber sign)
    v_cp = point_velocity(w, v_c, c, cp)
    pose["fv_dir"] = front_view_line(w, v_c, c, cp)
    pose["fvsa"] = v_cp[2] / w[0] if abs(w[0]) > 1e-12 else math.inf
    pose["rc_heave"] = cp[1] * v_cp[1] / v_cp[2]  # symmetric heave: half-track x lateral scrub rate
    L, h = veh["wheelbase_mm"], veh["cg_height_mm"]
    pose["anti_dive"] = v_cp[0] / v_cp[2] * (L / h) * veh["_front_brake_fraction"] * 100.0
    return pose


def heave_sweep(corner, veh, zs):
    poses = march(corner, zs, lambda z, warm: corner.pose(z, 0.0, warm))
    return [enrich(corner, p, veh) for p in poses]


def roll_sweep(corner, veh, phis_deg):
    T = veh["track_front_mm"]
    rows = []
    cache = {}

    def left(z):
        key = round(z, 9)
        if key not in cache:
            cache[key] = enrich(corner, corner.pose(z, 0.0), veh)
        return cache[key]

    static = left(0.0)
    for phi in phis_deg:
        z = T / 2 * math.tan(math.radians(phi))
        pl, pr = left(z), left(-z)                   # right wheel = mirror of left at -z
        cpL, cpR = pl["cp"], mirror(pr["cp"])
        dL, dR = pl["fv_dir"], pr["fv_dir"] * np.array([-1.0, 1.0])  # right line = mirror of left at -z
        u = (cpL - cpR)[1:]
        u = u / np.linalg.norm(u)
        alpha = math.degrees(math.atan2(u[1], u[0]))  # ground tilt seen from the body = roll angle
        if abs(phi) < 1e-9:
            rc = np.array([0.0, static["rc_heave"] + cpL[2]])
        else:
            rc = line_intersect_2d(cpL[1:], dL, cpR[1:], dR)
        nrm = np.array([-u[1], u[0]])                 # ground normal (up)
        rc_h = float(np.dot(rc - cpR[1:], nrm))
        # where the body centreline (y = 0) meets the ground line
        t0 = (0.0 - cpR[1]) / u[0]
        g0 = cpR[1:] + t0 * u
        rc_lat = float(np.dot(rc - g0, u))
        cam_out = pl["camber"] + alpha               # left wheel (outside, in bump, when roll > 0), vs ground
        cam_in = pr["camber"] - alpha                # right wheel (inside, in droop, when roll > 0), vs ground
        rec = -(pl["camber"] - static["camber"]) / alpha * 100.0 if abs(alpha) > 1e-6 else float("nan")
        twist = math.degrees(pl["be"] - pr["be"])     # ARB twist, deg
        rows.append({"phi_cmd": phi, "roll": alpha, "z": z, "rc_height": rc_h, "rc_lateral": rc_lat,
                     "camber_left_ground": cam_out, "camber_right_ground": cam_in, "camber_recovery_left": rec,
                     "arb_twist": twist, "toe_in_left": -pl["steer"]})
    return rows


def steer_sweep(corner, veh, racks):
    poses = march(corner, racks, lambda s, warm: corner.pose(0.0, s, warm))
    for p in poses:
        P, k = p["P"], p["kingpin"]
        arm = P["TR_O"] - P["LCA_O"]
        arm -= np.dot(arm, k) * k
        tie = P["TR_I"] - P["TR_O"]
        tie -= np.dot(tie, k) * k
        mu = math.degrees(math.acos(np.clip(np.dot(arm, tie) / np.linalg.norm(arm) / np.linalg.norm(tie), -1, 1)))
        p["transmission"] = mu
        p["toggle_margin"] = min(mu, 180.0 - mu)
        w, v_c, c, mr, _ = corner.velocity(p)
        p["mr"] = mr
    return poses


def ackermann_table(steer_poses, veh):
    """Left turn = +rack. Inner = left wheel at +s, outer = right wheel = mirror of left at -s."""
    L, t = veh["wheelbase_mm"], veh["track_front_mm"]
    by_s = {round(p["rack"], 9): p for p in steer_poses}
    rows = []
    for p in steer_poses:
        s = p["rack"]
        if s <= 0 or round(-s, 9) not in by_s:
            continue
        q = by_s[round(-s, 9)]
        d_in, d_out = p["steer"], -q["steer"]
        if d_out <= 0.05:
            continue
        d_in_ideal = math.degrees(math.atan(1.0 / (1.0 / math.tan(math.radians(d_out)) - t / L)))
        ack = (d_in - d_out) / (d_in_ideal - d_out) * 100.0
        rows.append({"rack": s, "inner": d_in, "outer": d_out, "inner_100pct": d_in_ideal, "ackermann": ack,
                     "camber_inner": p["camber"], "camber_outer": q["camber"],
                     "toggle_margin": min(p["toggle_margin"], q["toggle_margin"]),
                     "jacking_inner": p["cp"][2], "jacking_outer": q["cp"][2]})
    return rows


def required_lock(veh):
    """Outer front wheel angle for the FSAE hairpin (geometric, low-speed, rear axle = turn centre line)."""
    L = veh["wheelbase_mm"] / 1000.0
    r_edge = veh["hairpin_outside_diameter_m"] / 2 - veh["hairpin_line_margin_m_assumed"]
    r_cp = r_edge - veh["tyre"]["tread_width_mm"] / 2000.0
    return math.degrees(math.asin(L / r_cp)), r_cp


# ------------------------------------------------------------------ evaluation
def interp(xs, ys, x):
    return float(np.interp(x, xs, ys))


@stamps.producer("kinematics", when=lambda hp_path=None, *_, **__: hp_path is None)  # trial hardpoints: not stamped
def evaluate(hp_path=None, plots=True, write=True, quiet=False):
    hp, hp_meta = load_hardpoints(hp_path)
    veh = load_vehicle()
    ax_g = 1.5
    veh["_front_brake_fraction"] = veh["front_weight_fraction"] + ax_g * veh["cg_height_mm"] / veh["wheelbase_mm"]
    corner = make_corner(hp, hp_meta, veh["tyre"]["loaded_radius_mm"])

    zs = np.round(np.arange(-30.0, 30.0 + 1e-9, 1.0), 6)
    heave = heave_sweep(corner, veh, zs)
    phis = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.25), 6)
    roll = roll_sweep(corner, veh, phis)
    racks = np.round(np.arange(-40.0, 40.0 + 1e-9, 1.0), 6)
    steer = steer_sweep(corner, veh, racks)
    ack = ackermann_table(steer, veh)

    z_arr = np.array([p["z"] for p in heave])
    st = heave[int(np.argmin(np.abs(z_arr)))]
    win = np.abs(z_arr) <= HEAVE_EVAL + 1e-9
    toe = np.array([-(p["steer"] - st["steer"]) for p in heave])           # toe-in change, deg
    rc_h = np.array([p["rc_heave"] for p in heave])
    mr = np.array([p["mr"] for p in heave])
    dl = np.array([p["damper_len"] for p in heave])

    roll_arr = np.array([r["roll"] for r in roll])
    rwin = np.abs(roll_arr) <= ROLL_EVAL + 1e-9
    rc_roll = np.array([r["rc_height"] for r in roll])
    rec_mask = (np.abs(roll_arr) >= ROLL_RECOVERY[0]) & (np.abs(roll_arr) <= ROLL_RECOVERY[1]) & (roll_arr > 0)
    rec = np.array([r["camber_recovery_left"] for r in roll])

    d_req, r_cp = required_lock(veh)
    d_stop = d_req + STOP_MARGIN
    a_out = np.array([a["outer"] for a in ack])
    a_rack = np.array([a["rack"] for a in ack])
    s_lock = interp(a_out, a_rack, d_req) if a_out.max() >= d_req else float("nan")
    s_stop = interp(a_out, a_rack, d_stop) if a_out.max() >= d_stop else float("nan")
    lock_row = {k: interp(a_rack, np.array([a[k] for a in ack]), s_lock) for k in ack[0]} if not math.isnan(s_lock) else {}
    stop_row = {k: interp(a_rack, np.array([a[k] for a in ack]), s_stop) for k in ack[0]} if not math.isnan(s_stop) else {}

    mr_lo, mr0, mr_hi = interp(z_arr, mr, -HEAVE_EVAL), st["mr"], interp(z_arr, mr, HEAVE_EVAL)
    stroke = interp(z_arr, dl, -30.0) - interp(z_arr, dl, 30.0)
    rcm_heave = (rc_h[win] - st["rc_heave"])
    cp_z = np.array([p["cp"][2] for p in heave])
    # in bump the ground (and the CG with the body) moves cp_z relative to the body: CG-to-RC arm change
    rcm_body = (rc_h[win] + cp_z[win] - cp_z[np.argmin(np.abs(z_arr))] - st["rc_heave"])
    rcm_roll = (rc_roll[rwin] - st["rc_heave"])

    # rack-travel window that keeps the toggle margin and gives the required lock
    res = []

    def add(metric, value, note="", lo=None, hi=None, unit=None):
        t = TARGETS.get(metric)
        lo = t[0] if lo is None and t else lo
        hi = t[1] if hi is None and t else hi
        unit = t[2] if unit is None and t else (unit or "")
        ok = "" if lo is None or value is None or (isinstance(value, float) and math.isnan(value)) else \
            ("PASS" if lo - 1e-9 <= value <= hi + 1e-9 else "FAIL")
        res.append({"metric": metric, "value": value, "lo": lo, "hi": hi, "unit": unit, "status": ok, "note": note})

    add("static_camber", st["camber"], "body frame, z = 0")
    add("camber_recovery", float(np.min(rec[rec_mask])), f"min over roll {ROLL_RECOVERY[0]}-{ROLL_RECOVERY[1]} deg, outside wheel")
    add("camber_recovery_max", float(np.max(rec[rec_mask])), "max over the same window", 60.0, 80.0, "%")
    add("caster", st["caster"])
    add("kpi", st["kpi"])
    add("scrub", st["scrub"])
    add("trail", st["trail"])
    add("rc_height", st["rc_heave"], "static, above ground")
    add("rc_migration_heave", float(np.max(np.abs(rcm_heave))),
        f"max change of RC height above ground over +/-{HEAVE_EVAL:.0f} mm heave; FAIL accepted as deviation DV-1 "
        "(packaging-limited, see notes/requirements.md)", 0.0, 15.0, "mm")
    add("rc_arm_change_heave", float(np.max(np.abs(rcm_body))),
        f"same window, change of the CG-to-RC roll arm (RC relative to the sprung mass; information)", None, None, "mm")
    add("rc_migration_roll", float(np.max(np.abs(rcm_roll))), f"max |dRC height| over +/-{ROLL_EVAL:.0f} deg roll", 0.0, 15.0, "mm")
    add("rc_lateral_roll", float(np.max(np.abs([r["rc_lateral"] for r, m in zip(roll, rwin) if m]))),
        f"max |lateral RC shift| over +/-{ROLL_EVAL:.0f} deg roll (information)", None, None, "mm")
    add("bump_steer", float(np.max(np.abs(toe[win]))), f"max |toe change| over +/-{HEAVE_EVAL:.0f} mm", 0.0, 0.1, "deg")
    add("motion_ratio", mr0, "static, damper / wheel")
    add("motion_ratio_min", float(mr[win].min()), f"over +/-{HEAVE_EVAL:.0f} mm", 0.9, 1.1, "-")
    add("motion_ratio_max", float(mr[win].max()), f"over +/-{HEAVE_EVAL:.0f} mm", 0.9, 1.1, "-")
    add("mr_progression", (mr_hi - mr_lo) / mr0 * 100.0, f"(MR(+{HEAVE_EVAL:.0f}) - MR(-{HEAVE_EVAL:.0f})) / MR(0)")
    dmp = veh["damper"]
    add("damper_stroke_used", stroke, f"for -30..+30 mm wheel travel; {dmp['model']} stroke {dmp['stroke_mm']} mm",
        0.0, float(dmp["stroke_mm"]), "mm")
    add("damper_len_static", interp(z_arr, dl, 0.0), "eye-to-eye at static ride height (information)", None, None, "mm")
    # bump: travel ends where the damper length = compressed length + the bump rubber's solid height. That must be
    # at or beyond 25 mm of jounce (half of the 50 mm total travel FSAE V.3.1.1 requires), and no further than
    # +30 mm, where the clearance sweep stops.
    rubber, comp = float(dmp["bump_rubber_solid_mm_assumed"]), float(dmp["compressed_mm"])
    add("bump_stop_travel", interp(dl[::-1], z_arr[::-1], comp + rubber),
        f"wheel bump at which the bump rubber ({rubber:g} mm solid, assumed) goes solid; the damper cannot bottom",
        25.0, 30.0, "mm")
    add("bump_rubber_solid_min", interp(z_arr, dl, 30.0) - comp, "smallest rubber solid height that stops travel by "
        "+30 mm (information)", None, None, "mm")
    add("bump_rubber_solid_max", interp(z_arr, dl, 25.0) - comp, "largest rubber solid height that still allows 25 mm "
        "of jounce (information)", None, None, "mm")
    # droop: the damper's extended length is the droop stop (dl falls as z rises, so reverse for np.interp)
    ext = float(dmp["extended_mm"])
    droop_topout = -interp(dl[::-1], z_arr[::-1], ext) if dl.max() >= ext else -float(z_arr.min())
    add("droop_to_topout", droop_topout, "wheel droop at which the damper reaches its extended length (the droop stop); "
        "FSAE V.3.1.1 needs 50 mm in total: 25 mm each way here", 25.0, 1e9, "mm")
    add("anti_dive", st["anti_dive"], "outboard brakes, LC3 brake split (information)", None, None, "%")
    add("fvsa", st["fvsa"], "front-view swing-arm length (information)", None, None, "mm")
    add("hairpin_lock_required", d_req, f"outer wheel, CP radius {r_cp:.2f} m", None, None, "deg")
    add("rack_travel_at_lock", s_lock, "rack travel giving the required outer angle", None, None, "mm")
    if lock_row:
        add("ackermann", lock_row["ackermann"], "at the hairpin lock")
        add("inner_angle_at_lock", lock_row["inner"], "", None, None, "deg")
    if stop_row:
        add("rack_travel_at_stop", s_stop, f"steering stop on the rack at outer angle = lock + {STOP_MARGIN:g} deg", None, None, "mm")
        add("lock_margin_before_stop", s_stop - s_lock, "rack travel left between the hairpin lock and the stop", 0.0, 1e9, "mm")
        add("toggle_margin_at_stop", stop_row["toggle_margin"], "steering arm vs tie rod at the stop, deg from lock-up (FSAE V.3.2.4)",
            30.0, 90.0, "deg")
    a_mid = [a for a in ack if 5.0 <= a["outer"] <= d_stop]
    if a_mid:
        add("ackermann_min", min(a["ackermann"] for a in a_mid), "over 5 deg .. steering stop", 40.0, 60.0, "%")
        add("ackermann_max", max(a["ackermann"] for a in a_mid), "over 5 deg .. steering stop", 40.0, 60.0, "%")
    arb_per_deg = interp(roll_arr, np.array([r["arb_twist"] for r in roll]), 1.0)
    add("arb_twist_per_roll", arb_per_deg, "ARB twist per deg of roll (installation ratio)", None, None, "deg/deg")

    out = {"hp": hp, "veh": veh, "corner": corner, "heave": heave, "roll": roll, "steer": steer, "ack": ack,
           "results": res, "static": st, "toe": toe, "z": z_arr, "d_req": d_req, "s_lock": s_lock, "d_stop": d_stop,
           "version": read_version(hp_path or DATA / "hardpoints.csv")}
    if write:
        write_outputs(out)
    if plots:
        make_plots(out)
    if not quiet:
        print_results(out)
    return out


def print_results(out):
    print(f"hardpoints {out['version']}")
    for r in out["results"]:
        v = r["value"]
        vs = "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:9.3f}"
        band_s = "" if r["lo"] is None else f"[{r['lo']:g}, {r['hi']:g}]"
        print(f"  {r['metric']:24s} {vs} {r['unit']:7s} {band_s:16s} {r['status']:4s} {r['note']}")


# ------------------------------------------------------------------ outputs
def write_outputs(out):
    with open(DATA / "kinematics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# hardpoints", out["version"]])
        w.writerow(["metric", "value", "target_lo", "target_hi", "unit", "status", "note"])
        for r in out["results"]:
            w.writerow([r["metric"], f"{r['value']:.4f}" if r["value"] is not None else "", r["lo"] if r["lo"] is not None else "",
                        r["hi"] if r["hi"] is not None else "", r["unit"], r["status"], r["note"]])
    stamp = f"# hardpoints {out['version']}\n"
    with open(DATA / "sweep_heave.csv", "w", newline="") as f:
        f.write(stamp)
        w = csv.writer(f)
        w.writerow(["z_mm", "camber_deg", "toe_in_change_deg", "caster_deg", "kpi_deg", "scrub_mm", "trail_mm",
                    "cp_y_mm", "rc_height_mm", "fvsa_mm", "motion_ratio", "damper_len_mm", "rocker_deg", "anti_dive_pct"])
        for p, t in zip(out["heave"], out["toe"]):
            w.writerow([f"{p['z']:.1f}", f"{p['camber']:.4f}", f"{t:.4f}", f"{p['caster']:.4f}", f"{p['kpi']:.4f}",
                        f"{p['scrub']:.3f}", f"{p['trail']:.3f}", f"{p['cp'][1]:.3f}", f"{p['rc_heave']:.3f}",
                        f"{p['fvsa']:.1f}", f"{p['mr']:.5f}", f"{p['damper_len']:.3f}", f"{math.degrees(p['th']):.3f}",
                        f"{p['anti_dive']:.2f}"])
    with open(DATA / "sweep_roll.csv", "w", newline="") as f:
        f.write(stamp)
        keys = list(out["roll"][0].keys())
        w = csv.writer(f)
        w.writerow(keys)
        for r in out["roll"]:
            w.writerow([f"{r[k]:.4f}" for k in keys])
    with open(DATA / "sweep_steer.csv", "w", newline="") as f:
        f.write(stamp)
        keys = list(out["ack"][0].keys())
        w = csv.writer(f)
        w.writerow(keys)
        for r in out["ack"]:
            w.writerow([f"{r[k]:.4f}" for k in keys])
    lines = [f"# Kinematics results (hardpoints {out['version']})", "",
             "Generated by `scripts/kinematics.py`. Do not edit by hand.", "",
             "| Metric | Value | Target | Unit | Status | Note |", "|---|---|---|---|---|---|"]
    for r in out["results"]:
        v = r["value"]
        vs = "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
        tgt = "" if r["lo"] is None else f"{r['lo']:g} to {r['hi']:g}"
        note = r["note"].replace("|", "\\|")
        lines.append(f"| {r['metric']} | {vs} | {tgt} | {r['unit']} | {r['status']} | {note} |")
    (NOTES / "kinematics_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=160, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)


def _point_band(ax, x0, lo, hi, half=3.0, label="static target"):
    ax.fill_between([x0 - half, x0 + half], lo, hi, color="#0ca30c", alpha=0.18, linewidth=0, zorder=0)
    ax.text(x0 + half + 0.5, hi, label, fontsize=7, color=INK_2, va="bottom")


def make_plots(out):
    heave, roll, ack, st = out["heave"], out["roll"], out["ack"], out["static"]
    z = out["z"]
    ver = out["version"]
    win = HEAVE_EVAL

    def finish(ax):
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)

    # 1 camber in heave + static target
    fig, axs = figure(1, 2)
    ax = axs[0][0]
    ax.plot(z, [p["camber"] for p in heave], color=SERIES[0])
    _point_band(ax, 0.0, *TARGETS["static_camber"][:2])
    style_axes(ax, f"Camber vs wheel travel ({ver})", "wheel travel, + bump [mm]", "camber, body frame [deg]")
    finish(ax)
    ax = axs[0][1]
    rr = [r for r in roll if r["roll"] > 0.05]
    ax.plot([r["roll"] for r in rr], [r["camber_recovery_left"] for r in rr], color=SERIES[0])
    band(ax, *TARGETS["camber_recovery"][:2])
    ax.axvspan(*ROLL_RECOVERY, color=MUTED, alpha=0.06, linewidth=0)
    style_axes(ax, "Camber recovery in roll, outside wheel", "body roll [deg]", "roll recovered by camber gain [%]")
    _save(fig, "k01_camber.png")

    # 2 caster, KPI (static targets; both move with the upright rotation), scrub, trail
    fig, axs = figure(1, 2)
    ax = axs[0][0]
    ax.plot(z, [p["caster"] for p in heave], color=SERIES[0], label="caster")
    ax.plot(z, [p["kpi"] for p in heave], color=SERIES[1], label="kingpin inclination")
    ax.fill_between([-4.0, -0.5], *TARGETS["caster"][:2], color=SERIES[0], alpha=0.15, linewidth=0, zorder=0)
    ax.fill_between([0.5, 4.0], *TARGETS["kpi"][:2], color=SERIES[1], alpha=0.15, linewidth=0, zorder=0)
    ax.text(-4.0, TARGETS["caster"][0], "caster target (static) ", fontsize=7, color=INK_2, ha="right", va="bottom")
    ax.text(4.5, TARGETS["kpi"][1], "KPI target (static)", fontsize=7, color=INK_2, va="top")
    ax.text(0.02, 0.97, "KPI changes 1:1 with camber gain (same upright rotation)", transform=ax.transAxes,
            fontsize=7, color=INK_2, va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    style_axes(ax, "Caster and KPI vs wheel travel", "wheel travel, + bump [mm]", "angle [deg]")
    finish(ax)
    ax = axs[0][1]
    ax.plot(z, [p["scrub"] for p in heave], color=SERIES[0], label="scrub radius")
    ax.plot(z, [p["trail"] for p in heave], color=SERIES[1], label="mechanical trail")
    band(ax, *TARGETS["scrub"][:2], label="scrub target 0-20")
    ax.axhspan(*TARGETS["trail"][:2], color="#0ca30c", alpha=0.08, linewidth=0, zorder=0)
    ax.text(0.995, TARGETS["trail"][0], " trail target 10-20 ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    style_axes(ax, "Scrub radius and trail vs wheel travel", "wheel travel, + bump [mm]", "[mm]")
    finish(ax)
    _save(fig, "k02_caster_kpi_scrub_trail.png")

    # 3 roll centre
    fig, axs = figure(1, 2)
    ax = axs[0][0]
    rc0 = st["rc_heave"]
    cp0 = st["cp"][2]
    ax.plot(z, [p["rc_heave"] for p in heave], color=SERIES[1], label="height above ground (judged; deviation DV-1)")
    ax.plot(z, [p["rc_heave"] + p["cp"][2] - cp0 for p in heave], color=SERIES[0],
            label="relative to sprung mass = CG-to-RC roll arm (information)")
    _point_band(ax, 0.0, *TARGETS["rc_height"][:2], label="static RC target")
    ax.fill_between([-win, win], rc0 - 15, rc0 + 15, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
    ax.text(win, rc0 - 15, "migration limit +/-15 mm ", fontsize=7, color=INK_2, va="top", ha="right")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="lower left")
    style_axes(ax, "Roll-centre height in heave", "wheel travel, + bump [mm]", "RC height [mm]")
    finish(ax)
    ax = axs[0][1]
    ax.plot([r["roll"] for r in roll], [r["rc_height"] for r in roll], color=SERIES[0], label="height")
    ax.fill_between([-ROLL_EVAL, ROLL_EVAL], rc0 - 15, rc0 + 15, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
    ax.text(ROLL_EVAL, rc0 - 15, "migration limit +/-15 mm over +/-2 deg ", fontsize=7, color=INK_2, va="top", ha="right")
    lat_max = max(abs(r["rc_lateral"]) for r in roll)
    style_axes(ax, "Roll-centre height in roll", "body roll [deg]", "RC height above ground [mm]")
    ax.text(0.02, 0.96, f"lateral RC shift up to {lat_max:.0f} mm over +/-3 deg (information)", transform=ax.transAxes,
            fontsize=7, color=INK_2, va="top")
    finish(ax)
    _save(fig, "k03_roll_centre.png")

    # 4 bump steer
    fig, axs = figure(1, 1, w=6.4, h=3.6)
    ax = axs[0][0]
    ax.plot(z, out["toe"], color=SERIES[0])
    ax.fill_between([-win, win], -0.1, 0.1, color="#0ca30c", alpha=0.10, linewidth=0, zorder=0)
    ax.text(win, 0.1, " target +/-0.1 deg over +/-25 mm", fontsize=7, color=INK_2, va="bottom", ha="right")
    style_axes(ax, f"Bump steer ({ver})", "wheel travel, + bump [mm]", "toe change, + toe-in [deg]")
    finish(ax)
    _save(fig, "k04_bump_steer.png")

    # 5 motion ratio + damper length
    dmp = out["veh"]["damper"]
    fig, axs = figure(1, 2)
    ax = axs[0][0]
    ax.plot(z, [p["mr"] for p in heave], color=SERIES[0])
    band(ax, *TARGETS["motion_ratio"][:2])
    style_axes(ax, "Motion ratio (damper / wheel)", "wheel travel, + bump [mm]", "motion ratio [-]")
    finish(ax)
    ax = axs[0][1]
    ax.plot(z, [p["damper_len"] for p in heave], color=SERIES[0])
    for v, lab in ((dmp["extended_mm"], "fully extended"), (dmp["compressed_mm"], "fully compressed")):
        ax.axhline(v, color=FAIL, linewidth=0.8)
        ax.text(z[-1], v, f"{lab} {v:g} mm ", fontsize=7, color=INK_2, ha="right", va="bottom" if v < 170 else "top")
    style_axes(ax, f"Damper eye-to-eye length ({dmp['model']})", "wheel travel, + bump [mm]", "length [mm]")
    finish(ax)
    _save(fig, "k05_motion_ratio.png")

    # 6 Ackermann + steering lock (up to the steering stop; beyond it is shaded)
    d_req, d_stop = out["d_req"], out["d_stop"]
    shown = [a for a in ack if a["outer"] <= d_stop + 6.0]
    ao = [a["outer"] for a in shown]

    def lock_lines(ax, top_label=True):
        ax.axvline(d_req, color=INK_2, linewidth=0.8)
        ax.axvspan(d_stop, max(ao), color=MUTED, alpha=0.15, linewidth=0)
        if top_label:
            ax.text(d_req, 1.0, " hairpin lock", transform=ax.get_xaxis_transform(), fontsize=7, color=INK_2, va="top")
            ax.text(d_stop, 0.02, " beyond steering stop", transform=ax.get_xaxis_transform(), fontsize=7, color=INK_2)

    fig, axs = figure(1, 2)
    ax = axs[0][0]
    ax.plot(ao, [a["ackermann"] for a in shown], color=SERIES[0])
    band(ax, *TARGETS["ackermann"][:2])
    lock_lines(ax)
    style_axes(ax, "Ackermann vs steer (left turn)", "outer wheel angle [deg]", "Ackermann [%]")
    ax = axs[0][1]
    ax.plot(ao, [a["inner"] for a in shown], color=SERIES[0], label="design inner wheel")
    ax.plot(ao, [a["inner_100pct"] for a in shown], color=SERIES[1], label="100 % Ackermann")
    ax.plot(ao, ao, color=SERIES[2], label="parallel (0 %)")
    lock_lines(ax, top_label=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    style_axes(ax, "Inner vs outer wheel angle", "outer wheel angle [deg]", "inner wheel angle [deg]")
    _save(fig, "k06_ackermann_lock.png")

    # 7 steer side-effects: camber with steer, toggle margin
    fig, axs = figure(1, 2)
    ax = axs[0][0]
    ax.plot(ao, [a["camber_outer"] for a in shown], color=SERIES[0], label="outer wheel")
    ax.plot(ao, [a["camber_inner"] for a in shown], color=SERIES[1], label="inner wheel")
    lock_lines(ax, top_label=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    style_axes(ax, "Camber change with steer (caster and KPI)", "outer wheel angle [deg]", "camber, body frame [deg]")
    ax = axs[0][1]
    ax.plot(ao, [a["toggle_margin"] for a in shown], color=SERIES[0])
    ax.axhline(30, color=FAIL, linewidth=0.8)
    ax.text(ao[0], 30, " 30 deg minimum margin from lock-up", fontsize=7, color=INK_2, va="bottom")
    lock_lines(ax)
    style_axes(ax, "Steering arm / tie-rod angle margin", "outer wheel angle [deg]", "margin from toggle [deg]")
    _save(fig, "k07_steer_effects.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardpoints", default=None)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    canonical = a.hardpoints is None  # a trial file prints results but writes nothing
    evaluate(a.hardpoints, plots=canonical and not a.no_plots, write=canonical)


if __name__ == "__main__":
    main()
