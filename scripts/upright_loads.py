"""Upright interface loads for every load case: the input to the upright's detail design and FEA.

forces.py solves the wheel, hub and upright as ONE body against the six links, so it never sees the
loads INSIDE that body. Here the body is split:
  hub + wheel:  tyre loads + pad friction on the disc + bearing row reactions = 0
  caliper:      pad friction on the pads + two lug-bolt forces = 0
  upright:      link forces (ball joints, tie rod, pushrod via the LBJ) - bearing rows - caliper bolts = 0
The pad friction is whatever balances the tyre's moment about the spin axis (brakes on in the
contact-patch cases, off otherwise). The upright balance is then a CHECK, not an assumption: its
residual must be under 0.1 % of the largest interface load.

LC6 (steering abuse) gets an upright version: a kerb blocks the front of the tyre sideways and the
blocking force is scaled until the tie rod carries the full rack force.

Everything is reported in the upright's own static frame (the CAD's frame), so the FEA can apply the
loads to the static CAD geometry. Bearing static safety s0 = C0 / P0 per row, P0 = max(Fr, X0 Fr + Y0 Fa) (factors in vehicle.json hub).
Run: python upright_loads.py   Writes data/upright_loads.json and notes/upright_loads.md
"""
import json

import numpy as np

import stamps
from common import DATA, NOTES, load_hardpoints, load_vehicle, read_version, rotvec_matrix, unit
from forces import read_csv_rows, solve_links, tyre_loads
from suspension import make_corner

KERB_AHEAD_MM, KERB_HEIGHT_MM = 150.0, 100.0   # LC6: where the kerb touches the tyre, from the contact patch


def wheel_frame_static(hp):
    s = unit(hp["WA"] - hp["WC"])
    u = unit(np.array([0.0, 0.0, 1.0]) - s * s[2])
    r = unit(np.cross(s, u))
    if r[0] > 0:                         # rear-pointing, as fsae_parts.wheel_frame
        r = -r
    return s, u, r


def split(corner, pose, members, loads, veh, brakes_on):
    """All interface forces (world frame) for one pose and set of tyre loads."""
    P, R = pose["P"], rotvec_matrix(pose["q"][:3])
    acts_at = {"Pushrod": "LCA_O"}
    f = np.zeros(len(members))
    for Fi, ati in loads:
        f += solve_links(pose, members, Fi, acts_at, ati)[0]
    wc = P["WC"]
    s0, u0, r0 = wheel_frame_static(corner.hp)
    s, u, r = R @ s0, R @ u0, R @ r0
    hub, cal = veh["hub"], veh["brake"]["caliper"]
    t = unit(np.cross(s, r))
    # ---- link forces on the upright
    link = {"UBJ": np.zeros(3), "LBJ": np.zeros(3), "TR": np.zeros(3)}
    at = {"UBJ": P["UCA_O"], "LBJ": P["LCA_O"], "TR": P["TR_O"]}
    for j, m in enumerate(members):
        F = f[j] * unit(P[m["in"]] - P[m["out"]])
        key = {"UCA_O": "UBJ", "LCA_O": "LBJ", "TR_O": "TR", "PR_O": "LBJ"}[m["out"]]
        link[key] += F
    # ---- pad friction on the disc (balances the tyre's moment about the spin axis)
    r_eff = veh["brake"]["pad_effective_radius_mm_assumed"]
    x_pad = wc + s * hub["disc_axial_mm"] + r * r_eff
    m_tyre = sum(np.dot(np.cross(a - wc, F), s) for F, a in loads)
    m_unit = np.dot(np.cross(x_pad - wc, t), s)
    lam = -m_tyre / m_unit if brakes_on else 0.0
    if not brakes_on and abs(m_tyre) > 1.0:
        raise RuntimeError(f"brakes off but the tyre load has {m_tyre / 1000:.1f} N m about the spin axis")
    P_disc = lam * t
    # ---- hub: bearing rows (radial, at the pressure centres) + one axial force
    a1 = hub["row_centres_axial_mm"][0] - hub["pressure_centre_offset_mm"]       # inboard row, DB: outward
    a2 = hub["row_centres_axial_mm"][1] + hub["pressure_centre_offset_mm"]
    x1, x2 = wc + s * a1, wc + s * a2
    ext = [(F, a) for F, a in loads] + [(P_disc, x_pad)]
    Fsum = sum(F for F, _ in ext)
    Msum = sum(np.cross(a - wc, F) for F, a in ext)
    cols = []
    for x in (x1, x2):
        for e in (u, r):
            cols.append(np.concatenate([e, np.cross(x - wc, e)]))
    cols.append(np.concatenate([s, np.zeros(3)]))
    A = np.array(cols).T
    sol, *_ = np.linalg.lstsq(A, -np.concatenate([Fsum, Msum]), rcond=None)
    B1, B2, Ax = sol[0] * u + sol[1] * r, sol[2] * u + sol[3] * r, sol[4] * s      # forces ON the hub
    hub_res = np.abs(A @ sol + np.concatenate([Fsum, Msum])).max()
    # ---- caliper: two lug bolts carry the pad friction (forces ON the caliper from the upright)
    r_bolt = veh["brake"]["disc_od_mm_assumed"] / 2 - cal["mount_below_disc_od_mm"]
    lug_t = json.loads((DATA / "upright_design.json").read_text())["caliper_mount"]["lug_t_mm"]
    mount = wc + s * (hub["disc_axial_mm"] - cal["mount_offset_mm"] - lug_t / 2)      # lug mid-thickness
    xb = [mount + r * r_bolt + t * sg * cal["bolt_spacing_mm"] / 2 for sg in (1.0, -1.0)]
    cols = []
    for x in xb:
        for e in (s, r, t):
            cols.append(np.concatenate([e, np.cross(x - wc, e)]))
    A = np.array(cols).T
    rhs = np.concatenate([P_disc, np.cross(x_pad - wc, P_disc)])     # bolts must balance -P_disc on the pads
    k, *_ = np.linalg.lstsq(A, rhs, rcond=None)                       # minimum-norm: equal shear split
    K = [k[0] * s + k[1] * r + k[2] * t, k[3] * s + k[4] * r + k[5] * t]
    # ---- upright balance (the check)
    on_up = [(link["UBJ"], at["UBJ"]), (link["LBJ"], at["LBJ"]), (link["TR"], at["TR"]),
             (-B1, x1), (-B2, x2), (-Ax, wc), (-K[0], xb[0]), (-K[1], xb[1])]
    resF = sum(F for F, _ in on_up)
    resM = sum(np.cross(a - wc, F) for F, a in on_up)
    scale = max(np.linalg.norm(F) for F, _ in on_up)
    resid = max(np.linalg.norm(resF), np.linalg.norm(resM) / 100.0) / scale      # moments per 100 mm lever
    return {"links": f, "UBJ": link["UBJ"], "LBJ": link["LBJ"], "TR": link["TR"], "row1": -B1, "row2": -B2,
            "axial": -Ax, "bolt1": -K[0], "bolt2": -K[1], "pad": P_disc, "points": {
                "UBJ": at["UBJ"], "LBJ": at["LBJ"], "TR": at["TR"], "row1": x1, "row2": x2, "axial": wc,
                "bolt1": xb[0], "bolt2": xb[1]}, "s": s, "resid": resid, "hub_resid": hub_res, "R": R,
            "c0": corner.c0, "q": pose["q"]}


def to_static(res):
    """Rotate the interface forces and points back into the upright's static (CAD) frame."""
    R, c0, q = res["R"], res["c0"], res["q"]
    out = {}
    for k in ("UBJ", "LBJ", "TR", "row1", "row2", "axial", "bolt1", "bolt2"):
        out[k] = {"force_N": (R.T @ res[k]).round(2).tolist(),
                  "point_mm": (c0 + R.T @ (res["points"][k] - c0 - q[3:])).round(3).tolist()}
    return out


def row_s0(row_r, axial, C0, X0, Y0):
    """Static safety of each row; the whole axial load goes on the more loaded row (conservative)."""
    fr = [np.linalg.norm(row_r[0]), np.linalg.norm(row_r[1])]
    fa = abs(axial)
    worst = int(np.argmax(fr))
    p0 = [max(fr[i], X0 * fr[i] + Y0 * (fa if i == worst else 0.0)) for i in range(2)]
    return fr, p0, [C0 / p if p > 0 else float("inf") for p in p0]


@stamps.producer("upright_loads")
def main():
    hp, meta = load_hardpoints()
    veh = load_vehicle()
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    members = read_csv_rows(DATA / "members.csv")
    lcs = read_csv_rows(DATA / "loadcases.csv")
    ver = read_version(DATA / "hardpoints.csv")
    C0 = veh["hub"]["C0_kN"] * 1000
    cases = []
    for lc in lcs:
        pose = corner.pose(float(lc["z_pose_mm"]))
        res = split(corner, pose, members, tyre_loads(corner, pose, lc, veh), veh, (lc.get("apply_at") or "cp") == "cp")
        cases.append((lc["id"], lc["name"], res))
    # LC6: a kerb pushes the front of the tyre sideways; scaled so the tie rod carries the rack force
    rack = json.loads((DATA / "vehicle_derived.json").read_text())["rack_force_driver_effort_N"]
    pose = corner.pose(0.0)
    fwd = unit(np.cross(unit(pose["P"]["WA"] - pose["P"]["WC"]), [0.0, 0.0, 1.0]))
    kerb = pose["cp"] + fwd * KERB_AHEAD_MM + np.array([0.0, 0.0, KERB_HEIGHT_MM])
    unit_load = [(np.array([0.0, 1.0, 0.0]), kerb)]
    f_tr = solve_links(pose, members, unit_load[0][0], {"Pushrod": "LCA_O"}, kerb)[0][4]
    lam = rack / abs(f_tr)
    res = split(corner, pose, members, [(np.array([0.0, lam, 0.0]), kerb)], veh, True)   # car held on the brake
    cases.append(("LC6", f"Steering abuse: kerb {lam:.0f} N sideways at the tyre front, tie rod {rack:.0f} N", res))

    out = {"_hardpoints": ver, "frame": "upright static frame = the CAD's frame (mm, N)",
           "spin_axis": cases[0][2]["s"].round(6).tolist(), "cases": {}}
    md = [f"# Upright interface loads (hardpoints {ver})", "",
          "Generated by `scripts/upright_loads.py`. Do not edit by hand. Forces on the upright, in the "
          f"upright's static frame. Bearing rows at their pressure centres ({veh['hub']['bearings']}, "
          f"{veh['hub']['row_centres_axial_mm']} mm from the wheel centre plane, +{veh['hub']['pressure_centre_offset_mm']} "
          "mm outward each); s0 = C0 / P0 per row with the whole axial load on the more loaded row.", "",
          "| LC | Upper BJ [N] | Lower BJ [N] | Tie rod [N] | Row 1 (inboard) [N] | Row 2 (outboard) [N] | Axial on bore [N] | "
          "Caliper bolt max [N] | Pad force [N] | min s0 | Balance residual |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    worst_s0 = (np.inf, "")
    for lcid, name, res in cases:
        st = to_static(res)
        ax_s = float(np.dot(res["axial"], res["s"]))
        fr, p0, s0 = row_s0((res["row1"], res["row2"]), ax_s, C0, veh["hub"]["X0"], veh["hub"]["Y0"])
        if min(s0) < worst_s0[0]:
            worst_s0 = (min(s0), lcid)
        if res["resid"] > 1e-3:
            raise RuntimeError(f"{lcid}: upright not in balance (residual {res['resid']:.2e})")
        out["cases"][lcid] = {"name": name, **st, "row_radial_N": [round(x, 1) for x in fr],
                              "axial_along_spin_N": round(ax_s, 1), "pad_force_N": round(float(np.linalg.norm(res["pad"])), 1),
                              "s0": [round(x, 2) for x in s0], "residual": res["resid"]}
        bolt = max(np.linalg.norm(res["bolt1"]), np.linalg.norm(res["bolt2"]))
        md.append(f"| {lcid} | {np.linalg.norm(res['UBJ']):.0f} | {np.linalg.norm(res['LBJ']):.0f} | "
                  f"{np.linalg.norm(res['TR']):.0f} | {fr[0]:.0f} | {fr[1]:.0f} | {ax_s:+.0f} | {bolt:.0f} | "
                  f"{np.linalg.norm(res['pad']):.0f} | {min(s0):.2f} | {res['resid']:.1e} |")
    md += ["", "Axial on bore: + pushes the bearing cups outboard, - inboard; either way one cup bears on the central rib "
           "in the upright bore.", "",
           f"**Worst bearing static safety: s0 = {worst_s0[0]:.2f} in {worst_s0[1]}** (SKF: s0 >= 1 for normal "
           "duty, 1.5-2 with shock loads).", "",
           "Each case balances to within 0.1 % of its largest interface load, which checks the split against "
           "the six-link solution in `forces.py`."]
    out["worst_s0"] = {"value": worst_s0[0], "lc": worst_s0[1]}
    (DATA / "upright_loads.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (NOTES / "upright_loads.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
