"""Member forces for LC1-LC4 and the hand-calc checks (PLAN s4).

The upright + hub + wheel (outboard brakes) is one free body held by six two-force members:
UCA front/rear legs, LCA front/rear legs, tie rod, pushrod. The contact-patch force from
data/loadcases.csv acts at the contact patch, so brake torque is reacted inside the corner
(caliper -> upright) and appears in the link forces. Solve A f = b (6 x 6), tension positive.

Each load case is solved at its loaded pose (wheel travel from vehicle_calcs.py) and at the
static pose; the checks keep the tension and compression extremes of every member across all
load cases and poses. Column capacity is Euler or Johnson by slenderness. An independent
virtual-work check of the pushrod force uses only the kinematic solver. Unsprung weight and
inertia are not subtracted from the contact-patch force (conservative, about 15 % in LC1).

Run: python forces.py      Writes data/member_forces.csv, notes/force_table.md, plots/f01_member_forces.png
"""
import csv
import json
import math

import numpy as np

import stamps
from common import DATA, NOTES, PLOTS, INK_2, SERIES, figure, load_hardpoints, load_vehicle, read_version, style_axes, unit
from common import rotvec_matrix
from suspension import make_corner, point_velocity

FOS_YIELD, FOS_BUCKLE, FOS_HARDWARE = 1.5, 2.0, 1.5
# Catalogue ratings given only as ULTIMATE are held to 1.5 x 1.5 = 2.25: the 1.5 design factor on
# top of the usual 1.5 ultimate-to-limit factor, so they carry the same margin as a yield check.
FOS_ULTIMATE = FOS_HARDWARE * 1.5


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader([l for l in f if not l.lstrip().startswith("#")]))


def solve_links(pose, members, F, acts_at=None, at=None):
    """6x6 upright equilibrium for an external force F applied at point `at` (default: the
    contact patch). acts_at maps a member name to the hardpoint where its force enters the
    upright (default: its outboard end). A pushrod on the lower-wishbone node enters through the
    lower ball joint, along its own line."""
    acts_at = acts_at or {}
    P = pose["P"]
    c, cp = P["WC"], (pose["cp"] if at is None else at)
    A = np.zeros((6, 6))
    for j, m in enumerate(members):
        u = unit(P[m["in"]] - P[m["out"]])
        A[:3, j] = u
        A[3:, j] = np.cross(P[acts_at.get(m["member"], m["out"])] - c, u)
    b = -np.concatenate([F, np.cross(cp - c, F)])
    f = np.linalg.solve(A, b)
    resid = np.abs(A @ f - b).max()
    return f, np.linalg.cond(A), resid


def rocker_and_damper(pose, corner, f_push):
    """Damper force (tension +) from rocker moment balance about its axis (ARB link unloaded)."""
    P, a = pose["P"], corner.rk_axis
    piv = P["RKR_P"]
    u_p = unit(P["PR_I"] - P["PR_O"])
    F_push_on_rocker = -f_push * u_p
    e_d = unit(P["DMP_C"] - P["DMP_R"])
    m_push = np.dot(np.cross(P["PR_I"] - piv, F_push_on_rocker), a)
    m_unit = np.dot(np.cross(P["DMP_R"] - piv, e_d), a)
    f_d = -m_push / m_unit
    F_pivot = -(F_push_on_rocker + f_d * e_d)
    return f_d, F_pivot


def virtual_work_pushrod(corner, pose, F, at=None):
    """f_push = F . v_at / (dL_push/dz) with the pushrod removed (1-DOF mechanism). The pickup
    velocity comes from whichever body carries it (upright or lower wishbone)."""
    w, v_c, c, _, _ = corner.velocity(pose)
    v_cp = point_velocity(w, v_c, c, pose["cp"] if at is None else at)
    P = pose["P"]
    u_p = unit(P["PR_I"] - P["PR_O"])
    v_po = corner.point_velocities(pose)["PR_O"]
    dLp = -np.dot(u_p, v_po)  # pushrod-length rate if the rocker end were held
    return np.dot(F, v_cp) / dLp


def pushrod_line_offset(pose):
    """Distance [mm] from the lower ball-joint centre to the pushrod line (0 by design at static)."""
    P = pose["P"]
    u_p = unit(P["PR_I"] - P["PR_O"])
    d = P["LCA_O"] - P["PR_O"]
    return float(np.linalg.norm(d - np.dot(d, u_p) * u_p))


def load_point(corner, pose, lc, tread_mm):
    """Where a load case's force acts: contact patch, wheel centre, or a kerb point on the outer
    sidewall (fixed to the non-spinning wheel/upright)."""
    kind = lc.get("apply_at", "cp") or "cp"
    if kind == "cp":
        return pose["cp"]
    if kind == "wc":
        return pose["P"]["WC"]
    if kind == "kerb":
        k0 = np.array([0.0, 610.0 + tread_mm / 2, float(lc["kerb_height_mm"])])
        q = pose["q"]
        return corner.c0 + rotvec_matrix(q[:3]) @ (k0 - corner.c0) + q[3:]
    raise ValueError(kind)


def tyre_loads(corner, pose, lc, veh):
    """[(force N, point mm)] the ground or obstacle applies to the tyre in load case `lc` at `pose`.
    At the contact patch the lateral force acts a pneumatic trail behind it (the tyre's aligning
    moment, decision D12), so a contact-patch case with a side force is two point loads."""
    F = np.array([float(lc["Fx_N"]), float(lc["Fy_N"]), float(lc["Fz_N"])])
    at = load_point(corner, pose, lc, veh["tyre"]["tread_width_mm"])
    if (lc.get("apply_at") or "cp") != "cp" or abs(F[1]) == 0:
        return [(F, at)]
    spin = unit(pose["P"]["WA"] - pose["P"]["WC"])
    fwd = unit(np.cross(spin, [0.0, 0.0, 1.0]))
    return [(np.array([F[0], 0.0, F[2]]), at),
            (np.array([0.0, F[1], 0.0]), at - fwd * veh["tyre"]["pneumatic_trail_mm_assumed"])]


def stepped_column_pcr(L, E, I_tube, I_end, l_end, n=60):
    """Elastic buckling load of a pinned-pinned column whose end segments (rod-end shanks,
    length l_end, inertia I_end) are slenderer than the tube. Euler-Bernoulli beam FE with
    geometric stiffness; returns the lowest eigenvalue."""
    from scipy.linalg import eigh
    x = np.linspace(0, L, n + 1)
    K = np.zeros((2 * n + 2, 2 * n + 2))
    Kg = np.zeros_like(K)
    for e in range(n):
        le = x[e + 1] - x[e]
        xm = 0.5 * (x[e] + x[e + 1])
        I = I_end if (xm < l_end or xm > L - l_end) else I_tube
        k = E * I / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                        [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        g = 1 / (30 * le) * np.array([[36, 3 * le, -36, 3 * le], [3 * le, 4 * le ** 2, -3 * le, -le ** 2],
                                      [-36, -3 * le, 36, -3 * le], [3 * le, -le ** 2, -3 * le, 4 * le ** 2]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k
        Kg[np.ix_(idx, idx)] += g
    free = [i for i in range(2 * n + 2) if i not in (0, 2 * n)]  # pinned: no lateral displacement at the ends
    vals = eigh(K[np.ix_(free, free)], Kg[np.ix_(free, free)], eigvals_only=True)
    return float(vals[vals > 0].min())


def column_capacity(A, I, L, E, sy):
    """Pinned-pinned column capacity. Euler for slender members; Johnson parabola once the Euler
    stress exceeds sy/2 (L/r below the transition sqrt(2 pi^2 E / sy)), where Euler is unconservative."""
    r = math.sqrt(I / A)
    sigma_e = math.pi ** 2 * E / (L / r) ** 2
    if sigma_e <= sy / 2:
        return A * sigma_e, "Euler"
    return A * sy * (1 - sy / (4 * sigma_e)), "Johnson"


def section(od, wall):
    if not 0 < wall <= od / 2:  # wall = OD/2 is a solid bar
        raise ValueError(f"tube wall {wall} must be > 0 and <= OD/2 ({od / 2})")
    di = od - 2 * wall
    A = math.pi / 4 * (od ** 2 - di ** 2)
    I = math.pi / 64 * (od ** 4 - di ** 4)
    return A, I


@stamps.producer("forces")
def main():
    hp, hp_meta = load_hardpoints()
    veh = load_vehicle()
    mats = json.loads((DATA / "materials.json").read_text())
    members = read_csv_rows(DATA / "members.csv")
    lcs = read_csv_rows(DATA / "loadcases.csv")
    corner = make_corner(hp, hp_meta, veh["tyre"]["loaded_radius_mm"])
    acts_at = {"Pushrod": "LCA_O"} if corner.pushrod_on == "lca" else {}
    version = read_version(DATA / "hardpoints.csv")

    ip = [m["member"] for m in members].index("Pushrod")
    rows, extra, vw = [], [], []
    for lc in lcs:
        F = np.array([float(lc["Fx_N"]), float(lc["Fy_N"]), float(lc["Fz_N"])])
        f_poses = []
        for tag, z in (("static", 0.0), ("loaded", float(lc["z_pose_mm"]))):
            pose = corner.pose(z)
            loads = tyre_loads(corner, pose, lc, veh)
            (F_main, at), *rest = loads
            F_lat, at_lat = rest[0] if rest else (None, None)
            f, cond, resid = np.zeros(len(members)), 0.0, 0.0
            for Fi, ati in loads:                    # the link solve is linear: one solve per load, summed
                fi, ci, ri = solve_links(pose, members, Fi, acts_at, ati)
                f, cond, resid = f + fi, max(cond, ci), max(resid, ri)
            if resid > 1e-6 or cond > 1e4:
                raise RuntimeError(f"{lc['id']} {tag}: link system ill-posed (residual {resid:.1e}, cond {cond:.0f})")
            f_d, F_piv = rocker_and_damper(pose, corner, f[ip])
            P = pose["P"]
            bj_u = np.linalg.norm(f[0] * unit(P["UCA_F"] - P["UCA_O"]) + f[1] * unit(P["UCA_R"] - P["UCA_O"]))
            F_bj_l = f[2] * unit(P["LCA_F"] - P["LCA_O"]) + f[3] * unit(P["LCA_R"] - P["LCA_O"])
            if acts_at.get("Pushrod") == "LCA_O":  # the node passes the pushrod load through the ball joint too
                F_bj_l = F_bj_l + f[ip] * unit(P["PR_I"] - P["PR_O"])
            bj_l = np.linalg.norm(F_bj_l)
            fv = virtual_work_pushrod(corner, pose, F_main, at)
            if F_lat is not None:
                fv += virtual_work_pushrod(corner, pose, F_lat, at_lat)
            vw.append((lc["id"], tag, f[ip], fv))
            off = pushrod_line_offset(pose) if acts_at else float("nan")
            rec = {"lc": lc["id"], "pose": tag, "z": z, "f": f, "cond": cond, "resid": resid, "damper": f_d,
                   "rocker_pivot": np.linalg.norm(F_piv), "bj_upper": bj_u, "bj_lower": bj_l,
                   "push_offset": off, "node_moment": abs(f[ip]) * off / 1000.0}
            extra.append(rec)
            f_poses.append(f)
        f_poses = np.array(f_poses)
        env = f_poses[np.argmax(np.abs(f_poses), axis=0), np.arange(len(members))]  # larger |f|, for display
        for j, m in enumerate(members):
            rows.append({"lc": lc["id"], "member": m["member"], "force": env[j],
                         "fmax": f_poses[:, j].max(), "fmin": f_poses[:, j].min()})

    # LC6: steering effort with the wheels blocked puts the whole rack force on one tie rod
    derived = json.loads((DATA / "vehicle_derived.json").read_text())
    f_rack = derived["rack_force_driver_effort_N"]
    rows.append({"lc": "LC6", "member": "Tie rod", "force": -f_rack, "fmax": f_rack, "fmin": -f_rack})

    # ---- checks (tension and compression extremes kept separately across every LC and pose)
    table = []
    for j, m in enumerate(members):
        od, wall = float(m["tube_od_mm"]), float(m["tube_wall_mm"])
        A, I = section(od, wall)
        mat = mats[m["tube_material"]]
        L = np.linalg.norm(hp[m["in"]] - hp[m["out"]])
        allow = mat["Fty_MPa"] * mat.get("weld_knockdown", 1.0)
        Pcr, mode = column_capacity(A, I, L, mat["E_MPa"], allow)
        # rod-end shanks are slenderer than the tube: stepped-column elastic load, then the same
        # Johnson cut-off on the equivalent Euler stress
        d3 = float(m["bolt_d_mm"]) - 1.22687 * float(m["rod_end_pitch_mm"])
        I_end = math.pi / 64 * d3 ** 4
        Pe_step = stepped_column_pcr(L, mat["E_MPa"], I, I_end, float(m["rod_end_shank_mm"]))
        I_eq = Pe_step * L ** 2 / (math.pi ** 2 * mat["E_MPa"])     # uniform column with the same Euler load
        Pcr_step, mode_step = column_capacity(A, I_eq, L, mat["E_MPa"], allow)
        mine = [r for r in rows if r["member"] == m["member"]]
        f_t = max(max(r["fmax"] for r in mine), 0.0)
        f_c = min(min(r["fmin"] for r in mine), 0.0)
        f_abs = max(abs(f_t), abs(f_c))
        lc_abs = max(mine, key=lambda r: max(abs(r["fmax"]), abs(r["fmin"])))["lc"]
        sigma = f_abs / A
        bolt = mats[m["bolt_material"]]
        d, n = float(m["bolt_d_mm"]), int(m["shear_planes"])
        d_root = d - 1.22687 * float(m["bolt_pitch_mm"])           # ISO 724 minor diameter d3
        tau = f_abs / (n * math.pi * d_root ** 2 / 4)                # threads assumed in the shear plane
        tab = mats[m["tab_material"]]
        brg = f_abs / (n * d * float(m["tab_t_mm"]))
        fos_re = float(m["rod_end_ult_N"]) / f_abs
        table.append({
            "member": m["member"], "length_mm": L, "tube": f"{od:.2f} x {wall:.3f}", "max_tension_N": f_t,
            "max_compression_N": f_c, "critical_lc": lc_abs, "stress_MPa": sigma,
            "fos_yield": allow / sigma, "fos_yield_fsae_generic": mat["fsae_generic_welded_Sy_MPa"] / sigma,
            "slenderness_L_r": L / math.sqrt(I / A), "buckling_mode": mode,
            "Pcr_N": Pcr, "fos_buckling": Pcr / abs(f_c) if f_c < 0 else float("inf"),
            "Pcr_stepped_N": Pcr_step, "fos_buckling_stepped": Pcr_step / abs(f_c) if f_c < 0 else float("inf"),
            "fos_rod_end": fos_re, "fos_bolt_shear": bolt["Fty_MPa"] * bolt["shear_yield_factor"] / tau,
            "fos_tab_bearing": tab["Fty_MPa"] * tab["bearing_allow_factor"] / brg,
        })

    # ---- outputs
    with open(DATA / "member_forces.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# hardpoints", version, "forces in N, tension +; envelope of static and loaded pose"])
        w.writerow(["lc"] + [m["member"] for m in members])
        for lc in lcs:
            w.writerow([lc["id"]] + [f"{r['force']:.1f}" for r in rows if r["lc"] == lc["id"]])
        w.writerow(["LC6"] + [f"+/-{f_rack:.1f}" if m["member"] == "Tie rod" else "" for m in members])
    with open(DATA / "member_checks.csv", "w", newline="") as f:
        keys = list(table[0].keys())
        w = csv.writer(f)
        w.writerow(["# hardpoints", version])
        w.writerow(keys)
        for t in table:
            w.writerow([f"{t[k]:.3f}" if isinstance(t[k], float) else t[k] for k in keys])

    def fos(v, need):
        if v != v:
            return "TBC"
        if v == float("inf"):
            return "n/a (tension)"
        return f"{v:.2f} {'PASS' if v >= need else 'FAIL'}"

    md = [f"# Member forces and hand-calc checks (hardpoints {version})", "",
          "Generated by `scripts/forces.py`. Do not edit by hand. Forces in N, tension positive. "
          "Each load case is solved at its loaded pose and at static; the table shows the larger |force|, "
          "the checks use the tension and compression extremes over both poses.", "",
          "## Member forces", "", "| LC | " + " | ".join(m["member"] for m in members) + " |",
          "|---|" + "---|" * len(members)]
    for lc in lcs:
        md.append(f"| {lc['id']} | " + " | ".join(f"{r['force']:.0f}" for r in rows if r["lc"] == lc["id"]) + " |")
    md.append("| LC6 (steering effort, wheels blocked) | " +
              " | ".join(f"+/-{f_rack:.0f}" if m["member"] == "Tie rod" else "-" for m in members) + " |")
    md += ["", "## Checks (worst load case per member)", "",
           f"Targets: yield FoS >= {FOS_YIELD} (tube yield x 0.8 weld knock-down = "
           f"{mats['4130N_tube']['Fty_MPa'] * mats['4130N_tube']['weld_knockdown']:.0f} MPa), buckling FoS >= {FOS_BUCKLE} "
           f"(pinned-pinned, pin-to-pin length; Euler, or Johnson below the transition slenderness; also with the "
           f"slenderer rod-end shanks as a stepped column), rod end >= {FOS_ULTIMATE} on its ULTIMATE rating, "
           f"bolt shear (yield) and tab bearing (yield) >= {FOS_HARDWARE}. "
           "'FoS vs FSAE generic' uses the rulebook's 180 MPa welded-steel yield (F.3.4.2) as a worst-case sensitivity, "
           "not a requirement.", "",
           "| Member | L [mm] | Tube OD x t [mm] | Max T [N] | Max C [N] | Critical LC | Stress [MPa] | FoS yield | "
           "FoS vs FSAE generic | L/r | P_cr [N] | FoS buckling | FoS buckling incl. rod-end shanks | FoS rod end (ult) | "
           "FoS bolt shear (thread root) | FoS tab bearing |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['member']} | {t['length_mm']:.0f} | {t['tube']} | {t['max_tension_N']:.0f} | {t['max_compression_N']:.0f} | "
                  f"{t['critical_lc']} | {t['stress_MPa']:.1f} | {fos(t['fos_yield'], FOS_YIELD)} | "
                  f"{t['fos_yield_fsae_generic']:.2f} | {t['slenderness_L_r']:.0f} | {t['Pcr_N']:.0f} ({t['buckling_mode']}) | "
                  f"{fos(t['fos_buckling'], FOS_BUCKLE)} | {fos(t['fos_buckling_stepped'], FOS_BUCKLE)} | "
                  f"{fos(t['fos_rod_end'], FOS_ULTIMATE)} | "
                  f"{fos(t['fos_bolt_shear'], FOS_HARDWARE)} | {fos(t['fos_tab_bearing'], FOS_HARDWARE)} |")
    joints = read_csv_rows(DATA / "joints.csv")
    md += ["", "## Outboard ball joints (worst pose and load case)", "",
           f"Spherical bearing vs its radial static LIMIT rating >= {FOS_HARDWARE}; AN bolt in double shear vs its "
           f"ULTIMATE single-shear rating x planes >= {FOS_ULTIMATE}.", "",
           "| Joint | Max load [N] | Critical LC | Bearing | FoS bearing | Bolt | FoS bolt |", "|---|---|---|---|---|---|---|"]
    joint_rows = []
    for jt in joints:
        worst = max(extra, key=lambda e: e[jt["load_key"]])
        Fj = worst[jt["load_key"]]
        fb, fbolt = float(jt["bearing_rating_N"]) / Fj, float(jt["bolt_shear_ult_single_N"]) * int(jt["shear_planes"]) / Fj
        joint_rows.append({"joint": jt["joint"], "max_load_N": Fj, "critical_lc": worst["lc"], "fos_bearing": fb, "fos_bolt": fbolt})
        md.append(f"| {jt['joint']} | {Fj:.0f} | {worst['lc']} ({worst['pose']}) | {jt['bearing']} | "
                  f"{fos(fb, FOS_HARDWARE)} | {jt['bolt']} x{jt['shear_planes']} | {fos(fbolt, FOS_ULTIMATE)} |")
    with open(DATA / "joint_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# hardpoints", version])
        w.writerow(list(joint_rows[0].keys()))
        for r in joint_rows:
            w.writerow([f"{v:.3f}" if isinstance(v, float) else v for v in r.values()])
    md += ["", "## Other reactions (per pose)", "",
           "Pushrod offset = distance from the lower ball-joint centre to the pushrod line (0 at static by "
           "design); node moment = pushrod force x offset, the bending the lower-wishbone node must carry.", "",
           "| LC | Pose | z [mm] | Damper force [N] (ARB link unloaded: upper bound in LC2/LC4) | Rocker pivot [N] | "
           "Upper ball joint [N] | Lower ball joint [N] | Pushrod offset [mm] | Node moment [N m] | cond(A) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for e in extra:
        md.append(f"| {e['lc']} | {e['pose']} | {e['z']:.1f} | {e['damper']:.0f} | {e['rocker_pivot']:.0f} | "
                  f"{e['bj_upper']:.0f} | {e['bj_lower']:.0f} | {e['push_offset']:.2f} | {e['node_moment']:.2f} | {e['cond']:.1f} |")
    md += ["", "## Independent check: pushrod force by virtual work", "",
           "Pushrod removed, the corner becomes a 1-DOF mechanism; f = F . v_cp / (dL_pushrod / dz) "
           "using only the kinematic solver.", "", "| LC | Pose | 6x6 solve [N] | Virtual work [N] | Difference |", "|---|---|---|---|---|"]
    for lc_id, tag, f6, fv in vw:
        md.append(f"| {lc_id} | {tag} | {f6:.1f} | {fv:.1f} | {abs(f6 - fv) / max(abs(f6), 1e-9) * 100:.3f} % |")
    (NOTES / "force_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- plot: member forces per load case
    fig, axs = figure(1, 1, w=8.0, h=3.8)
    ax = axs[0][0]
    names = [m["member"] for m in members]
    x = np.arange(len(names))
    wbar = 0.18
    for i, lc in enumerate(lcs):
        vals = [r["force"] / 1000 for r in rows if r["lc"] == lc["id"]]
        ax.bar(x + (i - 1.5) * (wbar + 0.02), vals, width=wbar, color=SERIES[i], label=f"{lc['id']} {lc['name']}")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, names, fontsize=8)
    style_axes(ax, f"Member axial forces, tension + ({version})", None, "force [kN]")
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(PLOTS / "f01_member_forces.png", dpi=160, facecolor=fig.get_facecolor())

    print("\n".join(md))


if __name__ == "__main__":
    main()
