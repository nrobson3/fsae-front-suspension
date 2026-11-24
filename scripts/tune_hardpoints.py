"""Hardpoint tuning helpers (PLAN s5.1 'optionally optimise the hardpoints toward the targets').

Each stage moves only the points it owns, inside stated packaging bounds, and prints the
proposed coordinates. Nothing is written back automatically: proposals are copied into
data/hardpoints.csv by hand, rounded to 0.1 mm, and logged in notes/hardpoint_log.md.

  python tune_hardpoints.py arms      # wishbone inboard points + ball joints along kingpin
  python tune_hardpoints.py steer     # steering arm (TR_O) and rack (TR_I) for Ackermann + bump steer
  python tune_hardpoints.py bumpsteer # TR_I only
  python tune_hardpoints.py rocker    # rocker pushrod arm radius for motion ratio
"""
import math
import sys

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from common import load_hardpoints, load_vehicle, unit
from kinematics import enrich, required_lock
from suspension import Corner, SolveError, line_intersect_2d, mirror

VEH = load_vehicle()
VEH["_front_brake_fraction"] = VEH["front_weight_fraction"] + 1.5 * VEH["cg_height_mm"] / VEH["wheelbase_mm"]
R = VEH["tyre"]["loaded_radius_mm"]
T = VEH["track_front_mm"]
BIG = 1e6


def corner_of(hp):
    return Corner(hp, R)


# ------------------------------------------------------------------ stage: arms
def along_kingpin(hp, z_new, which):
    """Move a ball joint along the existing kingpin line to height z_new (keeps caster/KPI/scrub/trail)."""
    U, L = hp["UCA_O"], hp["LCA_O"]
    k = unit(U - L)
    base = L if which == "LCA_O" else U
    return base + (z_new - base[2]) / k[2] * k


def arm_hp(hp0, x):
    yLi, zLi, yUi, zUi, zLo, zUo = x
    hp = {k: v.copy() for k, v in hp0.items()}
    hp["LCA_O"] = along_kingpin(hp0, zLo, "LCA_O")
    hp["UCA_O"] = along_kingpin(hp0, zUo, "UCA_O")
    for k in ("LCA_F", "LCA_R"):
        hp[k][1], hp[k][2] = yLi, zLi
    for k in ("UCA_F", "UCA_R"):
        hp[k][1], hp[k][2] = yUi, zUi
    return hp


def arm_metrics(hp, zw=25.0, roll_deg=1.5):
    c = corner_of(hp)
    out = {}
    poses = {}
    for z in (-zw, 0.0, zw):
        poses[z] = enrich(c, c.pose(z), VEH)
    zr = T / 2 * math.tan(math.radians(roll_deg))
    pl, pr = enrich(c, c.pose(zr), VEH), enrich(c, c.pose(-zr), VEH)
    st = poses[0.0]
    out["rc0"] = st["rc_heave"]
    out["rc_mig"] = max(abs(poses[zw]["rc_heave"] - st["rc_heave"]), abs(poses[-zw]["rc_heave"] - st["rc_heave"]))
    cpL, cpR = pl["cp"], mirror(pr["cp"])
    u = unit((cpL - cpR)[1:])
    alpha = math.degrees(math.atan2(u[1], u[0]))
    rc = line_intersect_2d(cpL[1:], pl["fv_dir"], cpR[1:], pr["fv_dir"] * np.array([-1.0, 1.0]))
    out["rc_roll"] = float(np.dot(rc - cpR[1:], np.array([-u[1], u[0]])))
    out["recovery"] = -(pl["camber"] - st["camber"]) / alpha * 100.0
    out["fvsa"] = st["fvsa"]
    return out


def stage_arms(hp0):
    lo = np.array([180.0, 60.0, 190.0, 150.0, 108.0, 270.0])
    hi = np.array([240.0, 130.0, 270.0, 300.0, 125.0, 292.0])

    def cost(x):
        try:
            m = arm_metrics(arm_hp(hp0, x))
        except (SolveError, np.linalg.LinAlgError, ValueError):
            return BIG
        c = (m["rc_mig"] / 15.0) ** 2
        c += 4 * max(0.0, abs(m["recovery"] - 70.0) - 5.0) ** 2 / 25.0
        c += 4 * max(0.0, abs(m["rc0"] - 35.0) - 10.0) ** 2 / 25.0
        c += 0.5 * (m["rc_roll"] - m["rc0"]) ** 2 / 225.0
        # prefer a long kingpin (lower ball-joint loads)
        c += 0.02 * (292.0 - x[5] + x[4] - 108.0) / 10.0
        return c

    x0 = np.array([hp0["LCA_F"][1], hp0["LCA_F"][2], hp0["UCA_F"][1], hp0["UCA_F"][2], hp0["LCA_O"][2], hp0["UCA_O"][2]])
    print("start", np.round(x0, 1), arm_metrics(hp0), "cost", cost(x0))
    res = differential_evolution(cost, list(zip(lo, hi)), seed=1, tol=1e-6, maxiter=60, popsize=12, polish=True)
    x = res.x
    hp = arm_hp(hp0, x)
    print("best", np.round(x, 2), "cost", round(res.fun, 4))
    print("metrics", {k: round(v, 2) for k, v in arm_metrics(hp).items()})
    for k in ("LCA_F", "LCA_R", "UCA_F", "UCA_R", "LCA_O", "UCA_O"):
        print(f"  {k}: {np.round(hp[k], 2)}")
    return hp


# ------------------------------------------------------------------ stage: bump steer / steering
def toe_curve(hp, zs=np.linspace(-25, 25, 11)):
    c = corner_of(hp)
    toes = {}
    for z in sorted(zs, key=abs):
        p = c.pose(z, 0.0)
        toes[z] = -p["steer"]
    return np.array([toes[z] for z in zs])


def tune_bumpsteer(hp0, verbose=True):
    def resid(x):
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["TR_I"][1], hp["TR_I"][2] = x
        try:
            return toe_curve(hp) * 100.0
        except SolveError:
            return np.full(11, 1e3)

    x0 = np.array([hp0["TR_I"][1], hp0["TR_I"][2]])
    r = least_squares(resid, x0, diff_step=1e-4, xtol=1e-10)
    hp = {k: v.copy() for k, v in hp0.items()}
    hp["TR_I"][1], hp["TR_I"][2] = r.x
    if verbose:
        print("TR_I ->", np.round(hp["TR_I"], 2), "max |toe| over +/-25 =", round(np.abs(toe_curve(hp)).max(), 4), "deg")
    return hp


def ackermann_at_lock(hp):
    c = corner_of(hp)
    L, t = VEH["wheelbase_mm"], T
    d_req, _ = required_lock(VEH)

    def steer_at(s):
        return c.pose(0.0, s)["steer"]

    # find rack travel giving the outer (right) wheel the required angle: outer = -steer_L(-s)
    from scipy.optimize import brentq
    s = brentq(lambda s: -steer_at(-s) - d_req, 1.0, 40.0, xtol=1e-6)  # inside the validated +/-40 mm sweep
    d_in, d_out = steer_at(s), d_req
    ideal = math.degrees(math.atan(1.0 / (1.0 / math.tan(math.radians(d_out)) - t / L)))
    return (d_in - d_out) / (ideal - d_out) * 100.0, s, d_in


def stage_steer(hp0, target=50.0):
    """Choose steering-arm direction (TR_O y at fixed x, z) and rack x for ~target % Ackermann,
    re-solving TR_I for zero bump steer at each trial."""
    def build(x):
        y_to, x_ti = x
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["TR_O"][1] = y_to
        hp["TR_I"][0] = x_ti
        return tune_bumpsteer(hp, verbose=False)

    def cost(x):
        try:
            ack, s, _ = ackermann_at_lock(build(x))
        except Exception:
            return BIG
        return (ack - target) ** 2 + 0.05 * (x[1] - hp0["TR_O"][0]) ** 2 / 100.0

    x0 = np.array([hp0["TR_O"][1], hp0["TR_I"][0]])
    simplex = np.array([x0, x0 + [2.0, 0.0], x0 + [0.0, 5.0]])  # mm-scale first steps
    r = minimize(cost, x0, method="Nelder-Mead",
                 options={"xatol": 0.05, "fatol": 0.01, "maxiter": 200, "initial_simplex": simplex})
    hp = build(r.x)
    ack, s, d_in = ackermann_at_lock(hp)
    print(f"TR_O y={r.x[0]:.2f}, TR_I x={r.x[1]:.2f} -> Ackermann {ack:.1f} % at lock, rack {s:.2f} mm, inner {d_in:.2f} deg")
    print("  TR_O", np.round(hp["TR_O"], 2), " TR_I", np.round(hp["TR_I"], 2))
    print("  bump steer max", round(np.abs(toe_curve(hp)).max(), 4))
    return hp


# ------------------------------------------------------------------ stage: rocker
def stage_rocker(hp0, mr_target=0.92):
    """Scale the rocker pushrod arm (PR_I about RKR_P in the rocker plane) to hit a static motion ratio."""
    P = hp0["RKR_P"]
    arm0 = hp0["PR_I"] - P

    def mr_for(scale):
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["PR_I"] = P + arm0 * scale
        c = corner_of(hp)
        p = c.pose(0.0)
        return c.velocity(p)[3], hp

    from scipy.optimize import brentq
    sc = brentq(lambda s: mr_for(s)[0] - mr_target, 0.7, 1.6)
    mr, hp = mr_for(sc)
    print(f"pushrod arm {np.linalg.norm(arm0):.2f} -> {np.linalg.norm(arm0) * sc:.2f} mm, MR {mr:.4f}; PR_I = {np.round(hp['PR_I'], 2)}")
    return hp


def stage_rocker_shape(hp0, mr_target=0.92, prog_target=2.5, zw=25.0):
    """Rotate the pushrod arm about the rocker axis (angle a) and scale its radius (s) so the
    static motion ratio is mr_target and (MR(+zw) - MR(-zw)) / MR(0) is prog_target %."""
    from common import rotate_about
    P, ax = hp0["RKR_P"], unit(hp0["RKR_A"] - hp0["RKR_P"])
    arm0 = hp0["PR_I"] - P

    def build(x):
        a, s = x
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["PR_I"] = rotate_about(P + arm0 * s, P, ax, math.radians(a))
        return hp

    def mrs(hp):
        c = corner_of(hp)
        return [c.velocity(c.pose(z))[3] for z in (-zw, 0.0, zw)]

    def resid(x):
        try:
            lo, m0, hi = mrs(build(x))
        except SolveError:
            return np.array([1e3, 1e3])
        return np.array([(m0 - mr_target) * 100.0, (hi - lo) / m0 * 100.0 - prog_target])

    r = least_squares(resid, np.array([0.0, 1.0]), diff_step=1e-2, xtol=1e-10)
    hp = build(r.x)
    lo, m0, hi = mrs(hp)
    print(f"pushrod arm rotated {r.x[0]:.2f} deg, radius x{r.x[1]:.4f} -> MR(-{zw:.0f}) {lo:.4f}, MR(0) {m0:.4f}, "
          f"MR(+{zw:.0f}) {hi:.4f}; PR_I = {np.round(hp['PR_I'], 2)}")
    return hp


def stage_pushrod_on_lca(hp0, mr_target=0.93, prog_target=2.0, d_node=35.0, zw=25.0):
    """v1.3: pushrod picks up on the lower-wishbone node. PR_O sits d_node mm from the lower ball
    joint ON the line LCA_O -> PR_I, so the pushrod force passes through the ball-joint centre
    (no bending moment at the node at static). The rocker pushrod arm (angle, radius) is tuned
    for the motion-ratio target; PR_O is re-derived for every candidate PR_I."""
    from common import rotate_about
    P, ax = hp0["RKR_P"], unit(hp0["RKR_A"] - hp0["RKR_P"])
    arm0 = hp0["PR_I"] - P

    def build(x):
        a, s = x
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["PR_I"] = rotate_about(P + arm0 * s, P, ax, math.radians(a))
        hp["PR_O"] = hp["LCA_O"] + d_node * unit(hp["PR_I"] - hp["LCA_O"])
        return hp

    def mrs(hp):
        c = Corner(hp, R, pushrod_on="lca")
        return [c.velocity(c.pose(z))[3] for z in (-zw, 0.0, zw)]

    def resid(x):
        try:
            lo, m0, hi = mrs(build(x))
        except SolveError:
            return np.array([1e3, 1e3])
        return np.array([(m0 - mr_target) * 100.0, (hi - lo) / m0 * 100.0 - prog_target])

    r = least_squares(resid, np.array([0.0, 1.0]), diff_step=1e-2, xtol=1e-10)
    hp = build(r.x)
    lo, m0, hi = mrs(hp)
    print(f"pushrod arm rotated {r.x[0]:.2f} deg, radius x{r.x[1]:.4f} -> MR(-{zw:.0f}) {lo:.4f}, MR(0) {m0:.4f}, "
          f"MR(+{zw:.0f}) {hi:.4f}\n  PR_I = {np.round(hp['PR_I'], 3)}\n  PR_O = {np.round(hp['PR_O'], 3)}")
    return hp


def stage_arms_rim(hp0, half_w=88.9, rack_stop=25.8, min_clear=6.0, seed=4, maxiter=60, y_lca_min=195.0, z_lo_max=132.0):
    """Re-optimise both wishbones WITH a rim-clearance constraint (found by the CAD interference
    sweep): the legs must stay min_clear mm inside the rim barrel over heave x full steering
    lock. Variables: LCA/UCA inboard y and z, LCA leg spread in x, both ball-joint heights (along
    the kingpin, so caster/KPI/scrub/trail are unchanged). PR_O is re-derived on the LCA node."""
    import rim_clearance as RCL
    lo = np.array([y_lca_min, 60.0, 190.0, 150.0, 108.0, 268.0, 150.0])
    hi = np.array([240.0, 130.0, 290.0, 300.0, z_lo_max, 292.0, 220.0])
    racks = (-rack_stop, -0.8 * rack_stop, 0.8 * rack_stop, rack_stop)

    def build(x):
        yLi, zLi, yUi, zUi, zLo, zUo, sL = x
        hp = arm_hp(hp0, [yLi, zLi, yUi, zUi, zLo, zUo])
        hp["LCA_F"][0], hp["LCA_R"][0] = sL, -sL
        hp["PR_O"] = hp["LCA_O"] + 35.0 * unit(hp["PR_I"] - hp["LCA_O"])
        return hp

    def cost(x):
        try:
            hp = build(x)
            m = arm_metrics(hp)
            w = RCL.sweep(hp, {"PR_O": {"body": "lca"}}, half_w, racks=racks, zs=(-30.0, 30.0))
        except (SolveError, np.linalg.LinAlgError, ValueError):
            return BIG
        c = 4 * max(0.0, abs(m["recovery"] - 70.0) - 7.0) ** 2 / 25.0
        c += 4 * max(0.0, abs(m["rc0"] - 35.0) - 12.0) ** 2 / 25.0
        c += 0.5 * (m["rc_roll"] - m["rc0"]) ** 2 / 225.0
        for k in ("lca_F", "lca_R", "uca_F", "uca_R"):
            c += 10.0 * max(0.0, min_clear - w[k][0]) ** 2 / 9.0
        c += 2.0 * max(0.0, 185.0 - x[6]) ** 2 / 400.0          # narrow LCA spread raises leg loads
        c += 0.02 * (292.0 - x[5] + x[4] - 108.0) / 10.0       # prefer a long kingpin
        return c

    res = differential_evolution(cost, list(zip(lo, hi)), seed=seed, tol=1e-6, maxiter=maxiter, popsize=12, polish=True)
    hp = build(res.x)
    m = arm_metrics(hp)
    w = RCL.sweep(hp, {"PR_O": {"body": "lca"}}, half_w)
    print(f"rim half width {half_w}: cost {res.fun:.3f}, x = {np.round(res.x, 1)}")
    print("  metrics", {k: round(float(v), 1) for k, v in m.items()})
    print("  rim clearance (full sweep)", {k: round(v[0], 1) for k, v in w.items()})
    for k in ("LCA_F", "LCA_R", "UCA_F", "UCA_R", "LCA_O", "UCA_O", "PR_O"):
        print(f"  {k}: {np.round(hp[k], 2)}")
    return hp


def stage_steer_rim(hp0, half_w=76.2, ack_target=45.0, min_clear=6.0):
    """Steering arm (TR_O y, z) and rack x chosen for an Ackermann target while the tie rod clears
    the rim lip over heave x full lock and both wishbones; TR_I y, z re-solved for zero bump steer
    at every trial."""
    import rim_clearance as RCL

    def seg_gap(a1, a2, b1, b2, n=400):
        t = np.linspace(0, 1, n)[:, None]
        A, B = a1 + t * (a2 - a1), b1 + t * (b2 - b1)
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min()

    def build(x):
        y_to, z_to, x_ti = x
        hp = {k: v.copy() for k, v in hp0.items()}
        hp["TR_O"][1], hp["TR_O"][2] = y_to, z_to
        hp["TR_I"][0] = x_ti
        hp["TR_I"][2] = hp0["TR_I"][2] + (z_to - hp0["TR_O"][2])  # start the bump-steer solve nearby
        return tune_bumpsteer(hp, verbose=False)

    def evaluate(hp):
        ack, s, _ = ackermann_at_lock(hp)
        stop = s * 1.09
        c = Corner(hp, R, pushrod_on="lca")
        w = RCL.sweep(hp, {"PR_O": {"body": "lca"}}, half_w, racks=(-stop, -0.8 * stop, 0.8 * stop, stop), zs=(-30.0, 30.0))
        gaps = []
        for z in (-30.0, 0.0, 30.0):
            for r in (-stop, 0.0, stop):
                P = c.pose(z, r)["P"]
                for leg in ("LCA_F", "LCA_R", "UCA_F", "UCA_R"):
                    o = "LCA_O" if leg.startswith("LCA") else "UCA_O"
                    gaps.append(seg_gap(P["TR_I"], P["TR_O"], P[leg], P[o]) - 15.875)
        return ack, w["tie"][0], min(gaps), s

    def cost(x):
        try:
            ack, rim, legs, s = evaluate(build(x))
        except Exception:
            return BIG
        c = (ack - ack_target) ** 2 / 4.0
        c += 10.0 * max(0.0, min_clear - rim) ** 2 / 4.0
        c += 10.0 * max(0.0, 8.0 - legs) ** 2 / 4.0
        return c

    x0 = np.array([hp0["TR_O"][1], hp0["TR_O"][2], hp0["TR_I"][0]])
    simplex = np.array([x0, x0 + [3.0, 0.0, 0.0], x0 + [0.0, 8.0, 0.0], x0 + [0.0, 0.0, 15.0]])
    r = minimize(cost, x0, method="Nelder-Mead", options={"xatol": 0.1, "fatol": 0.01, "maxiter": 120, "initial_simplex": simplex})
    hp = build(r.x)
    ack, rim, legs, s = evaluate(hp)
    print(f"TR_O y={r.x[0]:.2f} z={r.x[1]:.2f}, TR_I x={r.x[2]:.2f}: Ackermann {ack:.1f} %, tie-rod rim clearance {rim:.1f} mm, "
          f"tie rod to wishbone legs {legs:.1f} mm, rack at lock {s:.2f} mm")
    print("  TR_O", np.round(hp["TR_O"], 2), " TR_I", np.round(hp["TR_I"], 2), " bump steer", round(np.abs(toe_curve(hp)).max(), 4))
    return hp


if __name__ == "__main__":
    hp, _ = load_hardpoints()
    stage = sys.argv[1] if len(sys.argv) > 1 else "arms"
    {"arms": stage_arms, "steer": stage_steer, "bumpsteer": tune_bumpsteer, "rocker": stage_rocker,
     "rockershape": stage_rocker_shape}[stage](hp)
