"""Navy's hand-calculation packets: seven problems to work on paper, with inputs taken from the current
design, and a separate answer key computed here by closed-form methods (no FEA, no 3D solver in the
method; each key compares its answer with the solver or FEA where one exists).

  HC-1  LC1 (3 g bump) in the front view: wishbone, ball-joint and pushrod forces
  HC-2  Rocker in LC1 (bump-stop force included): damper force, pivot load, plate stresses     (S-06e)
  HC-3  Pushrod bracket knee: offset moment, curved-beam stress at the inside radius
  HC-4  Hub spindle in rotating bending: shoulder fillet Kt, Miner over the duty spectrum         (S-15)
  HC-5  ARB bar: the stiff bar's body and its splined ends in torsion                       (S-09)
  HC-6  Steering arm: beam theory near the bolt (why it misses) and at the root with Kt (DV-2)
  HC-7  Ear roots the CAD kernel could not fillet: plate bending with Kt, lug checks, fatigue    (S-14)
  HC-8  Brake heat into the upright: disc temperature, the upright's heat budget, the mount     (S-16)
  HC-9  Member-end welds and rod-end threads in fatigue; thread stripping                      (S-07b)
  HC-10 Steering free play at the wheel                                                        (R-06)
  HC-11 Chassis tabs and brackets as cantilevers from their welds                             (S-19)
  HC-12 The drop link's lever end: single-shear bolt bending; the ARB lever at its hub       (S-20, S-21)
  (HC-8 to HC-12 are computed in week3b_checks.py, inputs in data/checks_week3b.json and data/chassis_tabs.json)

Stress concentration: shoulder fillets use the curve fit Kt = A (r/d)^b to Peterson's chart for a stepped
round bar in bending (Shigley, Table A-15-9), taking the largest fit among the tabulated D/d up to the next one
above the real one; flat plates get that value x 1.1 (flat bars run a little higher than round ones). Kf = Kt (notch sensitivity 1):
conservative. The paper version reads the charts directly, which checks the fit.

Run: python handcalcs.py   Writes notes/handcalcs/HC-1..HC-12 (problems), notes/handcalcs/answer_key.md and
data/handcalc_results.json (read by the requirements matrix for S-06e, S-14, S-15, S-16, S-07b, R-06, S-19, S-20 and S-21).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import fingerprint
import forces
import handcalc_lc1
import week3b_checks as W3
from common import DATA, NOTES, ROOT, load_hardpoints, load_materials, load_vehicle, read_version, unit
from suspension import make_corner

sys.path.insert(0, str(Path(__file__).resolve().parent / "fea"))
from upright_fatigue import margin, sn_line  # noqa: E402

OUT = NOTES / "handcalcs"
TARGET = 1.5
# stepped round bar in BENDING, Kt = A (r/d)^b (Shigley Table A-15-9 fit to Peterson's chart)
KT_FIT = {1.01: (0.91938, -0.17032), 1.02: (0.96048, -0.17711), 1.03: (0.98061, -0.18381),
          1.05: (0.98137, -0.19653), 1.07: (0.97527, -0.20958), 1.1: (0.95120, -0.23757),
          1.2: (0.97098, -0.21796), 1.5: (0.93836, -0.25759), 2.0: (0.90879, -0.28598),
          3.0: (0.89334, -0.30860), 6.0: (0.87868, -0.33243)}
FLAT_FACTOR = 1.1
# round bar with a shoulder fillet in TORSION: Kt = A (r/d)^b fits to Peterson's chart by D/d (Shigley Fig. A-15-8
# fits, read second-hand); used for the ARB bar's shoulder (HC-5)
KT_FIT_TORSION = {1.09: (0.90337, -0.12692), 1.2: (0.83425, -0.21649), 1.33: (0.84897, -0.23161), 2.0: (0.86331, -0.23865)}
# geometry that lives in the CAD builders (scripts/fusion/fsae_parts.py), copied here
ROCKER = {"plate_t": 5.0, "plates": 2, "w": {"OA": 22.0, "OB": 22.0, "AB": 22.0}, "hole_d": 8.1, "bolt_d": 8.0}
KNEE = {"offset_mm": 12.7, "plate_w": 16.0, "r_inside": 3.0}         # D5; plate = stadium 2 x 8 mm radius
SPINDLE = {"do": 40.0, "di": 26.0, "shoulder_D": 46.0, "r_fillet": 1.0, "r_ring": 3.0, "ring_w": 4.0}   # D19: ring + R3
ARB_STUB_FTY = 435.0                                                  # 4130N stub (same as the tube)
R_EAR = 2.0                                                           # ear-root fillet on the drawing (note 5)
# holes bigger than the CAD's bolt hole: LBJ1 takes the thread insert named in upright_design.json; the lug checks
# use its outside thread diameter (data/fasteners.json)
_UD = json.loads((DATA / "upright_design.json").read_text())
_INS = json.loads((DATA / "fasteners.json").read_text())["inserts"][_UD["ball_joints"]["lower"]["insert"]]
INSERT_HOLE_MM = {"LBJ1": _INS["countersink_mm"]}                  # countersink_mm: the insert's outside thread
TORQUE_SCATTER = 1.3                                                  # preload from a torque wrench: about ±30 % (friction)


def kt_shoulder(r_over_d, D_over_d, flat=False):
    """The largest fit among every tabulated D/d up to the next one above the real D/d: the fits cross at
    small r/d (1.1 runs above 1.2 at r/d 0.025), so taking only the next row up is not always conservative."""
    key = min((k for k in KT_FIT if k >= D_over_d - 1e-9), default=max(KT_FIT))
    kt = max(A * r_over_d ** b for k, (A, b) in KT_FIT.items() if k <= key)
    return kt * (FLAT_FACTOR if flat else 1.0)


def kt_torsion(r_over_d, D_over_d):
    """As kt_shoulder, on the torsion fits: the largest fit up to the next tabulated D/d above the real one."""
    key = min((k for k in KT_FIT_TORSION if k >= D_over_d - 1e-9), default=max(KT_FIT_TORSION))
    return max(A * r_over_d ** b for k, (A, b) in KT_FIT_TORSION.items() if k <= key)


def endurance_torsion(sut, d, surface=(1.58, -0.085), ke=0.814):
    """Fully reversed torsional endurance limit, Marin-corrected (Shigley Eq. 6-8, 6-19, 6-20, Table 6-5): Se' = 0.5
    Sut (700 MPa above 1400), surface ka = a Sut^b (default ground), size kb = 1.24 d^-0.107, torsion kc = 0.59,
    reliability ke (default 0.814, 99 %, as the upright's fatigue)."""
    se = 0.5 * sut if sut <= 1400 else 700.0
    return surface[0] * sut ** surface[1] * 1.24 * d ** -0.107 * 0.59 * ke * se


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


# ------------------------------------------------------------------ HC-2 rocker
def rocker_lc1(mats):
    hp, meta = load_hardpoints()
    veh = load_vehicle()
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    members = forces.read_csv_rows(DATA / "members.csv")
    lc1 = next(r for r in forces.read_csv_rows(DATA / "loadcases.csv") if r["id"] == "LC1")
    acts = {"Pushrod": "LCA_O"} if corner.pushrod_on == "lca" else {}
    pose = corner.pose(float(lc1["z_pose_mm"]))
    f = np.zeros(len(members))
    for Fi, ati in forces.tyre_loads(corner, pose, lc1, veh):
        f += forces.solve_links(pose, members, Fi, acts, ati)[0]
    f_push = f[[m["member"] for m in members].index("Pushrod")]
    f_d3, F_piv3 = forces.rocker_and_damper(pose, corner, f_push)
    P, ax = pose["P"], corner.rk_axis
    piv = P["RKR_P"]

    def flat(v):
        return v - ax * np.dot(v, ax)
    e1 = unit(flat(P["DMP_R"] - piv))
    e2 = np.cross(ax, e1)
    to2 = lambda v: np.array([np.dot(v, e1), np.dot(v, e2)])     # noqa: E731
    rA, rB = to2(P["PR_I"] - piv), to2(P["DMP_R"] - piv)
    F_A3 = -f_push * unit(P["PR_I"] - P["PR_O"])                  # pushrod force on the rocker
    FA, F_oop = to2(F_A3), float(np.dot(F_A3, ax))
    ed = to2(unit(P["DMP_C"] - P["DMP_R"]))
    # 1. moments about the pivot -> damper force (tension +)
    f_d = -cross2(rA, FA) / cross2(rB, ed)
    # 2. pivot reaction
    R = -(FA + f_d * ed)
    # 3. truss: joint A (pushrod), then joint B (damper); tension +
    uAO, uAB = unit(-rA), unit(rB - rA)
    f_OA, f_AB = np.linalg.solve(np.column_stack([uAO, uAB]), -FA)
    D = f_d * ed
    uBO = unit(-rB)
    rest = -(D + f_AB * (-uAB))
    f_OB = float(np.dot(rest, uBO))
    resid_B = float(np.linalg.norm(rest - f_OB * uBO))
    t, n, dh, db = ROCKER["plate_t"], ROCKER["plates"], ROCKER["hole_d"], ROCKER["bolt_d"]
    E, Fty = mats["7075_T7351_plate"]["E_MPa"], mats["7075_T7351_plate"]["Fty_MPa"]
    L = {"OA": float(np.linalg.norm(rA)), "OB": float(np.linalg.norm(rB)), "AB": float(np.linalg.norm(rB - rA))}
    rows = []
    for k, fk in (("OA", f_OA), ("OB", f_OB), ("AB", f_AB)):
        w = ROCKER["w"][k]
        if fk >= 0:
            s_axial = fk / (n * t * (w - dh))                            # net section at the pin hole
            buck = math.inf
        else:
            s_axial = -fk / (n * t * w)
            P_cap, _ = forces.column_capacity(w * t, w * t ** 3 / 12, L[k], E, Fty)   # each plate pinned, weak axis
            buck = n * P_cap / -fk
        s_oop = 0.0
        if k == "OA":    # out-of-plane pushrod component: plate bending, largest at the pivot hole (net width)
            s_oop = (abs(F_oop) * L[k] / n) * 6 / ((w - dh) * t ** 2)
        rows.append({"member": k, "L_mm": L[k], "f_N": float(fk), "axial_MPa": s_axial, "oop_MPa": s_oop,
                     "fos_yield": Fty / (s_axial + s_oop), "fos_buckling": buck})
    # bearing at the bolts carries the whole pin load, not one member's force
    pins = {"A (pushrod)": float(np.linalg.norm(F_A3)), "B (damper)": abs(float(f_d)),
            "O (pivot)": float(np.linalg.norm(F_piv3))}
    bearing = {p: v / (n * t * db) for p, v in pins.items()}
    fos_br = Fty / max(bearing.values())
    return {"rA": rA, "rB": rB, "FA": FA, "F_oop": F_oop, "ed": ed, "f_push": float(f_push), "f_d": float(f_d),
            "f_d_solver": float(f_d3), "R": R, "R_solver": float(np.linalg.norm(F_piv3)), "resid_B": resid_B,
            "rows": rows, "pins": pins, "bearing": bearing, "fos_bearing": fos_br,
            "fos_min": min(min(min(r["fos_yield"], r["fos_buckling"]) for r in rows), fos_br),
            "Fty": Fty, "z_pose": float(lc1["z_pose_mm"])}


# ------------------------------------------------------------------ HC-3 bracket knee
def winkler_inner(M, h, t, ri):
    """Inner-fibre stress of a curved beam (rectangle h deep, t thick, inside radius ri) under moment M."""
    ro = ri + h
    rn = h / math.log(ro / ri)
    e = (ri + h / 2) - rn
    return M * (rn - ri) / (h * t * e * ri)


def knee(mats, checks):
    F = abs(float(next(c for c in checks if c["member"] == "Pushrod")["max_compression_N"]))
    t = float(next(m for m in forces.read_csv_rows(DATA / "members.csv") if m["member"] == "Pushrod")["tab_t_mm"])
    h, e, ri = KNEE["plate_w"], KNEE["offset_mm"], KNEE["r_inside"]
    Fty = mats["4130N_tab"]["Fty_MPa"]
    M = F * e / 2                                                    # per plate
    s_direct = F / (2 * t * h)
    s_straight = 6 * M / (t * h ** 2)
    s_curved = winkler_inner(M, h, t, ri)

    def fos(r):
        return Fty / (winkler_inner(M, h, t, r) + s_direct)
    r_need = next((r / 10 for r in range(2, 400) if fos(r / 10) >= TARGET), float("nan"))
    return {"F": F, "t": t, "h": h, "e": e, "ri": ri, "M_plate": M, "s_direct": s_direct, "s_straight": s_straight,
            "s_curved": s_curved, "K_curved": s_curved / s_straight, "fos": fos(ri), "fos_straight": Fty / (s_straight + s_direct),
            "r_need": r_need, "Fty": Fty, "table": [(r, fos(r)) for r in (1.0, 2.0, 3.0, 5.0, 8.0)]}


# ------------------------------------------------------------------ HC-4 hub spindle
def spindle(mats, veh):
    hub, R = veh["hub"], veh["tyre"]["loaded_radius_mm"]
    do, di = SPINDLE["do"], SPINDLE["di"]
    Z = math.pi * (do ** 4 - di ** 4) / (32 * do)
    a_sh = hub["row_centres_axial_mm"][1] + hub["width_mm"] / 2       # outboard cone back face = shoulder
    mat = mats["7075_T7351_plate"]
    S1, b, Se = sn_line(mat)

    hp, meta = load_hardpoints()
    corner = make_corner(hp, meta, R)
    pose = corner.pose(0.0)
    s = unit(pose["P"]["WA"] - pose["P"]["WC"])

    def moment(lc, a=a_sh):
        """Bending moment [N mm] about the shoulder section at the static pose, through the force solver's own
        load points (contact patch with the pneumatic trail, wheel centre, or the kerb point on the sidewall)."""
        Q = pose["P"]["WC"] + s * a
        Mv = sum(np.cross(np.asarray(at) - Q, F) for F, at in forces.tyre_loads(corner, pose, lc, veh))
        return float(np.linalg.norm(Mv - s * np.dot(Mv, s)))
    static = [(lc["id"], moment(lc) / 1000, moment(lc) / Z) for lc in forces.read_csv_rows(DATA / "loadcases.csv")]
    worst = max(static, key=lambda x: x[2])
    duty = json.loads((DATA / "duty_loads.json").read_text())
    spec = {x["state"]: x["share"] for x in veh["duty"]["spectrum"]}
    base_of = {n: n.replace(", outside", "").replace(", inside", "") for n in duty["states"]}
    if set(base_of.values()) != set(spec):
        raise RuntimeError(f"duty states and the spectrum differ: {sorted(set(base_of.values()) ^ set(spec))}")
    revs_total = duty["life_km"] * 1e6 / (2 * math.pi * R)
    states = []
    for name, st in duty["states"].items():
        share = spec[base_of[name]] * (0.5 if name != base_of[name] else 1.0)   # inside / outside split evenly
        Fx, Fy, Fz = st["tyre_N"]
        lc = {"Fx_N": Fx, "Fy_N": Fy, "Fz_N": Fz, "apply_at": "cp", "kerb_height_mm": 0.0}
        M, M_ring = moment(lc), moment(lc, a_sh + SPINDLE["ring_w"])     # the ring moves the shoulder outboard
        states.append({"state": name, "revs": share * revs_total, "M_Nm": M / 1000, "s_nom": M / Z,
                       "s_nom_ring": M_ring / Z})
    used = sum(x["revs"] for x in states) / revs_total
    # this shaft's own size factor (Shigley: kb = 1.24 d^-0.107, rotating, 2.79-51 mm) in place of the upright's 0.9
    kb = 1.24 * do ** -0.107
    Se_sp = Se * kb / 0.9
    b_sp = math.log(Se_sp / S1) / math.log(5e8 / 1e3)

    def k_for(kt, key="s_nom"):
        amps = np.array([[kt * x[key]] for x in states])
        means = np.zeros_like(amps)
        ns = np.array([x["revs"] for x in states])
        return float(margin(amps, means, ns, S1, b_sp, mat["Ftu_MPa"])[0])
    d, D = do, SPINDLE["shoulder_D"]
    kt_1 = kt_shoulder(SPINDLE["r_fillet"] / d, D / d)
    kt_3 = kt_shoulder(SPINDLE["r_ring"] / d, D / d)
    return {"do": do, "di": di, "Z": Z, "a_sh": a_sh, "R": R, "static": static, "worst": worst,
            "fos_static": mat["Fty_MPa"] / worst[2], "fos_static_kt": mat["Fty_MPa"] / (kt_1 * worst[2]),
            "revs_total": revs_total, "states": states, "used": used,
            "kt": kt_1, "kt_ring": kt_3, "k": k_for(kt_1), "k_ring": k_for(kt_3, "s_nom_ring"), "S1": S1, "b": b_sp, "Se": Se_sp,
            "kb": kb, "D": D, "Fty": mat["Fty_MPa"]}


# ------------------------------------------------------------------ HC-5 ARB ends
def arb_ends(mats, veh):
    """The stiff bar (the largest torque) in torsion: its body, and its splined ends at the spline's minor diameter
    with Kt (assumed, vehicle.json); and the minor diameter an end would need without the upset."""
    d = json.loads((DATA / "vehicle_derived.json").read_text())
    bar = veh["arb_bar"]
    T = d["arb_bar_torque_max_Nm"] * 1000
    D, Dm, kt = bar["bars_mm"]["stiff"], bar["spline_minor_mm"], bar["spline_kt_assumed"]
    m = mats[bar["material"]]
    allow = 0.577 * m["Fty_MPa"]
    tau_body = 16 * T / (math.pi * D ** 3)
    tau_end = kt * 16 * T / (math.pi * Dm ** 3)
    dm_need = (TARGET * kt * 16 * T / (math.pi * allow)) ** (1 / 3)
    # fatigue: fully reversed cornering roll at the largest g, infinite life; Kf = Kt (notch sensitivity 1)
    Tr = d["arb_bar_torque_roll_Nm"] * 1000
    r_f = bar["shoulder_fillet_mm"]
    kt_sh = kt_torsion(r_f / D, bar["end_od_mm"] / D)
    tau_a_sh = kt_sh * 16 * Tr / (math.pi * D ** 3)
    tau_a_sp = kt * 16 * Tr / (math.pi * Dm ** 3)
    sf = json.loads((DATA / "checks_week3b.json").read_text(encoding="utf-8"))["arb_bar_fatigue"]
    se_sh = endurance_torsion(m["Ftu_MPa"], D, sf["surface_factors"][sf["surface_body"]], sf["reliability_ke"])
    se_sp = endurance_torsion(m["Ftu_MPa"], Dm, sf["surface_factors"][sf["surface_spline"]], sf["reliability_ke"])
    fat = {"Tr_Nm": Tr / 1000, "r_f": r_f, "kt_sh": kt_sh, "tau_a_sh": tau_a_sh, "tau_a_sp": tau_a_sp, "se_sh": se_sh,
           "se_sp": se_sp, "fos_fat_sh": se_sh / tau_a_sh, "fos_fat_sp": se_sp / tau_a_sp, "Ftu": m["Ftu_MPa"],
           "surf_sh": sf["surface_body"], "surf_sp": sf["surface_spline"]}
    fat["fos_fat"] = min(fat["fos_fat_sh"], fat["fos_fat_sp"])
    return fat | {"T_Nm": T / 1000, "case": "warp" if d["arb_bar_torque_warp_Nm"] >= d["arb_bar_torque_roll_Nm"] else "roll",
            "D": D, "Dm": Dm, "kt": kt, "Fty": m["Fty_MPa"], "allow": allow, "tau_body": tau_body,
            "tau_body_solver": d["arb_bar_shear_max_MPa"], "fos_body": allow / tau_body, "fos_body_solver": d["arb_bar_fos_shear"],
            "tau_end": tau_end, "fos_end": allow / tau_end, "dm_need": dm_need, "end_od": bar["end_od_mm"]}


# ------------------------------------------------------------------ HC-6 steering arm
def steering(build, loads, ud, mesh):
    fea = json.loads((DATA / "upright_fea.json").read_text())
    f = build["faces"]
    H = {h["name"]: h for h in f["holes"]}
    s, wc = unit(np.array(f["spin"])), np.array(f["wc"])
    tro = np.mean([H["TR1"]["centre"], H["TR2"]["centre"]], axis=0)
    b = unit(np.array(H["TR1"]["axis"]))
    v = tro - wc
    eL = -unit(v - s * v.dot(s))
    eL = unit(eL - b * eL.dot(b))
    eW = unit(np.cross(b, eL))
    t, h = ud["clevis"]["ear_t_mm"], 2 * ud["clevis"]["ear_r_mm"]
    F = np.array(loads["cases"]["LC6"]["TR"]["force_N"])
    I, A = 2 * t * h ** 3 / 12, 2 * t * h
    d_sec = fea["steering_arm_section_mm"]
    FW, FL = float(F.dot(eW)), float(F.dot(eL))
    ear1 = ear_model(mesh, H, "TR1", "TR2", 2 * ud["clevis"]["ear_r_mm"], t, ud)   # filleted R2: the FEA covers it
    s_root = ear1["stresses"](F)["s_root"]
    fea_root = fea["cases"]["LC6"]["peaks_MPa"]["steering arm root"]
    val = fea["cases"]["LC6"]["validation"]
    return {"F": F.tolist(), "FW": FW, "FL": FL, "t": t, "h": h, "d_sec": d_sec, "I": I, "A": A,
            "M_sec": FW * d_sec, "validation": val, "L_root": ear1["L"], "kt": ear1["kt_in"], "s_root_peak": s_root,
            "fea_root": fea_root, "cal": max(1.0, fea_root / s_root), "hole_r": H["TR1"]["radius"],
            "fea_inputs_sha": fea.get("_inputs_sha")}


# ------------------------------------------------------------------ HC-7 ear roots
def ear_model(mesh, H, name, other, width, t, ud, joint="TR"):
    """One clevis ear as a cantilever plate from its root, with the drawing's R2 root fillet.
    Root distance: rays in the ear plane from just above and just below the plate; the body side is the
    direction of the nearest hit; L = the farther of the two hits there (the thinner, longer cantilever).
    Loads: along the bolt into the one ear the force pushes; across the bolt shared by the two ears."""
    c, a = np.array(H[name]["centre"]), unit(np.array(H[name]["axis"]))
    a = a if a[2] > 0 else -a
    upper = np.dot(c, a) > np.dot(np.array(H[other]["centre"]), a)
    u = unit(np.cross(a, [1.0, 0, 0]))
    v = np.cross(a, u)
    boss = H[name].get("boss_mm", 0.0)                      # a boss on the ear's outer face: start that ray above it
    offs = (t / 2 + boss + 0.6, -(t / 2 + 0.6)) if upper else (t / 2 + 0.6, -(t / 2 + boss + 0.6))
    best = None
    for th in np.radians(np.arange(0, 360, 5)):
        d = np.cos(th) * u + np.sin(th) * v
        hits = []
        for off in offs:
            loc, _, _ = mesh.ray.intersects_location([c + a * off], [d])
            hits.append(float(np.linalg.norm(loc - (c + a * off), axis=1).min()) if len(loc) else math.inf)
        if best is None or min(hits) < best[0]:
            best = (min(hits), max(hits), th)
    L = best[1]
    D_hole = INSERT_HOLE_MM.get(name, 2 * H[name]["radius"])
    e_edge = min(ud["clevis"]["ear_r_mm"], ud["clevis"]["inboard_edge_lower_mm"]) if joint == "LBJ" else ud["clevis"]["ear_r_mm"]
    kt_o = kt_shoulder(R_EAR / t, 2.0, flat=True)
    kt_i = kt_shoulder(R_EAR / width, 2.0, flat=True)

    c_o = kt_o * 6 * L / (width * t ** 2)               # root stress per N along the bolt (out of plane)
    c_i = kt_i * 6 * L / (t * width ** 2)               # root stress per N across the bolt (in plane)

    def stresses(F):
        """Loads on this ear and its root stresses for the force F the joint puts on the upright."""
        F = np.asarray(F, float)
        Fa = float(np.dot(F, a))
        P_ax = max(Fa, 0.0) if upper else max(-Fa, 0.0)
        F_in = (F - Fa * a) / 2                         # vector, so a load reversal keeps its sign in fatigue
        P_in = float(np.linalg.norm(F_in))
        return {"s_o": c_o * P_ax, "F_in": F_in, "s_root": c_o * P_ax + c_i * P_in, "P_ax": P_ax, "P_in": P_in,
                "s_net": P_in / ((width - D_hole) * t), "tau": P_in / (2 * t * (e_edge - D_hole / 2))}

    def cycle(A, B):
        """(amplitude, mean) of the root stress between two states; the in-plane part from the vector range
        (an upper bound for any fibre), the one-sided along-bolt part from its magnitudes."""
        amp = abs(A["s_o"] - B["s_o"]) / 2 + c_i * float(np.linalg.norm(A["F_in"] - B["F_in"])) / 2
        mean = (A["s_o"] + B["s_o"]) / 2 + c_i * float(np.linalg.norm(A["F_in"] + B["F_in"])) / 2
        return amp, mean
    return {"L": L, "upper": bool(upper), "D_hole": D_hole, "e_edge": e_edge, "kt_out": kt_o, "kt_in": kt_i,
            "stresses": stresses, "cycle": cycle}


def ear_roots(build, loads, ud, mats, mesh, cal):
    """The ears the CAD kernel could not fillet (S-14). cal (>= 1) scales every root stress: the ratio of
    the FEA's peak to this same model's at the steering-arm root the FEA does cover (HC-6)."""
    f = build["faces"]
    H = {h["name"]: h for h in f["holes"]}
    mat = mats["7075_T7351_plate"]
    S1, b_sn, _ = sn_line(mat)
    duty = json.loads((DATA / "duty_loads.json").read_text())
    t = ud["clevis"]["ear_t_mm"]
    # which hole each unfilleted feature is, its pair, the joint key in the load files, the ear width
    ears = {"lower ball-joint ear 1": ("LBJ1", "LBJ2", "LBJ", ud["clevis"]["ear_r_mm"] + ud["clevis"]["inboard_edge_lower_mm"]),
            "steering-arm ear 2": ("TR2", "TR1", "TR", 2 * ud["clevis"]["ear_r_mm"])}
    sharp = build.get("sharp_edges")
    if not sharp:
        raise RuntimeError("upright_build.json lists no sharp edges: S-14 cannot be checked (a missing list must not pass)")
    unknown = [e["feature"] for e in sharp if e["feature"] not in ears]
    if unknown:
        raise RuntimeError(f"no ear model for the sharp edges {unknown}: add them to handcalcs.ear_roots")
    out = []
    for feat in sharp:
        name, other, joint, width = ears[feat["feature"]]
        em = ear_model(mesh, H, name, other, width, t, ud, joint)
        static = []
        for lc, case in loads["cases"].items():
            r = em["stresses"](case[joint]["force_N"])
            static.append({"lc": lc, "P_ax": r["P_ax"], "P_in": r["P_in"], "s_root": cal * r["s_root"],
                           "s_net": r["s_net"], "tau": r["tau"]})
        w = max(static, key=lambda r: r["s_root"])
        fos_root = mat["Fty_MPa"] / w["s_root"]
        fos_net = mat["Fty_MPa"] / max(r["s_net"] for r in static)
        fos_shear = 0.577 * mat["Fty_MPa"] / max(r["tau"] for r in static)
        st = {k: em["stresses"](v[joint]["force_N"]) for k, v in duty["states"].items()}
        cyc = [em["cycle"](st[c["a"]], st[c["b"]]) for c in duty["cycles"]]
        amps = cal * np.array([[a_] for a_, _ in cyc])
        means = cal * np.array([[m_] for _, m_ in cyc])
        ns = np.array([c["n"] for c in duty["cycles"]])
        k = float(margin(amps, means, ns, S1, b_sn, mat["Ftu_MPa"])[0]) if amps.max() > 0 else math.inf
        out.append({"feature": feat["feature"], "hole": name, "upper_ear": em["upper"], "L_mm": em["L"], "width_mm": width,
                    "t_mm": t, "r_mm": R_EAR, "D_hole_mm": em["D_hole"], "e_mm": em["e_edge"], "kt_out": em["kt_out"],
                    "kt_in": em["kt_in"], "cal": cal, "worst": w, "fos_root": fos_root, "fos_net": fos_net,
                    "fos_shear": fos_shear, "fatigue_k": k, "fos_min": min(fos_root, fos_net, fos_shear, k)})
    return out


# ------------------------------------------------------------------ writing
def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def f1(x):
    return f"{x:.1f}"


def main():
    OUT.mkdir(exist_ok=True)
    ver = read_version(DATA / "hardpoints.csv")
    mats = load_materials()
    veh = load_vehicle()
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    loads = json.loads((DATA / "upright_loads.json").read_text())
    ud = json.loads((DATA / "upright_design.json").read_text())
    checks = forces.read_csv_rows(DATA / "member_checks.csv")

    h1 = handcalc_lc1.solve()
    rk = rocker_lc1(mats)
    kn = knee(mats, checks)
    sp = spindle(mats, veh)
    ab = arb_ends(mats, veh)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "drawings"))
    import views as V
    mesh = V.mesh_solid(ROOT / "cad" / "upright" / "upright_L.step", size=1.5, curvature=12)
    sa = steering(build, loads, ud, mesh)
    er = ear_roots(build, loads, ud, mats, mesh, sa["cal"])
    ins_pull = 0.5 * math.pi * _INS["countersink_mm"] * _INS["length_mm"] * 0.577 * mats["7075_T7351_plate"]["Fty_MPa"]
    ins_load = ud["bolt_torque"]["LBJ"]["torque_Nm"] * 1000 / (ud["bolt_torque"]["K_nut_factor"] * ud["ball_joints"]["lower"]["bolt_d_mm"])
    bh = W3.brake_heat(veh, float(mesh.area), build["masses_kg"]["upright"])
    hp, meta = load_hardpoints()
    wt = W3.welds_threads(make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"]), veh, mats)
    fp = W3.free_play(veh)
    vder = json.loads((DATA / "vehicle_derived.json").read_text())
    tb = W3.tab_strength(max(rk["R_solver"], float(np.linalg.norm(rk["R"]))), abs(rk["f_d"]),
                         vder["rack_force_driver_effort_N"], mats)
    al = W3.arb_lever_joint(vder, veh, ud, mats, hp)

    head = f"_Hardpoints {ver}. Inputs are the current design's; generated by `scripts/handcalcs.py`. Work it on paper first, then check `answer_key.md`._"
    P = h1["pts"]
    write("HC-1_lc1_front_view.md", [
        "# HC-1: LC1 (3 g bump) in the front view", "", head, "",
        "**Why:** an independent 2D check of the 3D force solver. If your pushrod force lands within a few per cent of the 3D value, the 6x6 solve is right.", "",
        "**Given** (front view, y outboard and z up, mm; the left front corner at static ride height):", "",
        "| Point | y | z |", "|---|---|---|",
        *[f"| {k} | {f1(p[0])} | {f1(p[1])} |" for k, p in P.items()],
        f"| Contact patch | {f1(h1['cp'][0])} | {f1(h1['cp'][1])} |", "",
        f"Vertical tyre load Fz = {h1['Fz']:.0f} N up at the contact patch (3 x the static corner load). The upper wishbone is one two-force member along UCA_O -> its pivot trace; the lower wishbone likewise, and the pushrod acts on the lower node along PR_O -> PR_I. The tie rod carries nothing in a pure vertical case.", "",
        "**Find:** (1) the upper-wishbone force, (2) the lower ball-joint force on the upright, (3) the lower-wishbone and pushrod forces.", "",
        "**Method:** free body of the upright, moments about the lower ball joint; then force balance; then the lower node (two unknowns, two equations). Draw the free bodies to scale first.", ""])
    write("HC-2_rocker.md", [
        "# HC-2: Rocker in LC1 (bump-stop force included)", "", head, "",
        "**Why:** requirement S-06e (rocker stresses, FoS >= 1.5 on yield). In LC1 the wheel is at +30 mm, on the bump rubber, so the damper-axis force includes the rubber.", "",
        f"**Given** (the rocker's own plane at the LC1 pose, z = +{rk['z_pose']:.0f} mm; origin at the pivot, mm):", "",
        "| Item | x | y |", "|---|---|---|",
        f"| Pushrod eye A | {f1(rk['rA'][0])} | {f1(rk['rA'][1])} |",
        f"| Damper eye B | {f1(rk['rB'][0])} | {f1(rk['rB'][1])} |",
        f"| Pushrod force on the rocker at A (in-plane), N | {rk['FA'][0]:.0f} | {rk['FA'][1]:.0f} |",
        f"| Damper axis, unit vector from B towards the chassis eye | {rk['ed'][0]:.3f} | {rk['ed'][1]:.3f} |", "",
        f"Out-of-plane pushrod component at A: {rk['F_oop']:.0f} N. The ARB link is unloaded (both wheels bump together). "
        f"Rocker: two 7075 plates, {ROCKER['plate_t']:g} mm thick, arms {ROCKER['w']['OA']:g} mm wide, a third plate A-B; M8 bolts in {ROCKER['hole_d']:g} mm holes. Design yield {rk['Fty']:.0f} MPa.", "",
        "**Find:** (1) the damper force (moments about the pivot); (2) the pivot reaction; (3) the forces in O-A, O-B and A-B treating the rocker as a pin-jointed triangle; (4) the stress in each (net section at the hole in tension, gross section and plate buckling in compression), the bearing stress at the bolts, and the bending the out-of-plane load puts in arm O-A; (5) the lowest FoS.", "",
        "**Method:** 2D statics. Truss joint A first (two unknowns), then joint B. Treat each plate as a pinned column about its weak axis for buckling.", ""])
    write("HC-3_pushrod_bracket_knee.md", [
        "# HC-3: Pushrod bracket knee", "", head, "",
        "**Why:** D5. The bracket plates leave the pushrod's line to clear the upright's ear, so the knee carries the pushrod force times its offset as bending. Rev D got this wrong once (70 MPa written for 6 mm plates).", "",
        "**Given:**", "",
        f"- Pushrod force (LC1, compression) F = {kn['F']:.0f} N, shared by two plates.",
        f"- Knee offset from the force line e = {kn['e']:g} mm.",
        f"- Each plate: 4130N, t = {kn['t']:g} mm, h = {kn['h']:g} mm deep at the knee; inside radius of the knee r_i = {kn['ri']:g} mm (to be laser cut). Yield {kn['Fty']:.0f} MPa.", "",
        "**Find:** (1) the moment per plate; (2) the direct stress; (3) the straight-beam bending stress; (4) the curved-beam (Winkler) stress at the inside radius and the factor it adds; (5) FoS; (6) the smallest inside radius that gives FoS 1.5.", "",
        "**Method:** curved beam: r_n = h / ln(r_o / r_i), e = r_c - r_n, sigma_i = M (r_n - r_i) / (A e r_i). Add the direct stress.", ""])
    write("HC-4_hub_spindle.md", [
        "# HC-4: Hub spindle in rotating bending", "", head, "",
        "**Why:** requirement S-15. The spindle turns under a bending moment that is fixed in space, so every wheel revolution is one fully reversed stress cycle.", "",
        "**Given:**", "",
        f"- Hollow 7075 spindle at the bearing seats: {sp['do']:g} mm outside, {sp['di']:g} mm bore.",
        f"- The critical section is the shoulder at the outboard cone's back face, {sp['a_sh']:.1f} mm outboard of the wheel-centre plane; the shoulder is {sp['D']:g} mm across with a {SPINDLE['r_fillet']:g} mm fillet (the cone's back-face chamfer caps it).",
        f"- Tyre forces act at the contact patch, in the wheel-centre plane, R = {sp['R']:g} mm below the axis.",
        f"- Load cases (`data/loadcases.csv`) and the duty spectrum (`data/duty_loads.json`: each state's tyre forces; `vehicle.json` duty: each state's share of the {veh['duty']['season_km_assumed'] * veh['duty']['life_target_seasons']:.0f} km life; outside and inside wheel split evenly).",
        f"- S-N line: the upright's construction (`materials.json`) with this shaft's own size factor kb = 1.24 d^-0.107: S = {sp['S1']:.0f} MPa x (N / 1e3)^({sp['b']:.4f}). Design yield {sp['Fty']:.0f} MPa.",
        "- The key takes each load through the force solver's own load points (static camber, the pneumatic trail, the wheel centre or the kerb point), which moves the moments by about 1 % from the 2D formula below.", "",
        "**Find:** (1) the section modulus; (2) the bending moment at the shoulder for each load case (M_x = -a Fz + R Fy, M_z = a Fx) and the static FoS; (3) Kt for the shoulder fillet from Peterson's chart; (4) the revolutions in each duty state and their stress amplitude; (5) Miner's damage and the factor on stress that brings it to 1 (the fatigue FoS). (6) Repeat (3)-(5) for a 3 mm fillet behind a shoulder ring.", "",
        "**Method:** the brake torque does not pass this section (the disc and wheel bolt to the flange outboard of it). Mean stress is zero (rotating bending).", ""])
    write("HC-5_arb_bar_ends.md", [
        "# HC-5: Anti-roll bar, body and splined ends", "", head, "",
        "**Why:** requirement S-09. Since D26 the bar sets the ARB's rate: three solid bars, swapped to adjust, between two "
        "stiff levers. The stiff bar carries the largest torque; its ends are splined into the levers.", "",
        "**Given:**", "",
        f"- Worst bar torque T = {ab['T_Nm']:.0f} N m ({ab['case']}: one wheel at full bump, the other at the droop stop; stiff setting).",
        f"- Stiff bar: solid 4340 heat treated to 180 ksi (Fty {ab['Fty']:g} MPa, shear yield 0.577 x Fty), body {ab['D']:g} mm.",
        f"- Ends upset to {ab['end_od']:g} mm and splined, minor diameter {ab['Dm']:g} mm; Kt = {ab['kt']:g} on the minor diameter (assumed, the high end of the splined-shaft range).", "",
        "**Find:** (1) the body's torsional shear and FoS; (2) the end's shear at the minor diameter with Kt, and its FoS; "
        "(3) the smallest minor diameter for FoS 1.5, and why the ends are upset above the body.", "",
        "**Fatigue (S-23):** the bar reverses every left-right corner. Given the largest cornering roll's bar torque, "
        f"{ab['Tr_Nm']:.0f} N m, fully reversed; Ftu {ab['Ftu']:g} MPa; the body and its fillets {ab['surf_sh']}, "
        f"the spline {ab['surf_sp']}; an R{ab['r_f']:g} fillet "
        f"where the body meets the {ab['end_od']:g} mm upset.", "",
        "**Find:** (4) the torsional endurance limit with Marin's factors (surface, size, load) at the body and at the "
        "spline's minor diameter; (5) Kt at the shoulder (Peterson's torsion chart for a stepped round bar) and the "
        "infinite-life FoS at the shoulder and at the spline, taking Kf = Kt.", ""])
    write("HC-6_steering_arm.md", [
        "# HC-6: Steering arm, beam theory and where it breaks", "", head, "",
        "**Why:** DV-2. The plan wanted the FEA within 10 % of a hand calc at a plain section; the arm has none. This packet shows why, then checks the root properly.", "",
        "**Given:**", "",
        f"- LC6 (driver effort, wheels blocked) tie-rod force on the arm: {sa['FW']:.0f} N across the arm and {sa['FL']:.0f} N along it (+ towards the housing).",
        f"- Two ears, each {sa['t']:g} mm thick and {sa['h']:g} mm deep; the bolt hole is {2 * sa['hole_r']:.1f} mm.",
        f"- Section S at {sa['d_sec']:.1f} mm from the bolt. The upper ear (TR1, filleted R2, which the FEA covers) meets the body {sa['L_root']:.1f} mm from its bolt; its width is {sa['h']:g} mm.", "",
        "**Find:** (1) the beam stress at both edges of section S (sigma = -M y / I - P / A); (2) compare with the FEA there (answer key) and explain the gap (how far is S from the load, compared with the depth?); (3) the nominal stress at the root and Kt for the R2 fillet (Peterson, flat bar with shoulder fillets in bending); (4) compare the peak with the FEA's steering-arm root peak.", ""])
    write("HC-7_ear_roots.md", [
        "# HC-7: Ear roots the CAD kernel could not fillet", "", head, "",
        "**Why:** requirement S-14. Two ear roots are sharp in the CAD (the kernel refused every radius), so the FEA leaves them out; the drawing calls R2. They need a hand check, static and fatigue.", "",
        "**Given** (per ear; loads from `data/upright_loads.json` and `data/duty_loads.json`):", "",
        "| Ear | Hole | Cantilever length L [mm] | Width b [mm] | t [mm] | Hole [mm] | Edge distance [mm] |", "|---|---|---|---|---|---|---|",
        *[f"| {e['feature']} | {e['hole']} | {e['L_mm']:.1f} | {e['width_mm']:.1f} | {e['t_mm']:g} | {e['D_hole_mm']:.1f} | {e['e_mm']:g} |" for e in er], "",
        "Load model: the force along the bolt goes into the one ear it pushes (the conservative end of a clamped stack); the force across the bolt is shared by the two ears.", "",
        "**Find:** (1) root bending, out of plane (6 P L / (b t^2)) and in plane (6 P L / (t b^2)), each with its Kt; (2) net-section tension and shear-out of the lug; (3) the worst case and the static FoS; (4) Miner's factor over the duty spectrum.", ""])
    bhc, lim = bh["C"]["brake_heat"], bh["C"]["limits"]
    write("HC-8_brake_heat.md", [
        "# HC-8: Brake heat into the upright", "", head, "",
        "**Why:** requirement S-16. The cups' press fit is lost as the aluminium grows away from the steel, the cups' "
        "retaining compound and the 7075 temper have temperature limits, and nylon-insert nuts are allowed only below "
        f"{lim['nylon_insert_nut_C']:.0f} C (FSAE T.8.3.2(c)).", "",
        "**Given** (assumptions marked; `data/checks_week3b.json`):", "",
        f"- Car {veh['mass_total_kg']:.0f} kg; front brake share {bh['share']:.3f} of the braking force (LC3, 1.5 g, fixed bias).",
        f"- Disc 410 stainless, {veh['brake']['disc_od_mm_assumed']:.0f} mm outside, {2 * W3.DISC_RI_MM:.0f} mm inside, "
        f"{W3.DISC_T_MM:g} mm thick; {bhc['disc_density_kg_m3']:.0f} kg/m^3, c = {bhc['disc_c_J_kgK']:.0f} J/kg K, emissivity "
        f"{bhc['disc_emissivity_assumed']:g} (assumed).",
        f"- Hardest stop {bhc['hardest_stop_kph_assumed']:.0f} km/h to rest (assumed). Endurance {bhc['endurance_km']:.0f} km at "
        f"a mean {bhc['mean_speed_kph_assumed']:.0f} km/h with {bh['per_km']:.0f} braking events per km, each "
        f"{bhc['event_speed_kph_assumed'][0]:.0f} to {bhc['event_speed_kph_assumed'][1]:.0f} km/h (assumed). Ambient {bh['Ta']:.0f} C.",
        f"- Disc convection h = {', '.join(f'{h:g}' for h in bhc['disc_h_W_m2K_assumed'])} W/m^2 K (no verified correlation: a range).",
        f"- Upright: surface {bh['A_up'] * 1e4:.0f} cm^2 (from the CAD solid), {build['masses_kg']['upright']:.3f} kg, "
        f"c = {bhc['upright_c_J_kgK']:.0f} J/kg K, h = {bhc['upright_h_W_m2K_assumed']:g} W/m^2 K (assumed). Hub flange: inboard "
        f"face and rim {bh['A_hub'] * 1e4:.0f} cm^2 at h = {bhc['hub_h_W_m2K_assumed']:g} (assumed; the outboard face is under the "
        "wheel). The hub and the upright are taken at one temperature: the bearings between them conduct well.",
        f"- Limits: Loctite 641 {lim['loctite_641_C']:.0f} C; 7075-T73 long-term {lim['al7075_T73_long_term_C_assumed']:.0f} C "
        f"(assumed); nylon-insert nuts {lim['nylon_insert_nut_C']:.0f} C. Cup fit {bh['fit_um'][0]:.0f}-{bh['fit_um'][1]:.0f} µm "
        f"interference at 20 C on the {veh['hub']['od_mm']:g} mm bore (`notes/bearings.md`); expansion "
        f"{veh['hub']['fits']['alpha_al_per_C'] * 1e6:g}e-6 /K (7075) and {veh['hub']['fits']['alpha_steel_per_C'] * 1e6:g}e-6 /K (cup).",
        f"- Mounts: the solid steel hat of v1.7, {W3.HAT_R_MM[0]:g}-{W3.HAT_R_MM[1]:g} mm radius, "
        f"{bhc['hat_length_mm_assumed']:g} mm long, k = {bhc['disc_k_W_mK_assumed']:g} W/m K; or a floating disc on "
        f"{bhc['float']['bobbins']} bobbins, each with two {bhc['float']['contact_dia_mm']:g} mm contact faces and "
        f"{bhc['float']['path_mm']:g} mm of bobbin between them, in stainless (k "
        f"{bhc['float']['options']['stainless']['k_W_mK']:g}) or titanium (k {bhc['float']['options']['titanium']['k_W_mK']:g}). "
        f"Contact conductance {', '.join(f'{hc:.0f}' for hc in bh['hcs'])} W/m^2 K (assumed range).", "",
        "**Find:** (1) the energy into one front disc in the hardest stop and its temperature rise with no cooling; "
        "(2) the mean braking power per front disc over the endurance; (3) the disc's running temperature for each h "
        "(convection + radiation = mean power); (4) the housing rise at which the cup fit opens; (5) the heat the upright "
        "and hub can shed at their limit, and the time constant against the endurance's length; (6) the thermal "
        "resistance the disc mount needs, and what each mount gives across the contact range.", ""])
    write("HC-9_welds_and_threads.md", [
        "# HC-9: Member-end welds and rod-end threads in fatigue", "", head, "",
        "**Why:** requirement S-07b. Every link force passes through a fillet weld at each tube end (to the ball-joint "
        "node or the threaded insert) and through the rod end's first engaged thread.", "",
        "**Given:**", "",
        "- Member forces in every duty state from the 6 x 6 solve at static ride height (the duty states of "
        "`data/duty_loads.json`), and the duty cycles between them (" + f"{wt['n_total']:,.0f} cycles over the life).",
        f"- Weld: IIW, load-carrying fillet weld, toe FAT {wt['C']['welds']['fat_toe_MPa']:g} on the tube's nominal stress, root FAT "
        f"{wt['C']['welds']['fat_root_MPa']:g} on the throat (throat = tube wall, a drawing rule), slope {wt['C']['welds']['m']:g}, "
        "no endurance limit.",
        f"- Thread: M8 x 1.25 shank, stress area {wt['C']['threads']['stress_area_mm2']:g} mm^2, EN 1993-1-9 detail "
        f"{wt['C']['threads']['fat_MPa']:g}, slope {wt['C']['threads']['m']:g}; no credit for the jam nut. Engagement "
        f"{wt['C']['threads']['engagement_d']:g} d in a 4130 insert.", "",
        "| Member | Tube section [mm^2] | Largest force range [N] | Its cycles |", "|---|---|---|---|",
        *[f"| {r['member']} | {r['A_tube']:.1f} | {r['dF_max']:.0f} | {r['n_at_max']:,.0f} |" for r in wt["rows"]], "",
        "**Find:** (1) the stress range at the toe, on the throat and in the thread for each cycle; (2) Miner's damage on "
        "each S-N line and the factor on stress that brings it to 1; (3) the insert thread's stripping load "
        "(0.577 Fty over half the cylinder at the thread's major diameter) against the member's largest static force.", ""])
    fpc = fp["C"]
    write("HC-10_steering_free_play.md", [
        "# HC-10: Steering free play", "", head, "",
        f"**Why:** requirement R-06 (FSAE V.3.2.5: under {fpc['limit_deg']:g} deg at the steering wheel).", "",
        "**Given:**", "",
        f"- Rack: {fp['c']:g} mm of rack per steering-wheel turn (`vehicle.json`; the Kaz rack itself is "
        f"{fp['kaz_c']:.1f} mm). Lash after adjustment {fpc['rack_lash_mm_assumed']:g} mm at the rack (assumed).",
        f"- {fpc['rod_ends_in_chain']} PTFE-lined rod ends per tie rod, {fpc['rod_end_play_mm_assumed']:g} mm radial play each (assumed).",
        f"- The column, its two universal joints and the quick release: {fpc['column_allocation_deg_assumed']:g} deg allowed (chassis side).", "",
        "**Find:** the play at the rack, taken to the steering wheel, plus the column's share; compare with the rule.", ""])
    write("HC-11_chassis_tabs.md", [
        "# HC-11: Chassis tabs and brackets", "", head, "",
        "**Why:** requirement S-19. Each inboard joint is held by a pair of 4130N plates (or one bracket web) welded "
        "to a chassis tube (`scripts/chassis_tabs.py`); the joint's force bends each plate in its own plane about the weld.", "",
        "**Given** (per part; plates are 4130N, the weld-affected yield "
        f"{tb['allow']:.0f} MPa = {mats['4130N_tube']['Fty_MPa']:.0f} x {mats['4130N_tube']['weld_knockdown']:g}):", "",
        "| Part | Plates | t [mm] | Width [mm] | Bolt to weld [mm] | Force [N] | Basis |", "|---|---|---|---|---|---|---|",
        *[f"| {r['part']} | {r['plates']} | {r['t_mm']:g} | {r['w_mm']:.0f} | {r['L_mm']:.1f} | {r['F_N']:.0f} | {r['basis']} |" for r in tb["rows"]], "",
        "**Find:** for each part, split the force into its parts along and across the plate; the root bending "
        "6 F_across L / (n t w^2) and the direct stress F_along / (n t w); the FoS.", ""])

    write("HC-12_drop_link_lever_end.md", [
        "# HC-12: The drop link's lever end and the ARB lever", "", head, "",
        "**Why:** requirements S-20 and S-21. The drop link's lower rod end is bolted through the ARB lever's blade "
        "along the bar's axis, beside the blade on a misalignment spacer, with a second spacer and a 24 mm washer "
        "outside it (R-08: single shear, captured). The bolt is loaded as a cantilever; the link pulls beside the "
        "blade, so the blade bends and twists.", "",
        "**Given:**", "",
        f"- The ARB's largest torque {al['T_Nm']:.1f} N m ({al['case']}, stiff blade; `vehicle_derived.json`). "
        f"Lever: the link's moment arm about the bar {al['arm']:.1f} mm (link along ARB_L -> ARB_R).",
        f"- Bolt M8 10.9 (Fty {al['bolt_Fty']:g} MPa), stress area {al['As']:g} mm^2; blade face to the ball's "
        f"centre {al['a_b']:g} mm (half the 12 mm ball + a {al['a_b'] - 6:g} mm spacer).",
        f"- Blade 4130N (Fty {al['lever_Fty']:g} MPa), {al['t']:g} x {al['w']:g} mm; hub outer radius "
        f"{al['r_hub']:.1f} mm; the ball's centre {al['off']:g} mm beside the blade's mid-plane.", "",
        "**Find:** (1) the drop link's force; (2) the bolt's bending stress at the blade face on the stress area's "
        "diameter, its direct shear, von Mises and the FoS; (3) the blade's bending stress at the hub, its torsion "
        "from the offset (rectangle: T (3a + 1.8b) / (a^2 b^2)), von Mises and the FoS; (4) the end's shear-out past "
        "the M8 hole (two planes, 0.577 Fty).", ""])

    # ---------------- answer key
    K = ["# Answer key: hand-calc packets HC-1 to HC-12", "", f"_Hardpoints {ver}. Generated by `scripts/handcalcs.py`; do the packets first._", ""]
    K += ["## HC-1: LC1 front view", "",
          f"1. Moments about the lower ball joint: upper-wishbone force {h1['f_uca']:.0f} N (tension +).",
          f"2. Lower ball-joint force on the upright: ({h1['B'][0]:.0f}, {h1['B'][1]:.0f}) N, |B| = {math.hypot(*h1['B']):.0f} N.",
          f"3. Lower node: lower wishbone (front-view resultant) {h1['f_lca']:.0f} N, pushrod {h1['f_p']:.0f} N.",
          f"Check: the 3D solver gives the pushrod {h1['push3d']:.0f} N at the same pose ({abs(h1['f_p'] - h1['push3d']) / abs(h1['push3d']) * 100:.1f} % apart).", ""]
    K += ["## HC-2: Rocker (S-06e)", "",
          f"1. Damper force {rk['f_d']:.0f} N (tension +); the solver gives {rk['f_d_solver']:.0f} N.",
          f"2. Pivot reaction ({rk['R'][0]:.0f}, {rk['R'][1]:.0f}) N, |R| = {np.linalg.norm(rk['R']):.0f} N (solver {rk['R_solver']:.0f} N, which includes the out-of-plane part).",
          "3. Truss (tension +): " + ", ".join(f"{r['member']} {r['f_N']:.0f} N" for r in rk["rows"]) + f" (joint B closes to {rk['resid_B']:.1f} N).", "",
          "| Member | Length [mm] | Force [N] | Axial [MPa] | Out-of-plane bending at the pivot hole [MPa] | FoS yield | FoS buckling (Euler / Johnson, each plate) |",
          "|---|---|---|---|---|---|---|",
          *[f"| {r['member']} | {r['L_mm']:.1f} | {r['f_N']:.0f} | {r['axial_MPa']:.1f} | {r['oop_MPa']:.1f} | {r['fos_yield']:.2f} | {r['fos_buckling']:.1f} |" for r in rk["rows"]], "",
          "4b. Bearing at the bolts (the whole pin load on two 5 mm plates, 8 mm pin; the pivot taken the same way, which is conservative for its 22 mm bore): "
          + ", ".join(f"{p_} {rk['pins'][p_]:.0f} N, {rk['bearing'][p_]:.1f} MPa" for p_ in rk["pins"]) + f"; FoS {rk['fos_bearing']:.1f}.", "",
          f"Lowest FoS {rk['fos_min']:.2f}: S-06e {'PASS' if rk['fos_min'] >= TARGET else 'FAIL'} (>= {TARGET}).", ""]
    K += ["## HC-3: Pushrod bracket knee", "",
          f"1. M = F e / 2 = {kn['F']:.0f} x {kn['e']:g} / 2 = {kn['M_plate'] / 1000:.1f} N m per plate.",
          f"2. Direct stress F / (2 t h) = {kn['s_direct']:.1f} MPa.",
          f"3. Straight beam 6 M / (t h^2) = {kn['s_straight']:.1f} MPa (FoS {kn['fos_straight']:.2f} with the direct stress).",
          f"4. Curved beam at r_i = {kn['ri']:g} mm: {kn['s_curved']:.1f} MPa, {kn['K_curved']:.2f} x the straight-beam value.",
          f"5. FoS {kn['Fty']:.0f} / ({kn['s_curved']:.1f} + {kn['s_direct']:.1f}) = {kn['fos']:.2f}.",
          f"6. FoS 1.5 needs r_i >= {kn['r_need']:.1f} mm. FoS by inside radius: " + ", ".join(f"r {r:g}: {v:.2f}" for r, v in kn["table"]) + ".",
          "The knee's inside radius goes on the bracket's cut list (D5).", ""]
    K += ["## HC-4: Hub spindle (S-15)", "",
          f"1. Z = pi (do^4 - di^4) / (32 do) = {sp['Z']:.0f} mm^3.",
          "2. Static, moment at the shoulder (a = " + f"{sp['a_sh']:.1f} mm, R = {sp['R']:g} mm):", "",
          "| Case | M [N m] | Stress [MPa] |", "|---|---|---|",
          *[f"| {c} | {m:.0f} | {s:.1f} |" for c, m, s in sp["static"]], "",
          f"   Worst {sp['worst'][0]}: FoS {sp['Fty']:.0f} / {sp['worst'][2]:.1f} = {sp['fos_static']:.2f} on yield (nominal; {sp['fos_static_kt']:.2f} with Kt).",
          f"3. Shoulder D/d = {sp['D'] / sp['do']:.2f}, r/d = {SPINDLE['r_fillet'] / sp['do']:.3f}: Kt = {sp['kt']:.2f} (the largest of the bending fits up to the next tabulated D/d).",
          f"4. Life {sp['revs_total']:.3g} revolutions (life / 2 pi R); per state:", "",
          "| State | Revolutions | M [N m] | Nominal [MPa] | Amplitude Kt x nominal [MPa] |", "|---|---|---|---|---|",
          *[f"| {s['state']} | {s['revs']:.3g} | {s['M_Nm']:.0f} | {s['s_nom']:.1f} | {sp['kt'] * s['s_nom']:.1f} |" for s in sp["states"]], "",
          f"   The states cover {sp['used']:.0%} of the life's revolutions. S-N line with this shaft's size factor kb = 1.24 d^-0.107 = {sp['kb']:.3f} (the upright's uses 0.9): {sp['Se']:.0f} MPa at 5e8.",
          f"5. Miner factor k = {sp['k']:.2f}: S-15 {'PASS' if sp['k'] >= TARGET else 'FAIL'} (>= {TARGET}).",
          f"6. With a {SPINDLE['ring_w']:g} mm shoulder ring behind the cone, so the shaft can have a {SPINDLE['r_ring']:g} mm fillet "
          f"(its section {SPINDLE['ring_w']:g} mm further outboard): Kt = {sp['kt_ring']:.2f}, k = {sp['k_ring']:.2f}. "
          "The design takes the ring (D19): the bare fillet's margin is too thin for a part whose failure loses the wheel.", ""]
    K += ["## HC-5: ARB bar, body and splined ends (S-09)", "",
          f"1. Body: tau = 16 T / (pi D^3) = {ab['tau_body']:.1f} MPa (vehicle calcs {ab['tau_body_solver']:.1f}); allowable "
          f"{ab['allow']:.0f} MPa: FoS {ab['fos_body']:.2f}.",
          f"2. End: Kt x 16 T / (pi Dm^3) = {ab['tau_end']:.1f} MPa: FoS {ab['fos_end']:.2f}.",
          f"3. Minor diameter >= {ab['dm_need']:.1f} mm for FoS 1.5. A spline cut into the {ab['D']:g} mm body would leave "
          "a minor diameter well under that, so the ends are upset: the bar is turned down from larger stock, leaving the "
          f"{ab['end_od']:g} mm ends. S-09 {'PASS' if min(ab['fos_body'], ab['fos_end']) >= TARGET else 'FAIL'} "
          f"(>= {TARGET}). A static check of a ductile part takes no Kt at the shoulder; fatigue does (below).",
          f"4. Endurance in torsion: body {ab['se_sh']:.0f} MPa ({ab['surf_sh']}), spline {ab['se_sp']:.0f} MPa "
          f"({ab['surf_sp']}) (Se' = 0.5 Ftu, ka by surface, kb by size, kc 0.59, ke 0.814 for 99 % reliability).",
          f"5. Shoulder: r/d {ab['r_f'] / ab['D']:.3f}, D/d {ab['end_od'] / ab['D']:.2f}: Kt {ab['kt_sh']:.2f}; tau_a "
          f"{ab['tau_a_sh']:.0f} MPa: FoS {ab['fos_fat_sh']:.2f}. Spline: Kt {ab['kt']:g}, tau_a {ab['tau_a_sp']:.0f} MPa: "
          f"FoS {ab['fos_fat_sp']:.2f}. S-23 {'PASS' if ab['fos_fat'] >= TARGET else 'FAIL'} (>= {TARGET}). "
          "The full-warp torque is rare and is checked statically only.", ""]
    val = sa["validation"]
    K += ["## HC-6: Steering arm (DV-2)", "",
          f"1. I = 2 t h^3 / 12 = {sa['I']:.0f} mm^4, A = {sa['A']:.0f} mm^2, M at S = {sa['M_sec'] / 1000:.1f} N m.",
          "2. Section S, beam vs FEA:", "", "| Edge [mm] | Beam [MPa] | FEA [MPa] | Gap |", "|---|---|---|---|",
          *[f"| {v['fibre_mm']:.1f} | {v['hand_MPa']:.1f} | {v['fea_MPa']:.1f} | {v['error_pct']:+.0f} % |" for v in val], "",
          f"   S is {sa['d_sec']:.1f} mm from the load and the arm is {sa['h']:g} mm deep: well inside one depth, where Saint-Venant's principle says the bolt's local bearing field has not yet settled into the linear beam distribution. That is why no plain section exists (DV-2 stands).",
          f"3. Root of the upper ear (TR1): half the across-bolt force on a {sa['L_root']:.1f} mm cantilever, in-plane bending 6 P L / (t b^2) with Kt (R{R_EAR:g}, r/b {R_EAR / sa['h']:.3f}, flat x {FLAT_FACTOR:g}) = {sa['kt']:.2f}: peak {sa['s_root_peak']:.1f} MPa.",
          f"4. FEA steering-arm root peak (von Mises, LC6): {sa['fea_root']:.1f} MPa; the plate model is {(sa['s_root_peak'] / sa['fea_root'] - 1) * 100:+.0f} %.",
          "   The FEA's peak sits where the arm joins the housing: the ring wall bends locally under the arm's moment, which a "
          "cantilever-ear model cannot see. So the ear-root model is not trusted on its own: HC-7 multiplies its stresses by "
          f"FEA / model = {sa['cal']:.2f} (never less than 1).", ""]
    K += ["## HC-7: Ear roots (S-14)", ""]
    for e in er:
        w = e["worst"]
        K += [f"**{e['feature']}** (hole {e['hole']}, {'upper' if e['upper_ear'] else 'lower'} ear of the pair): L {e['L_mm']:.1f} mm, b {e['width_mm']:.1f} mm, t {e['t_mm']:g} mm, Kt out of plane {e['kt_out']:.2f}, in plane {e['kt_in']:.2f}; stresses x {e['cal']:.2f} (the HC-6 calibration).", "",
              f"- Worst case {w['lc']}: along the bolt into this ear {w['P_ax']:.0f} N, across per ear {w['P_in']:.0f} N; root peak {w['s_root']:.1f} MPa -> FoS {e['fos_root']:.2f}.",
              f"- Lug: net section FoS {e['fos_net']:.1f}, shear-out FoS {e['fos_shear']:.1f}.",
              f"- Fatigue (Miner over the duty spectrum, Kf = Kt): k = {e['fatigue_k']:.2f}." if math.isfinite(e["fatigue_k"]) else "- Fatigue: no cycling load reaches this ear in the duty spectrum.",
              f"- Lowest {e['fos_min']:.2f}: {'PASS' if e['fos_min'] >= TARGET else 'FAIL'}.", ""]
        if e["hole"] in INSERT_HOLE_MM:
            K += [f"Hole taken as the {_UD['ball_joints']['lower']['insert']} insert's {INSERT_HOLE_MM[e['hole']]:.1f} mm outside thread, "
                  f"not the CAD's bolt hole. The insert sits in an {_UD['clevis']['lbj_insert_boss']['dia_mm']:g} mm boss on this ear "
                  f"(ear + boss {_UD['clevis']['ear_t_mm'] + _UD['clevis']['lbj_insert_boss']['height_mm']:g} mm for a "
                  f"{_INS['length_mm']:.1f} mm coil; S-18). "
                  f"Thread pull-out: about {ins_pull / 1000:.0f} kN (half its outside cylinder in shear in 7075) against the bolt's "
                  f"{ins_load / 1000:.1f} kN nominal preload: FoS {ins_pull / ins_load:.1f}, or "
                  f"{ins_pull / (ins_load * TORQUE_SCATTER):.1f} at the top of a ±{(TORQUE_SCATTER - 1) * 100:.0f} % torque-friction scatter.", ""]
    T_mid = bh["T_eq"][bh["h_mid"]]
    K += ["## HC-8: Brake heat (S-16)", "",
          f"1. Hardest stop: 0.5 m v^2 x share / 2 = {bh['e_stop'] / 1000:.1f} kJ into one front disc ({bh['m_disc']:.3f} kg): "
          f"+{bh['dT_stop']:.0f} K with no cooling.",
          f"2. Each endurance braking event puts {bh['e_event'] / 1000:.1f} kJ into a front disc; {bh['per_km']:.0f} a km at "
          f"{bhc['mean_speed_kph_assumed']:.0f} km/h is a mean {bh['P'] / 1000:.2f} kW per front disc.",
          f"3. Mean running temperature, between stops (disc area {bh['A_disc'] * 1e4:.0f} cm^2, both faces and the rim): "
          + ", ".join(f"h {h:g}: {T:.0f} C" for h, T in bh["T_eq"].items())
          + f". Small solid discs run hot; h = {bh['h_mid']:g} is used below.",
          f"4. The fit loses {bh['loss_um_per_K']:.2f} µm per K of housing rise: the loosest cup is free at +{bh['open_K'][0]:.0f} K, "
          f"the tightest at +{bh['open_K'][1]:.0f} K. Above that the cups are held by the Loctite 641 alone, so its "
          f"{lim['loctite_641_C']:.0f} C limit, not the fit, bounds the housing.",
          f"5. Housing limit {bh['T_lim_metal']:.0f} C (the lower of Loctite 641 and the assumed 7075 limit) with all-metal "
          f"lock nuts on the upright; {bh['T_lim_nyloc']:.0f} C if any nylon-insert nut sits on it. Upright and hub shed "
          f"(h A)_upright + (h A)_hub = {bh['hA']:.2f} W/K, so {bh['q_allow']:.0f} W at the limit ({bh['q_allow_nyloc']:.0f} W "
          f"at {bh['T_lim_nyloc']:.0f} C). The time constant m c / h A = {bh['tau_s'] / 60:.0f} min (the upright's mass "
          f"only) against a {bh['t_end_s'] / 60:.0f} min endurance: steady state is reached, so the run's length gives no relief.",
          f"6. The mount must hold the heat to {bh['q_allow']:.0f} W across ({T_mid:.0f} - {bh['T_lim_metal']:.0f}) K: "
          f"R >= {bh['r_req']:.2f} K/W. The solid hat of v1.7 gives {bh['r_hat']:.2f} K/W (margin {bh['margin_hat']:.2f}): "
          "no thermal break at all. A floating disc on six bobbins, across the contact range "
          + "; ".join(f"{mat} " + ", ".join(f"{r:.2f}" for r in d.values()) for mat, d in bh["floats"].items())
          + " K/W. Not counted: the caliper's own path into its lugs (it adds heat).",
          f"**Result: S-16 {'PASS' if bh['margin_design'] >= 1 else 'FAIL'} on the estimate** with the {bh['design']} bobbins "
          f"(D23): {bh['r_design_worst']:.2f} K/W even at the high end of the contact range, margin "
          f"{bh['margin_design']:.2f} ({bh['margin_design_nom']:.2f} at {bh['hc_nom']:.0f} W/m^2 K). Stainless bobbins would "
          "fail at that end. The estimate leans on assumed h and contact values: the test plan measures the housing "
          "temperature after an endurance-length run.", ""]
    K += ["## HC-9: Member-end welds and rod-end threads (S-07b)", "",
          "| Member | Weld toe k | Weld root k | Thread k | Insert stripping [kN] | Largest force [kN] | Stripping FoS |",
          "|---|---|---|---|---|---|---|",
          *[f"| {r['member']} | {r['k_toe']:.2f} | {r['k_root']:.2f} | {r['k_thread']:.2f} | {r['strip_N'] / 1000:.1f} | "
            f"{r['f_max'] / 1000:.2f} | {r['strip_fos']:.1f} |" for r in wt["rows"]], "",
          f"k = the factor on every stress range that brings Miner's damage to 1. Lowest: welds {wt['k_weld_min']:.2f} (the "
          f"root, FAT {wt['C']['welds']['fat_root_MPa']:g}, governs), threads {wt['k_thread_min']:.2f}; stripping FoS "
          f"{wt['strip_fos_min']:.1f}. S-07b {'PASS' if min(wt['k_weld_min'], wt['k_thread_min']) >= TARGET else 'FAIL'} "
          f"(>= {TARGET}). The weld classes are IIW values read second-hand (confirm before publishing).", ""]
    K += ["## HC-10: Steering free play (R-06)", "",
          f"Play at the rack {fp['play_mm']:.3f} mm ({fpc['rack_lash_mm_assumed']:g} lash + {fpc['rod_ends_in_chain']} x "
          f"{fpc['rod_end_play_mm_assumed']:g} rod ends) x 360 / {fp['c']:g} mm = {fp['deg_rack']:.2f} deg, + "
          f"{fpc['column_allocation_deg_assumed']:g} deg for the column = **{fp['deg_total']:.2f} deg** against "
          f"{fpc['limit_deg']:g} ({fp['deg_total_kaz']:.2f} deg with the Kaz rack's own ratio). R-06 "
          f"{'PASS' if fp['deg_total'] <= fpc['limit_deg'] else 'FAIL'} on assumed clearances: the test plan measures it.", ""]
    K += ["## HC-11: Chassis tabs and brackets (S-19)", "",
          "| Part | Bending [MPa] | Direct [MPa] | FoS |", "|---|---|---|---|",
          *[f"| {r['part']} | {r['s_bend']:.1f} | {r['s_axial']:.1f} | {r['fos']:.2f} |" for r in tb["rows"]], "",
          f"Lowest FoS {tb['fos_min']:.2f}: S-19 {'PASS' if tb['fos_min'] >= TARGET else 'FAIL'} (>= {TARGET}). The first layout "
          "failed here (lower wishbone tabs 0.8, damper tabs 0.8, rack web 1.3): the forces run across those plates, so "
          "they were widened (D24). The stepped damper tab is checked at both its narrow part (moment arm to the step) "
          "and its wide root (arm to the weld).", ""]
    K += ["## HC-12: The drop link's lever end and the ARB lever (S-20, S-21)", "",
          f"F = {al['T_Nm']:.1f} N m / {al['arm']:.1f} mm = **{al['F']:.0f} N**.", "",
          f"Bolt: d = sqrt(4 x {al['As']:g} / pi) = {al['ds']:.2f} mm; bending 32 F a / (pi d^3) = 32 x {al['F']:.0f} x "
          f"{al['a_b']:g} / (pi x {al['ds']:.2f}^3) = {al['s_b']:.0f} MPa; shear {al['tau']:.1f} MPa; von Mises "
          f"{al['s_vm']:.0f} MPa; **FoS {al['fos_bolt']:.2f}** on {al['bolt_Fty']:g} MPa: S-20 "
          f"{'PASS' if al['fos_bolt'] >= TARGET else 'FAIL'} (>= {TARGET}). Clamping the stack shares the moment with "
          "the faces, so this is conservative.", "",
          f"Lever: at the hub M = {al['F']:.0f} x {al['L_root']:.1f} mm; bending 6 M / (t w^2) = {al['s_lever']:.0f} MPa; "
          f"torsion {al['F']:.0f} x {al['off']:g} mm = {al['F'] * al['off'] / 1000:.1f} N m -> {al['tau_lever']:.0f} MPa; "
          f"von Mises {al['s_lever_vm']:.0f} MPa: FoS {al['fos_lever_hub']:.2f} on {al['lever_Fty']:g} MPa. The end: "
          f"shear-out past the M8 hole, 2 x {al['lig']:.2f} mm x {al['t']:g} mm at 0.577 Fty: FoS {al['fos_lever_end']:.1f}. "
          f"**Lever FoS {al['fos_lever']:.2f}**: S-21 {'PASS' if al['fos_lever'] >= TARGET else 'FAIL'} (>= {TARGET}). "
          "D26 made the lever 8 x 25 mm; D25's 6 x 18 mm placeholder reached 0.95. The blade-to-hub corner is sharp "
          "in the CAD: an R3 there (drawing) keeps Kt well under the hub's margin.", ""]
    write("answer_key.md", K)
    write("README.md", ["# Hand-calculation packets", "",
                        "Twelve problems from the current design, to work on paper and scan into the write-up. Each packet lists its inputs; `answer_key.md` has the worked answers (generated, so it follows the design). Do the packet first.", "",
                        "| Packet | Topic | Requirement |", "|---|---|---|",
                        "| HC-1 | LC1 front-view statics | cross-check of the 3D solver |", "| HC-2 | Rocker | S-06e |",
                        "| HC-3 | Pushrod bracket knee | D5 |", "| HC-4 | Hub spindle in rotating bending | S-15 |",
                        "| HC-5 | ARB bar, body and splined ends | S-09 |", "| HC-6 | Steering arm | DV-2 |", "| HC-7 | Ear roots left sharp in CAD | S-14 |",
                        "| HC-8 | Brake heat into the upright | S-16 |", "| HC-9 | Member-end welds and rod-end threads | S-07b |",
                        "| HC-10 | Steering free play | R-06 |", "| HC-11 | Chassis tabs and brackets | S-19 |",
                        "| HC-12 | Drop link's lever end, ARB lever | S-20, S-21 |", ""])

    res = {"_hardpoints": ver, "_inputs_sha": fingerprint.handcalc_inputs_hash(), "rocker_fos_min": rk["fos_min"], "spindle_fatigue_k_bare": sp["k"], "spindle_fos_static": sp["fos_static"],
           "spindle_fatigue_k": sp["k_ring"], "ear_roots_fos_min": min(e["fos_min"] for e in er), "lbj_insert_pull_fos": ins_pull / ins_load,
           "knee_fos": kn["fos"], "knee_r_need_mm": kn["r_need"], "arb_bar_fos_body": ab["fos_body"], "arb_bar_fos_end": ab["fos_end"],
           "arb_bar_fos_min": min(ab["fos_body"], ab["fos_end"]), "arb_bar_fatigue_fos": ab["fos_fat"],
           "lc1_pushrod_hand_N": h1["f_p"], "lc1_pushrod_3d_N": h1["push3d"],
           "steering_root_peak_MPa": sa["s_root_peak"], "steering_root_fea_MPa": sa["fea_root"],
           "s16_mount_margin": bh["margin_design"], "s16_mount_margin_nominal": bh["margin_design_nom"],
           "s16_hat_margin": bh["margin_hat"], "s16_disc_T_C": T_mid,
           "s16_disc_T_min_C": min(bh["T_eq"].values()), "s16_disc_T_max_C": max(bh["T_eq"].values()),
           "s16_q_allow_W": bh["q_allow"], "s16_r_req_K_W": bh["r_req"], "s16_fit_open_K": bh["open_K"][0],
           "s07b_k_min": min(wt["k_weld_min"], wt["k_thread_min"]), "s07b_weld_k_min": wt["k_weld_min"],
           "s07b_thread_k_min": wt["k_thread_min"], "s07b_strip_fos_min": wt["strip_fos_min"],
           "r06_free_play_deg": fp["deg_total"], "s19_tabs_fos_min": tb["fos_min"],
           "s20_drop_bolt_fos": al["fos_bolt"], "s21_arb_lever_fos": al["fos_lever"], "drop_link_force_max_N": al["F"],
           "ears": [{k: v for k, v in e.items() if k != "worst"} | {"worst_lc": e["worst"]["lc"]} for e in er]}
    (DATA / "handcalc_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"hand calcs: rocker FoS {rk['fos_min']:.2f}; knee FoS {kn['fos']:.2f} (r_i >= {kn['r_need']:.1f} for 1.5); spindle static "
          f"{sp['fos_static']:.2f}, fatigue k {sp['k']:.2f} (ring {sp['k_ring']:.2f}); ARB bar FoS body {ab['fos_body']:.2f}, "
          f"ends {ab['fos_end']:.2f}; ear roots " + ", ".join(f"{e['hole']} {e['fos_min']:.2f}" for e in er) +
          f"; LC1 pushrod {h1['f_p']:.0f} vs {h1['push3d']:.0f} N; brake heat: {bh['design']} bobbins margin "
          f"{bh['margin_design']:.2f} (needs {bh['r_req']:.2f} K/W; the v1.7 hat {bh['margin_hat']:.2f}); welds k {wt['k_weld_min']:.2f}, threads k "
          f"{wt['k_thread_min']:.2f}; free play {fp['deg_total']:.1f} deg")


if __name__ == "__main__":
    main()
