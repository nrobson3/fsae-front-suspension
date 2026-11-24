"""Step B hand calcs (PLAN s17): brake heat into the upright (S-16), weld toes and rod-end threads (S-07b),
steering free play (R-06). Closed-form methods only; handcalcs.py writes the packets and the answer key.

  brake_heat     energy per stop and the endurance's mean braking power per front disc; the disc's running
                 temperature (convection over a range of h, plus radiation); the heat the upright can shed
                 at its temperature limit; the thermal resistance the disc mount needs, against a solid hat
                 (as the CAD has it) and a floating disc on bobbins
  welds_threads  member forces in every duty state (the 6 x 6 solve at static ride height), their ranges in
                 the duty cycles, and Miner's rule on three S-N lines: the member-end weld toe (IIW FAT 63)
                 and root (FAT 36 on the throat), and the rod-end shank's first engaged thread (EN 1993-1-9
                 detail 50 on the stress area); the insert thread's static stripping
  free_play      the play in the rack and one tie-rod chain, taken to the steering wheel, plus the
                 steering column's allocation
Inputs: data/checks_week3b.json (assumptions marked _assumed, each limit with its source).
"""
import csv
import json
import math

import numpy as np
from scipy.optimize import brentq

from common import DATA, unit
from forces import read_csv_rows, solve_links

G = 9.81
SIGMA_SB = 5.670e-8
# disc and hub geometry in the CAD builder (scripts/fusion/fsae_parts.py: DISC_T, DISC_RI, DISC_HAT of v1.7,
# HUB_FLANGE, BRG_ID), copied here
DISC_T_MM, DISC_RI_MM, HAT_R_MM = 4.0, 56.0, (56.0, 60.0)
HUB_FLANGE_R_MM, HUB_FLANGE_W_MM, SPINDLE_R_MM = 60.0, 14.0, 20.0


def inputs():
    return json.loads((DATA / "checks_week3b.json").read_text(encoding="utf-8"))


def load_case(lc_id):
    with open(DATA / "loadcases.csv", newline="") as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return next(r for r in rows if r["id"] == lc_id)


# ------------------------------------------------------------------ S-16 brake heat
def disc_equilibrium(P, A, h, eps, Ta):
    """Disc temperature [C] where convection + radiation shed the mean power P [W] (no input: ambient)."""
    if P <= 0:
        return Ta
    TaK = Ta + 273.15

    def excess(T):
        return h * A * (T - TaK) + eps * SIGMA_SB * A * (T ** 4 - TaK ** 4) - P
    if excess(3000.0) < 0:
        raise ValueError(f"{P:.0f} W cannot be shed below 3000 K with h {h:g} and emissivity {eps:g}")
    return brentq(excess, TaK, 3000.0) - 273.15


def brake_heat(veh, upright_area_mm2, upright_mass_kg):
    C = inputs()
    b, lim = C["brake_heat"], C["limits"]
    m = veh["mass_total_kg"]
    Ta = b["ambient_C_assumed"]
    lc3 = load_case("LC3")
    ax3 = next(lc["ax_g"] for lc in veh["load_cases"] if lc["id"] == "LC3")
    share = 2 * abs(float(lc3["Fx_N"])) / (m * G * abs(ax3))
    ro, ri, t = veh["brake"]["disc_od_mm_assumed"] / 2000, DISC_RI_MM / 1000, DISC_T_MM / 1000
    m_disc = math.pi * (ro ** 2 - ri ** 2) * t * b["disc_density_kg_m3"]
    c_disc = b["disc_c_J_kgK"]
    A_disc = 2 * math.pi * (ro ** 2 - ri ** 2) + 2 * math.pi * ro * t       # both faces and the rim
    v0 = b["hardest_stop_kph_assumed"] / 3.6
    e_stop = 0.5 * m * v0 ** 2 * share / 2
    dT_stop = e_stop / (m_disc * c_disc)
    v1, v2 = (s / 3.6 for s in b["event_speed_kph_assumed"])
    e_event = 0.5 * m * (v1 ** 2 - v2 ** 2) * share / 2
    per_km = veh["duty"]["cycle_model"]["events_per_km_assumed"]["brake"]
    t_km = 1000.0 / (b["mean_speed_kph_assumed"] / 3.6)
    P = e_event * per_km / t_km
    h_mid = b["disc_h_design_W_m2K_assumed"]
    T_eq = {h: disc_equilibrium(P, A_disc, h, b["disc_emissivity_assumed"], Ta)
            for h in sorted(set(b["disc_h_W_m2K_assumed"]) | {h_mid})}
    # the cup fit (bearings.json): interference lost per kelvin of housing rise
    fit = json.loads((DATA / "bearings.json").read_text())["fit"]
    hot_dT = veh["hub"]["fits"]["hot_delta_T_C_assumed"]
    loss_per_K = fit["hot_loss_mm"] / hot_dT
    open_K = [i / loss_per_K for i in fit["interference_mm"]]
    # the limit for the upright and hub (one temperature: the bearings between them conduct well), and the heat
    # their surfaces shed there: the upright's, and the hub flange's inboard face and rim
    T_lim_metal = min(lim["al7075_T73_long_term_C_assumed"], lim["loctite_641_C"])
    T_lim_nyloc = min(T_lim_metal, lim["nylon_insert_nut_C"])
    A_up = upright_area_mm2 / 1e6
    fr, fw = HUB_FLANGE_R_MM / 1000, HUB_FLANGE_W_MM / 1000
    A_hub = math.pi * (fr ** 2 - (SPINDLE_R_MM / 1000) ** 2) + 2 * math.pi * fr * fw
    hA = b["upright_h_W_m2K_assumed"] * A_up + b["hub_h_W_m2K_assumed"] * A_hub
    q_allow = hA * (T_lim_metal - Ta)
    q_allow_nyloc = hA * (T_lim_nyloc - Ta)
    tau = upright_mass_kg * b["upright_c_J_kgK"] / hA                     # the hub's mass not counted: shorter, conservative
    t_end = b["endurance_km"] * t_km
    r_req = max(0.0, (T_eq[h_mid] - T_lim_metal) / q_allow)                 # 0: the disc runs below the limit
    dT_d = T_eq[h_mid] - T_lim_metal
    # mounts: the solid steel hat of v1.7, and a floating disc on bobbins over the contact-conductance range
    r_mean, wall = sum(HAT_R_MM) / 2000, (HAT_R_MM[1] - HAT_R_MM[0]) / 1000
    r_hat = (b["hat_length_mm_assumed"] / 1000) / (b["disc_k_W_mK_assumed"] * 2 * math.pi * r_mean * wall)
    f = b["float"]
    a_c = math.pi * (f["contact_dia_mm"] / 2000) ** 2
    hcs = sorted(b["contact_conductance_W_m2K_assumed"])

    def r_float(k, hc):
        return (2 / (hc * a_c) + (f["path_mm"] / 1000) / (k * a_c)) / f["bobbins"]
    floats = {mat: {hc: r_float(o["k_W_mK"], hc) for hc in hcs} for mat, o in f["options"].items()}
    design = floats[f["design"]]
    hc_nom = hcs[len(hcs) // 2]

    def margin(r):
        return r / r_req if r_req else math.inf
    return {"share": share, "m_disc": m_disc, "A_disc": A_disc, "e_stop": e_stop, "dT_stop": dT_stop, "e_event": e_event,
            "per_km": per_km, "P": P, "T_eq": T_eq, "h_mid": h_mid, "Ta": Ta, "open_K": open_K, "loss_um_per_K": loss_per_K * 1000,
            "fit_um": [i * 1000 for i in fit["interference_mm"]],
            "T_lim_metal": T_lim_metal, "T_lim_nyloc": T_lim_nyloc, "A_up": A_up, "A_hub": A_hub, "hA": hA,
            "q_allow": q_allow, "q_allow_nyloc": q_allow_nyloc, "tau_s": tau, "t_end_s": t_end, "r_req": r_req,
            "r_hat": r_hat, "q_hat": dT_d / r_hat, "margin_hat": margin(r_hat), "floats": floats, "hcs": hcs, "hc_nom": hc_nom,
            "design": f["design"], "r_design_worst": design[hcs[-1]], "r_design_nom": design[hc_nom],
            "q_design_worst": dT_d / design[hcs[-1]], "margin_design": margin(design[hcs[-1]]),
            "margin_design_nom": margin(design[hc_nom]), "C": C}


# ------------------------------------------------------------------ S-07b weld toes and rod-end threads
def duty_member_forces(corner, veh):
    """{state: member forces [N], tension +} for every duty state, at static ride height (as duty_loads.py)."""
    duty = json.loads((DATA / "duty_loads.json").read_text())
    members = read_csv_rows(DATA / "members.csv")
    pose = corner.pose(0.0)
    hp = corner.hp
    fwd = unit(np.cross(hp["WA"] - hp["WC"], [0.0, 0.0, 1.0]))
    acts_at = {"Pushrod": "LCA_O"}
    out = {}
    for name, st in duty["states"].items():
        F = np.array(st["tyre_N"], float)
        loads = [(np.array([F[0], 0.0, F[2]]), pose["cp"])]
        if F[1]:
            loads.append((np.array([0.0, F[1], 0.0]), pose["cp"] - fwd * veh["tyre"]["pneumatic_trail_mm_assumed"]))
        out[name] = sum(solve_links(pose, members, Fi, acts_at, ati)[0] for Fi, ati in loads)
    return members, duty["cycles"], out


def miner_k(ranges, fat, m, n_ref):
    """Factor on stress that brings Miner's damage to 1 on an S-N line through fat at n_ref, slope m, no knee."""
    D = sum(n * (ds / fat) ** m / n_ref for ds, n in ranges if ds > 0)
    return (D ** (-1 / m) if D > 0 else math.inf), D


def welds_threads(corner, veh, mats):
    C = inputs()
    w, th = C["welds"], C["threads"]
    members, cycles, F = duty_member_forces(corner, veh)
    if {(float(mb["bolt_d_mm"]), float(mb["rod_end_pitch_mm"])) for mb in members} != {(8.0, 1.25)}:
        raise RuntimeError("thread check assumes M8 x 1.25 rod-end shanks on every member (stress area in checks_week3b.json)")
    checks = {r["member"]: r for r in read_csv_rows(DATA / "member_checks.csv")}
    rows = []
    for j, mb in enumerate(members):
        D, t = float(mb["tube_od_mm"]), float(mb["tube_wall_mm"])
        A_tube = math.pi / 4 * (D ** 2 - (D - 2 * t) ** 2)
        a = t * w["throat_over_wall"]
        A_throat = math.pi * D * a                                        # the weld runs round the tube's outside
        dF = [(abs(F[c["a"]][j] - F[c["b"]][j]), c["n"]) for c in cycles]
        k_toe, d_toe = miner_k([(f / A_tube, n) for f, n in dF], w["fat_toe_MPa"], w["m"], w["n_ref"])
        k_root, d_root = miner_k([(f / A_throat, n) for f, n in dF], w["fat_root_MPa"], w["m"], w["n_ref"])
        k_thr, d_thr = miner_k([(f / th["stress_area_mm2"], n) for f, n in dF], th["fat_MPa"], th["m"], th["n_ref"])
        worst = max(dF, key=lambda x: x[0])
        d_bolt = float(mb["bolt_d_mm"])
        L_e = th["engagement_d"] * d_bolt
        strip = 0.577 * mats[th["insert_material"]]["Fty_MPa"] * th["strip_shear_fraction"] * math.pi * d_bolt * L_e
        f_max = max(abs(float(checks[mb["member"]]["max_tension_N"])), abs(float(checks[mb["member"]]["max_compression_N"])))
        rows.append({"member": mb["member"], "A_tube": A_tube, "A_throat": A_throat, "dF_max": worst[0], "n_at_max": worst[1],
                     "k_toe": k_toe, "k_root": k_root, "k_weld": min(k_toe, k_root), "k_thread": k_thr,
                     "L_e": L_e, "strip_N": strip, "f_max": f_max, "strip_fos": strip / f_max})
    return {"rows": rows, "k_weld_min": min(r["k_weld"] for r in rows), "k_thread_min": min(r["k_thread"] for r in rows),
            "strip_fos_min": min(r["strip_fos"] for r in rows), "cycles": cycles, "C": C,
            "n_total": sum(c["n"] for c in cycles)}


# ------------------------------------------------------------------ S-19 chassis tabs and brackets
def tab_strength(rocker_pivot_N, damper_N, rack_N, mats):
    """Each tab pair or bracket web (chassis_tabs.build) as a cantilever from its weld at the tube: the joint force
    split into its parts along and across the plate, in-plane bending at the root (6 M / (t w^2), w = 2 r) plus the
    direct stress, against the weld-affected 4130N yield. Wishbone tabs take their member's largest force along
    the leg; the rocker bracket, damper tabs and rack web take theirs wholly across the plate unless the
    direction is known (the rack force runs along the rack)."""
    import chassis_tabs as T
    static, rack, _, _ = T.build()
    hp = {k: tuple(v) for k, v in T.load()[0].items()}
    checks = {r["member"]: r for r in read_csv_rows(DATA / "member_checks.csv")}
    tube = mats["4130N_tube"]
    allow = tube["Fty_MPa"] * tube["weld_knockdown"]
    legs = {"Upper wishbone front tab": ("UCA_F", "UCA_O", "UCA front leg"), "Upper wishbone rear tab": ("UCA_R", "UCA_O", "UCA rear leg"),
            "Lower wishbone front tab": ("LCA_F", "LCA_O", "LCA front leg"), "Lower wishbone rear tab": ("LCA_R", "LCA_O", "LCA rear leg")}
    groups = {}
    for p in static:
        if p["kind"] != "plate" or p["name"] == "Rack bracket pad":
            continue
        groups.setdefault(p["name"].rsplit(" ", 1)[0] if p["name"].endswith((" A", " B")) else p["name"], []).append(p)
    rows = []
    for name, plates in groups.items():
        p = plates[0]
        # the moment arm runs from the bolt (the load) to this plate's root: for a stepped tab's wide part that
        # is the whole tab, not the wide part alone
        load_at = groups[name.replace(" root", "")][0]["p1"] if " root" in name else p["p1"]
        e = np.subtract(p["p2"], load_at)
        L = float(np.linalg.norm(e))
        e = e / L
        if name in legs:
            pin, o, member = legs[name]
            F = max(abs(float(checks[member]["max_tension_N"])), abs(float(checks[member]["max_compression_N"])))
            d = unit(np.subtract(hp[o], hp[pin]))
            along = abs(float(np.dot(d, e)))
            basis = f"{member}, largest force, along the leg"
        elif name == "Rack bracket web":
            F, along = rack_N, abs(float(e[1]))
            basis = "LC6 rack force along the rack, all on one bracket"
        else:
            F, along = (rocker_pivot_N, 0.0) if name.startswith("Rocker") else (damper_N, 0.0)
            basis = "rocker pivot reaction, LC1 with the bump stop" if name.startswith("Rocker") else "damper force, LC1 with the bump stop"
        F_ax, F_tr = F * along, F * math.sqrt(max(0.0, 1 - along ** 2))
        n, w, t = len(plates), 2 * p["r"], p["t"]
        s_b = 6 * F_tr * L / (n * t * w ** 2)
        s_a = F_ax / (n * t * w)
        rows.append({"part": name, "plates": n, "t_mm": t, "w_mm": w, "L_mm": L, "F_N": F, "s_bend": s_b, "s_axial": s_a,
                     "fos": allow / (s_b + s_a), "basis": basis})
    return {"rows": rows, "allow": allow, "fos_min": min(r["fos"] for r in rows)}


# ------------------------------------------------------------------ S-20, S-21 the drop link's lever end, the ARB lever
def arb_lever_joint(vder, veh, ud, mats, hp):
    """The ARB's largest torque (vehicle_derived: full warp or max-g roll, stiff blade) through the lever into the
    drop link: F = T / the lever's moment arm about the bar for a force along the link.
    S-20: the lever-end bolt (M8 10.9) in single shear is a cantilever from the blade's face to the ball's centre
    (half the ball + the spacer): bending on the thread's stress area plus direct shear, von Mises.
    S-21: the lever bends at its hub (6 M / (t w^2) on the blade's section, 4130N) and twists, because the link
    pulls lever_offset_mm beside the blade's mid-plane (rectangle in torsion, T (3 a + 1.8 b) / (a^2 b^2));
    combined as von Mises, conservatively at the same point; and the end's shear-out past the M8 hole."""
    bar = veh["arb_bar"]
    T = vder["arb_bar_torque_max_Nm"] * 1000.0                        # N mm
    ax = unit(hp["ARB_A"] - hp["ARB_P"])
    r = np.asarray(hp["ARB_L"], float) - np.asarray(hp["ARB_P"], float)
    d = unit(hp["ARB_R"] - hp["ARB_L"])
    arm = abs(float(np.dot(np.cross(ax, r), d)))
    F = T / arm
    As = inputs()["threads"]["stress_area_mm2"]
    ds = math.sqrt(4 * As / math.pi)
    rod_w, sp = 12.0, ud["clevis"]["spacer_mm"]                        # fsae_parts.ROD_END_W; misalignment spacer
    a_b = rod_w / 2 + sp
    s_b = 32 * F * a_b / (math.pi * ds ** 3)
    tau = F / As
    s_vm = math.sqrt(s_b ** 2 + 3 * tau ** 2)
    bolt = mats["bolt_10.9"]
    t, w, off = bar["lever_blade_t_mm"], bar["lever_blade_w_mm"], bar["lever_offset_mm"]
    r_hub = bar["end_od_mm"] / 2 + 0.1 + bar["lever_hub_wall_mm"]     # fsae_parts.arb_lever: bore + wall
    L_root = float(np.linalg.norm(r - ax * np.dot(r, ax))) - r_hub
    s_lever = 6 * F * L_root / (t * w ** 2)
    a_, b_ = max(t, w), min(t, w)
    tau_lever = F * off * (3 * a_ + 1.8 * b_) / (a_ ** 2 * b_ ** 2)
    s_lever_vm = math.sqrt(s_lever ** 2 + 3 * tau_lever ** 2)
    lm = mats[bar["lever_material"]]
    lig = w / 2 - 4.05                                                 # the lever's end beyond its M8 hole
    fos_end = 2 * lig * t * 0.577 * lm["Fty_MPa"] / F                  # shear-out of the bolt through the end
    return {"T_Nm": T / 1000, "case": "warp" if vder["arb_bar_torque_warp_Nm"] >= vder["arb_bar_torque_roll_Nm"] else "roll",
            "arm": arm, "F": F, "As": As, "ds": ds, "a_b": a_b, "s_b": s_b, "tau": tau, "s_vm": s_vm,
            "bolt_Fty": bolt["Fty_MPa"], "fos_bolt": bolt["Fty_MPa"] / s_vm, "t": t, "w": w, "off": off, "r_hub": r_hub,
            "L_root": L_root, "s_lever": s_lever, "tau_lever": tau_lever, "s_lever_vm": s_lever_vm,
            "lever_Fty": lm["Fty_MPa"], "fos_lever_hub": lm["Fty_MPa"] / s_lever_vm, "lig": lig, "fos_lever_end": fos_end,
            "fos_lever": min(lm["Fty_MPa"] / s_lever_vm, fos_end)}


# ------------------------------------------------------------------ R-06 steering free play
def free_play(veh):
    C = inputs()["free_play"]
    c = veh["steering"]["rack_c_factor_mm_per_rev_assumed"]
    play = C["rack_lash_mm_assumed"] + C["rod_ends_in_chain"] * C["rod_end_play_mm_assumed"]
    deg_rack = play / c * 360
    kaz = 4.71 * 25.4
    return {"c": c, "play_mm": play, "deg_rack": deg_rack, "deg_total": deg_rack + C["column_allocation_deg_assumed"],
            "deg_total_kaz": play / kaz * 360 + C["column_allocation_deg_assumed"], "kaz_c": kaz, "C": C}
