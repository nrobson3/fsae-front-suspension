"""Reference-vehicle calculations: corner loads, ride/wheel/spring rates, roll stiffness and
gradient, ARB sizing, the balance shift caused by roll-centre migration, the steering-lock
requirement, and the contact-patch forces for every load case.

Inputs: data/vehicle.json and data/kinematics_summary.csv + data/sweep_heave.csv (motion ratio,
RC height and its migration, ARB installation ratio; refused if stale). Writes
data/vehicle_derived.json, data/loadcases.csv and notes/vehicle_calcs.md.
Run: python vehicle_calcs.py
"""
import csv
import json
import math

import numpy as np
from scipy.optimize import brentq

import stamps
from common import DATA, NOTES, load_hardpoints, load_materials, load_vehicle, read_version

G = 9.81
LBF_IN = 175.126835  # N/m per lbf/in


def kin_summary():
    """Motion ratio, RC height and ARB ratio from kinematics.py; refuses a stale or missing file."""
    path = DATA / "kinematics_summary.csv"
    if not path.exists():
        raise FileNotFoundError("run kinematics.py first: data/kinematics_summary.csv is missing")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    kin_version = rows[0][1] if rows and rows[0][0] == "# hardpoints" else "unknown"
    hp_version = read_version(DATA / "hardpoints.csv")
    if kin_version != hp_version:
        raise RuntimeError(f"kinematics_summary.csv is for hardpoints {kin_version}, hardpoints.csv is {hp_version}: "
                           "re-run kinematics.py")
    out = {"_version": kin_version}
    for r in rows[2:]:
        try:
            out[r[0]] = float(r[1])
        except (ValueError, IndexError):
            pass
    return out


def rc_vs_heave():
    """(z [mm], RC height above ground [mm]) from the heave sweep."""
    with open(DATA / "sweep_heave.csv", newline="") as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return np.array([float(r["z_mm"]) for r in rows]), np.array([float(r["rc_height_mm"]) for r in rows])


def lateral_transfer_per_g(v, k_arb_front, hrc_f=None):
    """Front and rear lateral load transfer [N per g] from the roll-stiffness model.
    hrc_f overrides the front RC height [m] (to study RC migration)."""
    hrc_f = v["_hrc_f"] if hrc_f is None else hrc_f
    tf, tr = v["track_front_mm"] / 1000, v["track_rear_mm_assumed"] / 1000
    kf = 1 / (1 / (v["_Kw_f"] * tf ** 2 / 2 + k_arb_front) + 1 / (v["_Kt"] * tf ** 2 / 2))
    kr = 1 / (1 / (v["_Kw_r"] * tr ** 2 / 2) + 1 / (v["_Kt"] * tr ** 2 / 2))
    hra = hrc_f + (v["_hrc_r"] - hrc_f) * v["_a_s"] / (v["wheelbase_mm"] / 1000)
    m_roll = v["_ms"] * G * (v["_hs"] - hra)                 # N*m per g
    dF_f = (kf / (kf + kr) * m_roll + v["_msf"] * G * hrc_f + v["_muf"] * G * v["_hu"]) / tf
    dF_r = (kr / (kf + kr) * m_roll + v["_msr"] * G * v["_hrc_r"] + v["_mur"] * G * v["_hu"]) / tr
    return dF_f, dF_r, kf, kr, m_roll


@stamps.producer("vehicle_calcs")
def main():
    v = load_vehicle()
    ks = kin_summary()
    mr = ks["motion_ratio"]
    arb_ratio = abs(ks["arb_twist_per_roll"])

    m, L, h = v["mass_total_kg"], v["wheelbase_mm"] / 1000, v["cg_height_mm"] / 1000
    tf = v["track_front_mm"] / 1000
    mf_axle = m * v["front_weight_fraction"]
    mr_axle = m - mf_axle
    muf, mur = v["unsprung_mass_front_corner_kg"], v["unsprung_mass_rear_corner_kg_assumed"]
    hu = v["unsprung_cg_height_mm_assumed"] / 1000
    Kt = v["tyre"]["vertical_stiffness_N_per_mm"] * 1000
    Rl = v["tyre"]["loaded_radius_mm"] / 1000

    Fz0 = mf_axle * G / 2
    msf_c, msr_c = mf_axle / 2 - muf, mr_axle / 2 - mur
    ms = m - 2 * muf - 2 * mur
    hs = (m * h - 2 * (muf + mur) * hu) / ms

    def wheel_rate(ms_c, f):
        k_ride = (2 * math.pi * f) ** 2 * ms_c
        return k_ride, k_ride * Kt / (Kt - k_ride)        # tyre in series with the wheel rate

    kride_f, kw_f = wheel_rate(msf_c, v["ride_frequency_front_hz"])
    kride_r, kw_r = wheel_rate(msr_c, v["ride_frequency_rear_hz_assumed"])
    k_spring = kw_f / mr ** 2                              # N/m
    steps = [s * LBF_IN / 1000 for s in v["spring_steps_lbf_in"]]   # the catalogue's lb/in steps, N/mm
    pick = min(steps, key=lambda s: abs(s * 1000 - k_spring))
    kw_pick = pick * 1000 * mr ** 2
    kride_pick = kw_pick * Kt / (kw_pick + Kt)
    f_pick = math.sqrt(kride_pick / msf_c) / (2 * math.pi)
    static_defl = msf_c * G / kw_pick * 1000               # wheel travel from spring free length to static
    droop = min(v["wheel_travel_design_mm"]["droop"], ks["droop_to_topout"])   # the damper top-out is the droop stop

    v.update({"_Kw_f": kw_pick, "_Kw_r": kw_r, "_Kt": Kt, "_ms": ms, "_hs": hs, "_a_s": L * (mr_axle - 2 * mur) / ms,
              "_msf": mf_axle - 2 * muf, "_msr": mr_axle - 2 * mur, "_hrc_f": ks["rc_height"] / 1000,
              "_hrc_r": v["roll_centre_rear_mm_assumed"] / 1000, "_muf": 2 * muf, "_mur": 2 * mur, "_hu": hu})

    def tlltd(k_arb, hrc_f=None):
        dF_f, dF_r, *_ = lateral_transfer_per_g(v, k_arb, hrc_f)
        return dF_f / (dF_f + dF_r)

    def roll_grad(k_arb):
        _, _, kf, kr, m_roll = lateral_transfer_per_g(v, k_arb)
        return math.degrees(m_roll / (kf + kr))             # incl. tyre deflection, deg/g

    # ARB: the MIDDLE setting gives the balance target, so the bars adjust it both ways (D26)
    target = v["tlltd_front_target_assumed"]
    mid = v["arb_bar"]["settings"]["middle"]
    tl0 = tlltd(0.0)
    k_base = brentq(lambda k: tlltd(k) - target, 0.0, 5e6) if tl0 < target else 0.0
    if k_base <= 0:
        raise RuntimeError("the springs alone reach the TLLTD target: there is no ARB rate to set")
    k_stiff = k_base / mid
    k_bar_stiff = k_stiff / arb_ratio ** 2                 # the stiff setting's rate at the bar, N m/rad

    # balance shift from front RC migration in heave (only the front RC height changes)
    zs, rcs = rc_vs_heave()
    shift = [(z, tlltd(k_base, float(np.interp(z, zs, rcs)) / 1000)) for z in (-25.0, 0.0, 25.0)]

    dF_f, dF_r, *_ = lateral_transfer_per_g(v, k_stiff)    # load cases: stiff setting = most front transfer
    r_cp = v["hairpin_outside_diameter_m"] / 2 - v["hairpin_line_margin_m_assumed"] - v["tyre"]["tread_width_mm"] / 2000
    d_out = math.degrees(math.asin(L / r_cp))
    rc_axle = math.sqrt(r_cp ** 2 - L ** 2) - tf / 2
    d_in_100 = math.degrees(math.atan(L / (rc_axle - tf / 2)))
    st = v["steering"]
    f_rack = 2 * math.pi * st["driver_torque_at_wheel_Nm_assumed"] / (st["rack_c_factor_mm_per_rev_assumed"] / 1000)
    roll_grad_susp = math.degrees(_susp_roll(v, kw_pick, kw_r, k_stiff, tf))

    # ARB (D26): a solid torsion bar between two stiff levers. The bar sets the rate: three bars, swapped to adjust.
    # Each bar in series with the two levers' bending must give its setting within setting_tolerance_pct (S-22);
    # the levers must be much stiffer than the stiff bar (S-08); the stiff bar must not yield at the largest twist
    # (max-g roll or full warp, S-09; its splined ends are HC-5)
    hp, _ = load_hardpoints()
    bar = v["arb_bar"]
    allm = load_materials()
    steel, lm = allm[bar["material"]], allm[bar["lever_material"]]
    # the body runs between the upsets; each upset runs on to its hub's centre, where the lever takes the torque
    up_len = bar["lever_hub_half_len_mm"] + bar["upset_runout_mm"]
    body_len = 2 * (abs(hp["ARB_P"][1]) - bar["lever_offset_mm"] - up_len)
    bar_len = body_len + 2 * up_len                                               # hub centre to hub centre, mm
    ax_arb = (hp["ARB_A"] - hp["ARB_P"]) / np.linalg.norm(hp["ARB_A"] - hp["ARB_P"])
    r_lev = hp["ARB_L"] - hp["ARB_P"]
    lever_arm = float(np.linalg.norm(r_lev - ax_arb * np.dot(r_lev, ax_arb)))     # mm
    i_lev = bar["lever_blade_t_mm"] * bar["lever_blade_w_mm"] ** 3 / 12          # in-plane bending, mm^4
    l_bend = lever_arm - (bar["end_od_mm"] / 2 + 0.1 + bar["lever_hub_wall_mm"])   # the lever bends from its hub out
    k_lever = 3 * lm["E_MPa"] * i_lev * lever_arm ** 2 / l_bend ** 3 / 1000       # N m/rad at the bar

    def k_bar_of(d):                                                             # N m/rad: body + two upsets
        gj = lambda dd: bar["shear_modulus_MPa"] * math.pi * dd ** 4 / 32       # noqa: E731
        return 1 / (body_len / gj(d) + 2 * up_len / gj(bar["end_od_mm"])) / 1000

    bars = {}
    for n, f in bar["settings"].items():
        d_ = bar["bars_mm"][n]
        k_eff = 1 / (1 / k_bar_of(d_) + 2 / k_lever)
        frac = k_eff / k_bar_stiff
        bars[n] = {"d_mm": d_, "target": f, "fraction": frac, "err_pct": 100 * (frac / f - 1),
                   "tlltd": tlltd(k_stiff * frac), "roll_grad": roll_grad(k_stiff * frac)}
    err_max = max(abs(b["err_pct"]) for b in bars.values())
    lever_ratio = k_lever / k_bar_of(bar["bars_mm"]["stiff"])
    ay_max = max(lc["ay_g"] for lc in v["load_cases"] if lc["type"] == "g")
    # the bar torques use the stiff bar as built (its series rate with the levers), not the setting's target rate:
    # the ends are moved by the suspension, so a stiffer-than-target bar carries proportionally more torque. The roll
    # gradient here and the load cases' lateral transfer keep the target rate on purpose: the roll angle is then about
    # 0.4 % high (conservative), and the load cases (fingerprinted by the FEA and fatigue) stay on the design setting
    k_stiff_built = k_bar_stiff * bars["stiff"]["fraction"]
    t_roll = k_stiff_built * arb_ratio * math.radians(ay_max * roll_grad_susp)   # bar torque at max-g roll, N m
    # full warp: one wheel at full bump, the other at the droop stop. The bar twists as for a roll of
    # atan(dz / track); the stiff setting is the worst
    warp_roll = math.atan((v["wheel_travel_design_mm"]["bump"] + droop) / v["track_front_mm"])
    t_warp = k_stiff_built * arb_ratio * warp_roll
    t_bar = max(t_roll, t_warp)
    tau_bar = 16 * t_bar * 1000 / (math.pi * bar["bars_mm"]["stiff"] ** 3)        # the stiff bar's body
    fos_bar = steel["Fty_MPa"] * 0.577 / tau_bar

    # load cases (left front corner; 'outside' = left wheel in a right-hand turn)
    lcs = []
    r_eff = v["brake"]["pad_effective_radius_mm_assumed"] / 1000
    for lc in v["load_cases"]:
        if lc["type"] == "g":
            ax, ay, bz = lc["ax_g"], lc["ay_g"], lc["bump_g"]
            sgn = -1.0 if lc["wheel"] == "inside" else 1.0
            dz_long = -m * ax * G * h / L / 2               # braking (ax < 0) loads the front
            dz_lat = sgn * ay * dF_f                         # outside wheel gains load, inside loses it
            Fz = bz * Fz0 + dz_long + dz_lat
            Fx = ax * Fz + 0.0                               # equal friction utilisation at each tyre (+0.0: no "-0")
            Fy = -sgn * ay * Fz + 0.0                        # towards the turn centre
            apply_at, kerb_h = "cp", 0.0
            roll = ay * roll_grad_susp * sgn
            z_heave = (Fz - Fz0 - dz_lat) / kw_pick * 1000   # roll part is carried by z_roll
        else:
            Fx, Fy, Fz = (lc["fx_static"] * Fz0 + 0.0, lc["fy_static"] * Fz0 + 0.0, lc["fz_static"] * Fz0)
            apply_at, kerb_h = lc["apply_at"], lc.get("kerb_height_mm", 0.0)
            roll, z_heave = 0.0, (Fz - Fz0) / kw_pick * 1000
        Tb = abs(Fx) * Rl if apply_at == "cp" else 0.0       # brakes react torque only for contact-patch Fx
        z_roll = tf / 2 * math.tan(math.radians(roll)) * 1000
        z_pose = max(-30.0, min(30.0, z_heave + z_roll))     # +30 mm = bump stop, -30 = full droop
        lcs.append({"id": lc["id"], "name": lc["name"], "Fx_N": Fx, "Fy_N": Fy, "Fz_N": Fz, "apply_at": apply_at,
                    "kerb_height_mm": kerb_h, "brake_torque_Nm": Tb, "pad_tangential_force_N": Tb / r_eff,
                    "pad_clamp_force_N": Tb / (2 * v["brake"]["pad_mu_assumed"] * r_eff), "roll_deg": roll,
                    "z_pose_mm": z_pose})

    # bump-stop share in LC1: what the spring cannot carry at +30 mm goes through the bump stop
    lc1 = next(r for r in lcs if r["id"] == "LC1")
    spring_wheel_force_at_bump = Fz0 + kw_pick * v["wheel_travel_design_mm"]["bump"] / 1000
    bump_stop_wheel_force = max(0.0, lc1["Fz_N"] - spring_wheel_force_at_bump)

    derived = {
        "static_corner_load_N": Fz0, "sprung_corner_mass_front_kg": msf_c, "sprung_mass_kg": ms,
        "sprung_cg_height_mm": hs * 1000, "ride_rate_front_N_mm": kride_f / 1000, "wheel_rate_front_req_N_mm": kw_f / 1000,
        "motion_ratio_used": mr, "spring_rate_req_N_mm": k_spring / 1000, "spring_chosen_N_mm": pick,
        "spring_chosen_lbf_in": pick * 1000 / LBF_IN, "wheel_rate_front_N_mm": kw_pick / 1000,
        "ride_frequency_front_hz": f_pick, "static_wheel_deflection_mm": static_defl,
        "spring_slack_beyond_droop_mm": max(0.0, droop - static_defl),
        "wheel_rate_rear_N_mm": kw_r / 1000, "roll_moment_per_g_Nm": lateral_transfer_per_g(v, k_base)[4],
        "tlltd_front_no_arb": tl0, "tlltd_front_target": target,
        "arb_roll_stiffness_baseline_Nm_per_deg": k_base * math.pi / 180,
        "arb_roll_stiffness_stiff_Nm_per_deg": k_stiff * math.pi / 180,
        "arb_installation_ratio": arb_ratio, "arb_bar_stiffness_stiff_Nm_per_deg": k_bar_stiff * math.pi / 180,
        "arb_bar_length_mm": bar_len, "arb_bar_body_length_mm": body_len,
        "arb_lever_stiffness_Nm_per_deg": k_lever * math.pi / 180,
        "arb_lever_stiffness_ratio": lever_ratio, "arb_setting_error_max_pct": err_max,
        **{f"arb_{n}_tlltd": b["tlltd"] for n, b in bars.items()},
        **{f"arb_{n}_roll_gradient_deg_g": b["roll_grad"] for n, b in bars.items()},
        "arb_bar_torque_roll_Nm": t_roll,
        "arb_bar_torque_warp_Nm": t_warp, "arb_bar_torque_max_Nm": t_bar,
        "arb_bar_shear_max_MPa": tau_bar, "arb_bar_fos_shear": fos_bar, "droop_travel_mm": droop,
        "wheel_travel_total_mm": v["wheel_travel_design_mm"]["bump"] + droop, "wheelbase_mm": v["wheelbase_mm"],
        "track_ratio_pct": 100 * min(v["track_front_mm"], v["track_rear_mm_assumed"])
        / max(v["track_front_mm"], v["track_rear_mm_assumed"]),
        "wheel_diameter_mm": v["wheel"]["diameter_in"] * 25.4,
        "roll_gradient_no_arb_deg_g": roll_grad(0.0), "roll_gradient_baseline_deg_g": roll_grad(k_base),
        "roll_gradient_suspension_stiff_deg_g": roll_grad_susp,
        "tlltd_front_bump25": shift[2][1], "tlltd_front_droop25": shift[0][1],
        "lat_transfer_front_stiff_N_per_g": dF_f, "lat_transfer_rear_stiff_N_per_g": dF_r,
        "rack_force_driver_effort_N": f_rack,
        "brake_line_pressure_case": max(lcs, key=lambda r: r["pad_clamp_force_N"])["id"],
        "brake_line_pressure_max_bar": max(r["pad_clamp_force_N"] for r in lcs)          # clamp = p x one side's pistons
        / (v["brake"]["caliper"]["pistons_per_side"] * math.pi * (v["brake"]["caliper"]["piston_dia_in"] * 25.4 / 2) ** 2) * 10,
        "lc1_bump_stop_wheel_force_N": bump_stop_wheel_force,
        "hairpin_outer_angle_req_deg": d_out, "hairpin_inner_angle_100pct_deg": d_in_100,
        "hairpin_turn_radius_rear_axle_m": rc_axle,
    }
    ver = ks["_version"]
    (DATA / "vehicle_derived.json").write_text(json.dumps({"_hardpoints": ver, **derived}, indent=2), encoding="utf-8")
    with open(DATA / "loadcases.csv", "w", newline="") as f:
        f.write(f"# hardpoints {ver}; forces on the LEFT front tyre from the ground or obstacle; vehicle axes x fwd / y left / z up; N\n")
        w = csv.writer(f)
        keys = list(lcs[0].keys())
        w.writerow(keys)
        for r in lcs:
            w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.1f}" for k in keys])

    units = {"_Nm_per_deg": ("N m/deg", 1), "_N_per_g": ("N/g", 0), "_deg_g": ("deg/g", 3), "_N_mm": ("N/mm", 2),
             "_lbf_in": ("lbf/in", 1), "_hz": ("Hz", 2), "_Nm": ("N m", 0), "_mm": ("mm", 1), "_kg": ("kg", 1),
             "_deg": ("deg", 2), "_m": ("m", 2), "_N": ("N", 0), "_MPa": ("MPa", 1), "_pct": ("%", 1)}
    md = [f"# Reference-vehicle calculations (hardpoints {ver})", "",
          "Generated by `scripts/vehicle_calcs.py` from `data/vehicle.json`, `data/kinematics_summary.csv` and "
          "`data/sweep_heave.csv`. Do not edit by hand.", "", "| Quantity | Value | Unit |", "|---|---|---|"]
    for k, val in derived.items():
        suffix = next((u for u in sorted(units, key=len, reverse=True) if k.endswith(u)), None)
        unit, nd = units[suffix] if suffix else ("-", 3)
        md.append(f"| {k} | {val} | - |" if isinstance(val, str) else f"| {k} | {val:.{nd}f} | {unit} |")
    md += ["", f"Roll gradient target (PLAN s2): about {v['roll_gradient_target_deg_per_g']} deg/g. Baseline (middle setting, "
           f"{target:.0%} front TLLTD): {roll_grad(k_base):.2f} deg/g; springs only: {roll_grad(0.0):.2f} deg/g "
           "(both include tyre deflection).",
           "", "## ARB settings: three bars (D26; the middle bar gives the TLLTD target)", "",
           f"Solid {bar['material'].split('_')[0]} bars: a {body_len:.0f} mm body between {bar['end_od_mm']:g} mm upset ends "
           f"({bar_len:.0f} mm hub to hub), in series with two "
           f"{bar['lever_blade_t_mm']:g} x {bar['lever_blade_w_mm']:g} mm levers ({k_lever * math.pi / 180:.0f} N m/deg each "
           f"at the bar). Disconnected = drop links off: front TLLTD {tl0:.1%}, roll gradient {roll_grad(0.0):.2f} deg/g.", "",
           "| Setting | Target fraction of the stiff rate | Bar body [mm] | Fraction with the levers | Error | Front TLLTD | "
           "Roll gradient [deg/g] |", "|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {b['target']:.2f} | {b['d_mm']:g} | {b['fraction']:.3f} | {b['err_pct']:+.1f} % | {b['tlltd']:.1%} | "
           f"{b['roll_grad']:.2f} |" for n, b in bars.items()]
    md += ["", "## Balance shift from front roll-centre migration (baseline setting, front RC only)", "",
           "| Front wheel travel [mm] | Front RC above ground [mm] | Front TLLTD |", "|---|---|---|"]
    md += [f"| {z:+.0f} | {np.interp(z, zs, rcs):.1f} | {t:.1%} |" for z, t in shift]
    md += ["", "## Load cases (left front corner)", "",
           "Applied at: cp = contact patch (brake torque reacted by the caliper), wc = wheel centre (no brake torque), "
           "kerb = outer sidewall at the stated height. Load cases use the stiff ARB setting (most front load transfer). "
           f"LC6 (steering effort, wheels blocked) loads one tie rod with the full rack force, {f_rack:.0f} N, and is "
           "checked directly in forces.py.", "",
           "| LC | Case | Fx [N] | Fy [N] | Fz [N] | At | Brake torque [N m] | Pad tangential force [N] | Pad clamp force [N] | "
           "Roll [deg] | Pose z [mm] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in lcs:
        md.append(f"| {r['id']} | {r['name']} | {r['Fx_N']:.0f} | {r['Fy_N']:.0f} | {r['Fz_N']:.0f} | {r['apply_at']} | "
                  f"{r['brake_torque_Nm']:.0f} | {r['pad_tangential_force_N']:.0f} | {r['pad_clamp_force_N']:.0f} | "
                  f"{r['roll_deg']:.2f} | {r['z_pose_mm']:.1f} |")
    (NOTES / "vehicle_calcs.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for k, val in derived.items():
        print(f"{k:42s} {val:>10s}" if isinstance(val, str) else f"{k:42s} {val:10.3f}")
    for r in lcs:
        print({k: (round(x, 1) if isinstance(x, float) else x) for k, x in r.items()})


def _susp_roll(v, kw_f, kw_r, k_arb, tf):
    """Body roll relative to the wheels [rad per g] (springs + ARB only, no tyre)."""
    tr = v["track_rear_mm_assumed"] / 1000
    k_susp = (kw_f * tf ** 2 / 2 + k_arb) + kw_r * tr ** 2 / 2
    _, _, _, _, m_roll = lateral_transfer_per_g(v, k_arb)
    return m_roll / k_susp


if __name__ == "__main__":
    main()
