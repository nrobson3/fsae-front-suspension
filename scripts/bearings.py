"""Hub bearings: rating life over a season load spectrum, and the cup fit in the aluminium upright.

Life (ISO 281 basic rating life, 90 % reliability): for each load state in vehicle.json `duty`
the tyre loads are built the way vehicle_calcs builds the load cases, split into bearing-row
loads by the upright free body (upright_loads.split), and turned into an equivalent load per row
for a back-to-back tapered pair:
  induced axial  Fi = 0.5 Fr / Y      (each row pushes the other)
  axial per row  from the external thrust and the induced forces (the standard paired-bearing cases)
  P = Fr if Fa / Fr <= e, else 0.4 Fr + Y Fa
  P_eq = (sum share_k P_k^(10/3))^(3/10)   (distance shares = revolution shares)
  L10 = (C / P_eq)^(10/3) million revolutions -> km
Fit: the cups' interference range at 20 C, the fit lost when the aluminium upright runs hot
(it expands about twice as much as the steel cup), and the hoop stress in the housing at the
largest interference (thick-walled cylinders, Lame), which must stay low for stress-corrosion.
Run: python bearings.py   Writes data/bearings.json and notes/bearings.md
"""
import json
import math

import numpy as np

import stamps
from common import DATA, NOTES, load_hardpoints, load_vehicle, read_version
from forces import read_csv_rows
from suspension import make_corner
from upright_loads import split

G = 9.81
P6_68 = (-0.045, -0.026)        # ISO 286 P6 deviations for a 68 mm hole [mm]
CUP_OD_TOL = (-0.013, 0.0)      # normal-class tapered roller cup OD tolerance, 50-80 mm [mm]


def state_loads(veh, derived, st, outside):
    """(Fx, Fy, Fz) on the LEFT front tyre for one spectrum state (as vehicle_calcs.py builds load cases)."""
    m, L, h = veh["mass_total_kg"], veh["wheelbase_mm"] / 1000, veh["cg_height_mm"] / 1000
    Fz0 = derived["static_corner_load_N"]
    sgn = 1.0 if outside else -1.0
    Fz = st["bump_g"] * Fz0 - m * st["ax_g"] * G * h / L / 2 + sgn * st["ay_g"] * derived["lat_transfer_front_stiff_N_per_g"]
    Fz = max(Fz, 0.0)
    return np.array([st["ax_g"] * Fz, -sgn * st["ay_g"] * Fz, Fz])


def row_axial(fr, ka, Y):
    """Axial load on each row of a back-to-back pair. Row 2 (outboard) carries hub-inboard thrust (ka < 0),
    row 1 carries hub-outboard thrust (standard paired-bearing cases with induced forces 0.5 Fr / Y)."""
    fi = [0.5 * fr[0] / Y, 0.5 * fr[1] / Y]
    k = abs(ka)
    a, b = (1, 0) if ka < 0 else (0, 1)        # a carries the external thrust
    if fi[b] + k >= fi[a]:
        fa = {a: fi[b] + k, b: fi[b]}
    else:
        fa = {a: fi[a], b: fi[a] - k}
    return [fa[0], fa[1]]


def equivalent(fr, fa, e, Y):
    return fr if fr <= 0 or fa / fr <= e else 0.4 * fr + Y * fa


def lame_hoop(delta_mm, d, wall_h, wall_c, Eh, nuh, Ec=207e3, nuc=0.3):
    """Interface pressure and housing hoop stress at the bore for a diametral interference delta."""
    a, b, c = d / 2, d / 2 + wall_h, d / 2 - wall_c
    kh = ((b ** 2 + a ** 2) / (b ** 2 - a ** 2) + nuh) / Eh
    kc = ((a ** 2 + c ** 2) / (a ** 2 - c ** 2) - nuc) / Ec
    p = (delta_mm / d) / (kh + kc)
    return p, p * (b ** 2 + a ** 2) / (b ** 2 - a ** 2)


@stamps.producer("bearings", when=lambda housing_wall_mm=None: housing_wall_mm is None)  # a trial wall: not stamped
def main(housing_wall_mm=None):
    hp, meta = load_hardpoints()
    veh = load_vehicle()
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    members = read_csv_rows(DATA / "members.csv")
    derived = json.loads((DATA / "vehicle_derived.json").read_text())
    hub, duty = veh["hub"], veh["duty"]
    mat = json.loads((DATA / "materials.json").read_text())["7075_T7351_plate"]
    ver = read_version(DATA / "hardpoints.csv")
    pose = corner.pose(0.0)
    fwd = np.cross(hp["WA"] - hp["WC"], [0.0, 0.0, 1.0])
    fwd /= np.linalg.norm(fwd)
    rows = []
    for st in duty["spectrum"]:
        sides = (True, False) if st["ay_g"] else (True,)
        for outside in sides:
            F = state_loads(veh, derived, st, outside)
            loads = [(np.array([F[0], 0.0, F[2]]), pose["cp"])]
            if F[1]:
                loads.append((np.array([0.0, F[1], 0.0]), pose["cp"] - fwd * veh["tyre"]["pneumatic_trail_mm_assumed"]))
            res = split(corner, pose, members, loads, veh, brakes_on=True)
            fr = [float(np.linalg.norm(res["row1"])), float(np.linalg.norm(res["row2"]))]
            ka = -float(np.dot(res["axial"], res["s"]))          # thrust ON the hub, + outboard
            fa = row_axial(fr, ka, hub["Y"])
            P = [equivalent(fr[i], fa[i], hub["e"], hub["Y"]) for i in range(2)]
            rows.append({"state": st["state"] + ("" if not st["ay_g"] else (", outside" if outside else ", inside")),
                         "share": st["share"] / len(sides), "Fr": fr, "Fa": fa, "P": P})
    C = hub["C_kN"] * 1000
    revs_per_km = 1e6 / (2 * math.pi * veh["tyre"]["loaded_radius_mm"])
    life = []
    for i in range(2):
        p_eq = sum(r["share"] * r["P"][i] ** (10 / 3) for r in rows) ** 0.3
        L10 = (C / p_eq) ** (10 / 3) * 1e6 / revs_per_km
        life.append((p_eq, L10))
    target_km = duty["season_km_assumed"] * duty["life_target_seasons"]
    # ---- cup fit
    f = hub["fits"]
    D = hub["od_mm"]
    i_min, i_max = CUP_OD_TOL[0] - P6_68[1], CUP_OD_TOL[1] - P6_68[0]      # interference = cup OD - bore
    hot_loss = (f["alpha_al_per_C"] - f["alpha_steel_per_C"]) * D * f["hot_delta_T_C_assumed"]
    wall = housing_wall_mm or json.loads((DATA / "upright_design.json").read_text())["housing"]["wall_mm"]
    p_max, hoop_max = lame_hoop(i_max, D, wall, f["cup_wall_mm_estimated"], mat["E_MPa"], mat["nu"])
    hot_fit = (i_min - hot_loss, i_max - hot_loss)
    out = {"_hardpoints": ver, "rows": rows, "P_eq_N": [l[0] for l in life], "L10_km": [l[1] for l in life],
           "L10_min_km": min(l[1] for l in life), "L10_min_Mkm": min(l[1] for l in life) / 1e6, "target_km": target_km,
           "fit": {"interference_mm": [i_min, i_max], "hot_loss_mm": hot_loss, "hot_fit_mm": list(hot_fit),
                   "housing_wall_mm": wall, "interface_pressure_MPa": p_max, "hoop_stress_max_MPa": hoop_max,
                   "hoop_fos": mat["Fty_MPa"] / hoop_max}}
    (DATA / "bearings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Hub bearings: life and fits (hardpoints {ver})", "",
          "Generated by `scripts/bearings.py`. Do not edit by hand.", "",
          f"**Bearings:** {hub['bearings']}. C = {hub['C_kN']:g} kN (lowest listed), e = {hub['e']}, Y = {hub['Y']}.", "",
          f"## Rating life over the season spectrum (assumed, `vehicle.json` duty: {duty['season_km_assumed']:.0f} km per season)", "",
          "| Load state | Share of distance | Row 1 Fr / Fa [N] | Row 2 Fr / Fa [N] | P row 1 / row 2 [N] |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['state']} | {r['share']:.3f} | {r['Fr'][0]:.0f} / {r['Fa'][0]:.0f} | {r['Fr'][1]:.0f} / "
                  f"{r['Fa'][1]:.0f} | {r['P'][0]:.0f} / {r['P'][1]:.0f} |")
    md += ["", f"Equivalent load: row 1 {life[0][0]:.0f} N, row 2 {life[1][0]:.0f} N. **L10 life: row 1 "
           f"{life[0][1]:,.0f} km, row 2 {life[1][1]:,.0f} km** against a target of {target_km:,.0f} km "
           f"({duty['life_target_seasons']:g} seasons).", "",
           "## Cup fit in the 7075 upright", "",
           f"- **Housing bore P6, cups normal class:** {i_min * 1000:.0f}-{i_max * 1000:.0f} µm interference at 20 °C.",
           f"- **Running hot (+{f['hot_delta_T_C_assumed']:.0f} °C):** aluminium grows away from the steel cup by "
           f"{hot_loss * 1000:.0f} µm, so the fit becomes {hot_fit[0] * 1000:+.0f} to {hot_fit[1] * 1000:+.0f} µm "
           "(negative = clearance). A press fit alone cannot hold the cups hot: the interference needed (about "
           f"{(hot_loss + 0.010) * 1000:.0f} µm) would roughly double the hoop stress below.",
           f"- **Hoop stress at the largest interference** (Lamé, {wall:g} mm housing wall): interface pressure "
           f"{p_max:.1f} MPa, **hoop stress {hoop_max:.0f} MPa**. Sustained tensile stress in 7075 should stay "
           "low for stress-corrosion; T7351 temper and this level are acceptable.",
           f"- **So:** {f['housing']}; {f['retention']}. Spindle: {f['shaft']}."]
    (NOTES / "bearings.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    main()
