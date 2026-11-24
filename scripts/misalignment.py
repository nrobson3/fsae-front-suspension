"""Spherical-joint misalignment over the full heave x roll x steer envelope.

A rod end or spherical bearing turns freely about its bolt (bore) axis but only tilts a limited
angle away from it. For each joint the bolt axis is fixed in one body and the bore axis in the
other; they coincide at static by design, and the misalignment at a pose is the angle between
the two axes after each body moves. Bolt-axis choices follow common FSAE practice:

  ball joints        bolt along the kingpin axis (steering turns the ball about its bore)
  inboard rod ends   bolt normal to the leg, in the plane of the leg and the wishbone pivot axis
  tie-rod ends       bolt normal to the tie rod, as close to the kingpin (outer) / vertical (inner)
  pushrod ends       bolt normal to the pushrod, as close to the LCA pivot axis (lower) /
                     rocker axis (upper) as possible
  damper eyes        bolt normal to the damper, as close to the rocker axis as possible

Ratings come from data/joints_misalignment.csv. Run: python misalignment.py
Writes data/misalignment.csv and notes/misalignment.md.
"""
import csv
import math

import numpy as np

import stamps
from common import DATA, NOTES, EZ, load_hardpoints, load_vehicle, read_version, unit
from poses import left_transforms, sweep_grid
from suspension import make_corner


def normal_to(v, hint):
    """Unit vector normal to v, as close as possible to `hint`."""
    v = unit(v)
    n = hint - np.dot(hint, v) * v
    return unit(n)


def joint_defs(h):
    k = unit(h["UCA_O"] - h["LCA_O"])
    pu = unit(h["UCA_R"] - h["UCA_F"])
    pl = unit(h["LCA_R"] - h["LCA_F"])
    rk = unit(h["RKR_A"] - h["RKR_P"])
    tie = h["TR_O"] - h["TR_I"]
    push = h["PR_I"] - h["PR_O"]
    dmp = h["DMP_R"] - h["DMP_C"]
    drop = h["ARB_R"] - h["ARB_L"]
    # (name, rating key, bolt body, bore body, static axis)
    return [
        ("Upper ball joint", "ball_joint", "upright", "uca", k),
        ("Lower ball joint", "ball_joint_lower", "upright", "lca", k),
        ("UCA front inboard rod end", "rod_end", "chassis", "uca", normal_to(h["UCA_O"] - h["UCA_F"], pu)),
        ("UCA rear inboard rod end", "rod_end", "chassis", "uca", normal_to(h["UCA_O"] - h["UCA_R"], pu)),
        ("LCA front inboard rod end", "rod_end", "chassis", "lca", normal_to(h["LCA_O"] - h["LCA_F"], pl)),
        ("LCA rear inboard rod end", "rod_end", "chassis", "lca", normal_to(h["LCA_O"] - h["LCA_R"], pl)),
        ("Tie rod outer rod end", "rod_end", "upright", "tie_rod", normal_to(tie, k)),
        ("Tie rod inner joint", "rack_ball_joint", "rack", "tie_rod", normal_to(tie, EZ)),
        ("Pushrod lower rod end", "rod_end", "lca", "pushrod", normal_to(push, pl)),
        ("Pushrod upper rod end", "rod_end", "rocker", "pushrod", normal_to(push, rk)),
        ("Damper chassis eye", "damper_eye", "chassis", "damper_body", normal_to(dmp, rk)),
        ("Damper rocker eye", "damper_eye", "rocker", "damper_shaft", normal_to(dmp, rk)),
        ("Drop link rocker end", "rod_end", "rocker", "drop_link", normal_to(drop, rk)),
        ("Drop link lever end", "rod_end_captured", "arb_lever", "drop_link", normal_to(drop, unit(h["ARB_A"] - h["ARB_P"]))),
    ]


def rot(T, body):
    if body in ("chassis", "rack"):
        return np.eye(3)
    return T[body][:3, :3]


def worst_angles(hp, meta, veh, rack_stop):
    """Worst misalignment [deg] and where it occurs, per joint, over the sweep grid."""
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    pushrod_on = corner.pushrod_on
    grid = sweep_grid(veh, rack_stop)
    joints = joint_defs(hp)
    worst = {j[0]: (-1.0, (0.0, 0.0)) for j in joints}
    for zl, zr, s in grid:
        for z, rack in ((zl, s), (zr, -s)):  # each corner sees its own travel; right = mirror at -rack
            T, _ = left_transforms(corner, z, rack)
            for name, key, b_bolt, b_bore, a0 in joints:
                if b_bolt == "lca" and pushrod_on == "upright":
                    b_bolt = "upright"
                ab, ar = rot(T, b_bolt) @ a0, rot(T, b_bore) @ a0
                ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(ab, ar))))))
                if ang > worst[name][0]:
                    worst[name] = (ang, (z, rack))
    return joints, worst, len(grid)


@stamps.producer("misalignment")
def main():
    hp, meta = load_hardpoints()
    veh = load_vehicle()
    ratings = {}
    with open(DATA / "joints_misalignment.csv", newline="") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            ratings[r["key"]] = r
    rack_stop = None
    with open(DATA / "kinematics_summary.csv") as f:
        for line in f:
            if line.startswith("rack_travel_at_stop,"):
                rack_stop = float(line.split(",")[1])
    joints, worst, n_poses = worst_angles(hp, meta, veh, rack_stop)
    rows = []
    for name, key, b_bolt, b_bore, _ in joints:
        ang, at = worst[name]
        lim = float(ratings[key]["max_misalignment_deg"])
        margin = lim - ang
        rows.append({"joint": name, "part": ratings[key]["part"], "worst_deg": ang, "limit_deg": lim,
                     "margin_deg": margin, "status": "PASS" if margin >= 2.0 else "FAIL",
                     "at_z_mm": at[0], "at_rack_mm": at[1], "rating_source": ratings[key]["source"]})
    with open(DATA / "misalignment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# hardpoints", read_version(DATA / "hardpoints.csv")])
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([f"{v:.2f}" if isinstance(v, float) else v for v in r.values()])
    ver = read_version(DATA / "hardpoints.csv")
    md = [f"# Joint misalignment over the full envelope (hardpoints {ver})", "",
          "Generated by `scripts/misalignment.py`. Do not edit by hand. Envelope: heave -30..+30 mm, roll +/-2 deg, "
          f"rack 0..+/-{rack_stop:.1f} mm (the steering stop), {n_poses} poses, both corners. "
          "Requirement: worst angle at least 2 deg inside the part's rated misalignment.", "",
          "| Joint | Part | Worst [deg] | Rated [deg] | Margin [deg] | Status | At z, rack [mm] |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['joint']} | {r['part']} | {r['worst_deg']:.1f} | {r['limit_deg']:.0f} | {r['margin_deg']:.1f} | "
                  f"{r['status']} | {r['at_z_mm']:.0f}, {r['at_rack_mm']:.1f} |")
    md += ["", "Rated angles and their sources: `data/joints_misalignment.csv`."]
    (NOTES / "misalignment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
