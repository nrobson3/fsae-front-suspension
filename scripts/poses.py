"""Rigid-body poses for the CAD assembly, computed by the kinematic solver.

Fusion's joint solver does not drive a closed spatial loop (two wishbones + tie rod + pushrod +
rocker) reliably, so the CAD is posed from this solver instead: one source of truth for the
analysis and the model. A pose is (z_left, z_right, rack); each moving part gets a 4x4 world
transform (mm, row-major) that takes it from its static position to the pose.

Right-corner parts are built as mirror images of the left ones, so their transform is
M T_left(z_right, -rack) M with M = diag(1, -1, 1).

    python poses.py              # writes data/poses.json for the standard sweep grid
"""
import json
import math

import numpy as np

import stamps
from common import DATA, load_hardpoints, load_vehicle, read_version, rotvec_matrix, unit
from suspension import make_corner

M4 = np.diag([1.0, -1.0, 1.0, 1.0])


def rigid(R, p0, p1):
    """4x4 transform: rotate by R about p0, then move p0 to p1."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p1 - R @ p0
    return T


def align(a, b):
    """Smallest rotation taking direction a onto direction b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3)
    return rotvec_matrix(v / s * math.atan2(s, c))


def axis_rotation(p, axis_a, axis_b, p_new):
    """Rotation about the fixed axis a->b that takes point p to p_new (wishbones)."""
    k = unit(axis_b - axis_a)
    r0 = p - axis_a
    r1 = p_new - axis_a
    r0 -= np.dot(r0, k) * k
    r1 -= np.dot(r1, k) * k
    ang = math.atan2(np.dot(np.cross(r0, r1), k), np.dot(r0, r1))
    return rigid(rotvec_matrix(k * ang), axis_a, axis_a)


def left_transforms(corner, z, rack):
    """Transforms for every moving LEFT part at wheel travel z, rack travel `rack`."""
    h = corner.hp
    p = corner.pose(z, rack)
    P = p["P"]
    q = p["q"]
    T = {}
    R_up = rotvec_matrix(q[:3])
    T["upright"] = rigid(R_up, corner.c0, corner.c0 + q[3:])  # also hub, wheel, tyre, disc, caliper
    T["uca"] = axis_rotation(h["UCA_O"], h["UCA_F"], h["UCA_R"], P["UCA_O"])
    T["lca"] = axis_rotation(h["LCA_O"], h["LCA_F"], h["LCA_R"], P["LCA_O"])  # carries PR_O when pushrod_on == "lca"
    T["rocker"] = rigid(rotvec_matrix(corner.rk_axis * p["th"]), h["RKR_P"], h["RKR_P"])
    T["arb_lever"] = rigid(rotvec_matrix(corner.arb_axis * p["be"]), h["ARB_P"], h["ARB_P"])
    # two-force links: map the static segment onto the posed segment
    for name, a, b in (("tie_rod", "TR_I", "TR_O"), ("pushrod", "PR_I", "PR_O"), ("drop_link", "ARB_R", "ARB_L")):
        T[name] = rigid(align(h[b] - h[a], P[b] - P[a]), h[a], P[a])
    # damper: body hangs from the chassis eye, shaft from the rocker eye, both on the eye-to-eye line
    R_d = align(h["DMP_R"] - h["DMP_C"], P["DMP_R"] - P["DMP_C"])
    T["damper_body"] = rigid(R_d, h["DMP_C"], P["DMP_C"])
    T["damper_shaft"] = rigid(R_d, h["DMP_R"], P["DMP_R"])
    T["rack_bar"] = rigid(np.eye(3), np.zeros(3), np.array([0.0, rack, 0.0]))
    return T, p


def pose_transforms(corner, z_left, z_right, rack):
    TL, pl = left_transforms(corner, z_left, rack)
    TR, pr = left_transforms(corner, z_right, -rack)
    out = {f"L_{k}": v for k, v in TL.items() if k != "rack_bar"}
    out.update({f"R_{k}": M4 @ v @ M4 for k, v in TR.items() if k != "rack_bar"})
    out["rack_bar"] = TL["rack_bar"]
    # ARB bar: rigid in heave, twisted in roll; show it at the mean lever angle
    be = 0.5 * (pl["be"] + pr["be"])
    out["arb_bar"] = rigid(rotvec_matrix(corner.arb_axis * be), corner.hp["ARB_P"], corner.hp["ARB_P"])
    meta = {"z_left": z_left, "z_right": z_right, "rack": rack,
            "steer_left_deg": pl["steer"], "steer_right_deg": -pr["steer"],
            "camber_left_deg": pl["camber"], "camber_right_deg": pr["camber"],
            "damper_left_mm": pl["damper_len"], "damper_right_mm": pr["damper_len"],
            "arb_twist_deg": math.degrees(pl["be"] - pr["be"])}
    return out, meta


HEAVES = tuple(np.arange(-30.0, 30.0 + 1e-9, 7.5))   # every 7.5 mm, so a minimum between grid points is
ROLLS = (-2.0, -1.0, 0.0, 1.0, 2.0)                   # at most a few tenths of a mm off (worst pairs sit at corners)
RACK_STEPS = 9
WARPS = ((30.0, -30.0), (30.0, 0.0), (0.0, -30.0))   # one wheel up, the other down (mirrored by the -rack poses)


def sweep_grid(veh, rack_stop):
    """Heave x roll x steer over the operating envelope, plus warp (opposite wheels in opposite
    directions, which only the ARB parts feel) and the pure extremes."""
    t2 = veh["track_front_mm"] / 2
    racks = np.linspace(-rack_stop, rack_stop, RACK_STEPS)
    poses = []
    for heave in HEAVES:
        for roll in ROLLS:
            dz = t2 * math.tan(math.radians(roll))
            zl, zr = heave + dz, heave - dz
            if max(abs(zl), abs(zr)) > 30.0 + 1e-9:  # stay inside the design travel
                continue
            poses += [(round(zl, 3), round(zr, 3), round(r, 3)) for r in racks]
    for zl, zr in WARPS:
        poses += [(zl, zr, round(r, 3)) for r in racks] + [(zr, zl, round(r, 3)) for r in racks]
    return sorted(set(poses))


@stamps.producer("poses")
def main():
    hp, meta = load_hardpoints()
    veh = load_vehicle()
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    summary = {}
    with open(DATA / "kinematics_summary.csv") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 1 and parts[0] == "rack_travel_at_stop":
                summary["rack_stop"] = float(parts[1])
    grid = sweep_grid(veh, summary["rack_stop"])
    out = {"_hardpoints": read_version(DATA / "hardpoints.csv"),
           "units": "mm, 4x4 row-major world transforms from the static (design) position",
           "rack_stop_mm": summary["rack_stop"], "poses": []}
    for zl, zr, s in grid:
        T, meta = pose_transforms(corner, zl, zr, s)
        out["poses"].append({**meta, "T": {k: np.round(v, 6).tolist() for k, v in T.items()}})
    (DATA / "poses.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(grid)} poses written to data/poses.json (rack stop {summary['rack_stop']:.2f} mm)")


if __name__ == "__main__":
    main()
