"""Fast analytic clearance between the wheel rim (and tyre bead) and the links that pass through
it - lower/upper wishbone legs, tie rod, pushrod - over heave x steer.

The rim interior is sampled as its barrel inner surface (radius RIM_IN from the axle, between
the inboard lip and the wheel centre disc); each link is a capsule (its centre line + tube
radius). Clearance = point-to-segment distance - tube radius. A sample grid can only overstate
the gap (by up to about 0.2 mm at this grid), so with refine=True (what main() records) each
link's closest sample is re-sampled finely around it until the grid's own error is far below
0.01 mm. Unrefined it runs fast enough to drive the hardpoint optimisers.

    python rim_clearance.py [--hardpoints path] [--rim-half-width mm] [--rack-stop mm]
With no overrides it reads the rim's width from vehicle.json and the steering stop from poses.json, and writes
data/rim_clearance.json (run_all.py runs it so; the write-up quotes it), with a fingerprint of the hardpoint file.
"""
import argparse
import hashlib
import json
import math

import numpy as np

import stamps
from common import DATA, load_hardpoints, load_vehicle, read_version, unit
from suspension import make_corner

RIM_IN = 121.0          # barrel inner radius (10 in bead seat 127 mm - 6 mm wall, assumed)
LIP_EXTRA = 8.0         # inboard flange beyond the half width
FACE = 35.0             # wheel centre disc (mounting face) axial position, ET35
TUBE_R = {"lca_F": 9.53, "lca_R": 9.53, "uca_F": 7.94, "uca_R": 7.94, "tie": 7.94, "push": 9.53}
ROD_END_R = 12.0        # rod-end heads inside the wheel are checked as spheres


N_ANG, N_AX = 144, 36


def hardpoints_sha(path=None):
    """The hardpoint file's content fingerprint (line endings ignored): data/rim_clearance.json records it."""
    return hashlib.sha256((path or DATA / "hardpoints.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _barrel(P, ang, ax):
    """Points on the barrel's inner surface at angles ang [rad] and axial positions ax [mm] from the wheel centre."""
    c = P["WC"]
    s = unit(P["WA"] - c)
    u = unit(np.cross(s, np.array([1.0, 0.0, 0.0])))
    v = np.cross(s, u)
    return c + np.outer(ax, s) + RIM_IN * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v))


def _grid(half_w):
    ang = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    ax = np.linspace(-(half_w + LIP_EXTRA), FACE, N_AX)
    A, X = np.meshgrid(ang, ax)
    return A.ravel(), X.ravel()


def rim_points(P, half_w):
    return _barrel(P, *_grid(half_w))


def closest(P, half_w, dist, refine=False, fine=41, passes=2):
    """The smallest dist(points) over the barrel. Sampled on the grid; with refine, re-sampled around the best point
    (two grid spacings each way, fine x fine, twice), which leaves the grid's own error far below 0.01 mm."""
    A, X = _grid(half_w)
    d = dist(_barrel(P, A, X))
    i = int(np.argmin(d))
    if not refine:
        return float(d[i])
    lo, hi = -(half_w + LIP_EXTRA), FACE
    a0, x0, da, dx, best = A[i], X[i], 2 * np.pi / N_ANG, (hi - lo) / (N_AX - 1), float(d[i])
    for _ in range(passes):
        FA, FX = np.meshgrid(np.linspace(a0 - 2 * da, a0 + 2 * da, fine),
                             np.clip(np.linspace(x0 - 2 * dx, x0 + 2 * dx, fine), lo, hi))
        dd = dist(_barrel(P, FA.ravel(), FX.ravel()))
        j = int(np.argmin(dd))
        a0, x0, best = FA.ravel()[j], FX.ravel()[j], min(best, float(dd[j]))
        da, dx = 4 * da / (fine - 1), 4 * dx / (fine - 1)
    return best


def seg_dist(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + np.outer(t, ab)), axis=1)


def clearances(corner, z, rack, half_w, refine=False):
    p = corner.pose(z, rack)
    P = p["P"]
    links = {"lca_F": (P["LCA_O"], P["LCA_F"]), "lca_R": (P["LCA_O"], P["LCA_R"]),
             "uca_F": (P["UCA_O"], P["UCA_F"]), "uca_R": (P["UCA_O"], P["UCA_R"]),
             "tie": (P["TR_O"], P["TR_I"]), "push": (P["PR_O"], P["PR_I"])}
    out = {k: closest(P, half_w, lambda q, a=a, b=b: seg_dist(q, a, b), refine) - TUBE_R[k] for k, (a, b) in links.items()}
    out["tie_rod_end"] = closest(P, half_w, lambda q: np.linalg.norm(q - P["TR_O"], axis=1), refine) - ROD_END_R
    return out, p["steer"]


def sweep(hp, meta, half_w, racks=None, zs=(-30.0, -15.0, 0.0, 15.0, 30.0), rack_stop=25.8, refine=False):
    veh = load_vehicle()
    corner = make_corner(hp, meta, veh["tyre"]["loaded_radius_mm"])
    racks = racks if racks is not None else np.linspace(-rack_stop, rack_stop, 13)
    worst = {}
    for z in zs:
        for r in racks:
            c, st = clearances(corner, z, float(r), half_w, refine)
            for k, v in c.items():
                if k not in worst or v < worst[k][0]:
                    worst[k] = (v, z, float(r), st)
    return worst


@stamps.producer("rim_clearance")      # only the design's own case writes the file, so only it is stamped
def main(argv=None):
    veh = load_vehicle()
    half_w = veh["wheel"]["rim_width_in"] * 25.4 / 2                  # the CAD's rim (fsae_parts.RIM_HALF_W)
    poses = json.loads((DATA / "poses.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardpoints", default=None)
    ap.add_argument("--rim-half-width", type=float, default=None)
    ap.add_argument("--rack-stop", type=float, default=None)
    a = ap.parse_args(argv)
    default = a.hardpoints is None and a.rim_half_width is None and a.rack_stop is None
    hw = half_w if a.rim_half_width is None else a.rim_half_width
    stop = poses["rack_stop_mm"] if a.rack_stop is None else a.rack_stop
    hp, meta = load_hardpoints(a.hardpoints)
    w = sweep(hp, meta, hw, rack_stop=stop, refine=True)
    for k, (v, z, r, st) in sorted(w.items(), key=lambda kv: kv[1][0]):
        print(f"  {k:12s} min clearance {v:6.2f} mm at z {z:+.0f}, rack {r:+.1f} (steer {st:+.1f} deg)")
    if default:                                    # only the design's own case is recorded
        k, (v, z, r, st) = min(w.items(), key=lambda kv: kv[1][0])
        (DATA / "rim_clearance.json").write_text(json.dumps({
            "_hardpoints": read_version(DATA / "hardpoints.csv"), "_hardpoints_sha": hardpoints_sha(), "min_mm": v,
            "link": k, "z_mm": z, "rack_mm": r, "steer_deg": st, "rim_half_width_mm": hw, "rack_stop_mm": stop,
            "rim_inner_radius_mm_assumed": RIM_IN, "refined": True}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
