"""Probe: how wide must the upper chassis pickups be to cut ground-referenced RC migration?

The cause and cost behind deviation DV-1. Same rim-clearance-constrained search as
tradeoff_rc_camber.py --rim (wishbone legs 6 mm inside the rim barrel over heave x full lock,
LCA pickups at y >= 195, LCA ball joint at z <= 132), with camber recovery anywhere in its
60-80 % band, but the UPPER wishbone's chassis pickups capped at a series of widths from the
centreline. The ghost chassis has its mid rail at y = 180 mm at that height, so each cap is a
bracket (or frame) that much further out.

Run: python probe_rc_vs_chassis_width.py   (about 30 minutes, one process per cap)
Writes data/tradeoff_rc_chassis_width.csv (stamped).
"""
import csv
from multiprocessing import Pool

import numpy as np
from scipy.optimize import differential_evolution

from common import DATA, load_hardpoints, load_vehicle, read_version
from suspension import SolveError
from tradeoff_rc_camber import MIN_CLEAR, RIM_HI, RIM_LO, _kin, rim_build
from tune_hardpoints import BIG, arm_metrics

UCA_Y_CAPS = [210.0, 230.0, 250.0, 270.0, 330.0]


def run(cap):
    import rim_clearance as RCL
    hp0, _ = load_hardpoints()
    half_w = load_vehicle()["wheel"]["rim_width_in"] * 25.4 / 2
    pr_offset = float(np.linalg.norm(hp0["PR_O"] - hp0["LCA_O"]))
    rack_stop = _kin("rack_travel_at_stop")
    racks = (-rack_stop, -0.8 * rack_stop, 0.8 * rack_stop, rack_stop)
    meta = {"PR_O": {"body": "lca"}}
    lo, hi = RIM_LO.copy(), RIM_HI.copy()
    hi[2] = cap

    def cost(x):
        try:
            hp = rim_build(hp0, x, pr_offset)
            m = arm_metrics(hp)
            w = RCL.sweep(hp, meta, half_w, racks=racks, zs=(-30.0, 30.0))
        except (SolveError, np.linalg.LinAlgError, ValueError):
            return BIG
        c = m["rc_mig"] / 15.0
        c += 10 * max(0.0, 60.5 - m["recovery"]) ** 2 + 10 * max(0.0, m["recovery"] - 80.0) ** 2
        c += 10 * max(0.0, abs(m["rc0"] - 35.0) - 15.0) ** 2
        c += 10 * max(0.0, abs(m["rc_roll"] - m["rc0"]) - 12.0) ** 2
        for k in ("lca_F", "lca_R", "uca_F", "uca_R"):
            c += 10.0 * max(0.0, MIN_CLEAR - w[k][0]) ** 2
        return c

    r = differential_evolution(cost, list(zip(lo, hi)), seed=7, tol=1e-6, maxiter=35, popsize=10, polish=True)
    hp = rim_build(hp0, r.x, pr_offset)
    m = arm_metrics(hp)
    w = RCL.sweep(hp, meta, half_w, racks=np.linspace(-rack_stop, rack_stop, 9), zs=(-30.0, -15.0, 0.0, 15.0, 30.0))
    return cap, r.x, m, min(w[k][0] for k in ("lca_F", "lca_R", "uca_F", "uca_R"))


if __name__ == "__main__":
    with Pool(len(UCA_Y_CAPS)) as p:
        rows = p.map(run, UCA_Y_CAPS)
    with open(DATA / "tradeoff_rc_chassis_width.csv", "w", newline="") as f:
        f.write(f"# hardpoints {read_version(DATA / 'hardpoints.csv')}; rim-clearance-constrained, recovery 60-80 %\n")
        w = csv.writer(f)
        w.writerow(["uca_pickup_y_max_mm", "rc_migration_ground_pm25_mm", "recovery_pct", "rc_static_mm",
                    "rim_clearance_min_mm", "LCA_in_y", "LCA_in_z", "UCA_in_y", "UCA_in_z", "LCA_O_z", "UCA_O_z",
                    "LCA_spread_x"])
        for cap, x, m, clear in rows:
            w.writerow([cap, f"{m['rc_mig']:.2f}", f"{m['recovery']:.2f}", f"{m['rc0']:.2f}", f"{clear:.2f}"]
                       + [f"{v:.1f}" for v in x])
            print(f"UCA pickup y <= {cap:5.0f}: migration {m['rc_mig']:5.1f} mm, recovery {m['recovery']:5.1f} %, "
                  f"RC {m['rc0']:5.1f} mm, rim clearance {clear:4.1f} mm, x={np.round(x, 1)}")
