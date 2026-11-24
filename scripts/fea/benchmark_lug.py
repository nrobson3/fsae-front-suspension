"""Validate the upright FEA's method on a part beam theory can check: a slender pin-loaded bar.

The upright has no slender section (its steering arm is 20 mm long and 32 mm wide; every other arm is
shorter), so beam theory cannot be held to 10 % anywhere on it. This benchmark uses exactly the upright's
method instead: the same quadratic tetrahedra from gmsh, the same pin-bearing load (Model.bearing: cosine
pressure in the hole, turned onto the load), and the same raw integration-point stress, on a 7075 bar
160 x 32 x 6 mm (the steering-arm ear's section, made long) clamped at x = 0 and loaded through a
8.1 mm hole 16 mm from its free end by 1000 N across the bar plus 300 N along it.
At mid-span, about 2.3 widths from both ends, sigma_xx = -M y / I + N / A with M = F_y (x_hole - x).
The stress is averaged over a 3 mm band at each edge, as the upright's steering-arm check does.
Run: python benchmark_lug.py   (also run, and recorded, by upright_fea.py)
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT  # noqa: E402
import fea  # noqa: E402

L, W, T, X_HOLE, R_HOLE = 160.0, 32.0, 6.0, 144.0, 4.05
F = np.array([300.0, 1000.0, 0.0])          # N on the bar at the pin: along x, across y
E, NU = 71.7e3, 0.33
X_SEC = 75.0
TOL_PCT = 3.0


def run(size=2.0):
    import gmsh
    t0 = time.time()
    g = fea._gmsh_start()
    box = gmsh.model.occ.addBox(0, -W / 2, -T / 2, L, W, T)
    hole = gmsh.model.occ.addCylinder(X_HOLE, 0, -T, 0, 0, 2 * T, R_HOLE)
    gmsh.model.occ.cut([(3, box)], [(3, hole)])
    gmsh.model.occ.synchronize()
    work = ROOT / ".tmp" / "fea"
    work.mkdir(parents=True, exist_ok=True)
    path = fea._gmsh_mesh(g, size, work / "lug_bar.msh", refine=[([np.array([X_HOLE, 0, 0])], 12.0, size / 2)])
    m = fea.Model(fea.load_mesh(path), E, NU)
    m.fix(lambda x: x[0] < 1e-6)

    def bore(x):
        return (np.abs(np.hypot(x[0] - X_HOLE, x[1]) - R_HOLE) < 0.6) & (np.abs(x[2]) < T / 2 - 0.01)
    m.bearing(bore, [0, 0, 1], [X_HOLE, 0, 0], F)
    m.solve("amg")                                 # as the upright: the direct solver's fill-in takes GBs here
    sig, X = m.qp_stress()
    I, A = T * W ** 3 / 12, T * W
    M = F[1] * (X_HOLE - X_SEC)                    # sagging about z for +F_y at the free end: tension at y < 0
    rows = []
    for side in (1.0, -1.0):
        band = (np.abs(X[0] - X_SEC) < 1.2) & (side * X[1] > W / 2 - 3.5) & (side * X[1] < W / 2 - 0.5)
        y = float(X[1][band].mean())
        fem = float(sig[0, 0][band].mean())
        hand = -M * y / I + F[0] / A
        rows.append({"fibre_mm": y, "fea_MPa": fem, "hand_MPa": hand, "error_pct": (fem / hand - 1) * 100})
    return {"dofs": m.K.shape[0], "rows": rows, "max_err_pct": max(abs(r["error_pct"]) for r in rows),
            "seconds": time.time() - t0, "ok": max(abs(r["error_pct"]) for r in rows) <= TOL_PCT}


def main():
    res = run()
    for r in res["rows"]:
        print(f"fibre {r['fibre_mm']:+.1f} mm: FEA {r['fea_MPa']:.2f} MPa, beam theory {r['hand_MPa']:.2f} MPa, "
              f"error {r['error_pct']:+.2f} %")
    print(f"{'PASS' if res['ok'] else 'FAIL'} (within {TOL_PCT:g} %), {res['dofs']} DOFs, {res['seconds']:.0f} s")
    return res["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
