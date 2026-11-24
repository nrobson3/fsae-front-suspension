"""Independent 2D hand calculation of LC1 (3 g bump), front view only. It shares no code with the
3D kinematic or force solvers, so agreement is a real cross-check of the 6x6 result.

Model (the textbook front-view simplification):
  * Everything is projected onto the front (y-z) plane at static ride height.
  * The upper wishbone is one two-force member along its front-view line (UCA_O to the point
    where its pivot axis pierces the plane). Same for the lower wishbone (LCA_O to its pivot).
  * The tie rod carries no load in a pure vertical case (its steering moment is out of plane).
  * The pushrod acts on the lower-wishbone node along its front-view line.
Steps (do them on paper the same way):
  1. Upright free body: Fz at the contact patch, UCA force along its line, lower ball-joint
     force (2 components). Moments about the lower ball joint give the UCA force.
  2. Force balance gives the lower ball-joint force.
  3. Lower-wishbone node: ball-joint reaction, LCA leg force along its line, pushrod force along
     its line. Two equations, two unknowns.
Run: python handcalc_lc1.py   (prints the steps and compares the pushrod force with forces.py)
"""
import csv
import math

from common import DATA, load_hardpoints, read_version


def yz(p):
    return (p[1], p[2])


def solve2(a11, a12, a21, a22, b1, b2):
    det = a11 * a22 - a12 * a21
    return ((b1 * a22 - a12 * b2) / det, (a11 * b2 - b1 * a21) / det)


def solve():
    """The three steps and the 3D comparison, as numbers (used by main() and by handcalcs.py)."""
    hp, _ = load_hardpoints()
    with open(DATA / "loadcases.csv", newline="") as f:
        lc1 = next(r for r in csv.DictReader(l for l in f if not l.startswith("#")) if r["id"] == "LC1")
    Fz = float(lc1["Fz_N"])
    cp = (610.0, 0.0)                       # contact patch (front view), static
    uo, lo, pro, pri = yz(hp["UCA_O"]), yz(hp["LCA_O"]), yz(hp["PR_O"]), yz(hp["PR_I"])
    ui, li = yz(hp["UCA_F"]), yz(hp["LCA_F"])   # pivot axes are parallel to x: any point gives the trace

    def u(a, b):  # unit vector a -> b
        d = (b[0] - a[0], b[1] - a[1])
        n = math.hypot(*d)
        return (d[0] / n, d[1] / n)

    eU = u(uo, ui)                          # tension pulls the upright towards the chassis
    # 1. moments about the lower ball joint (counter-clockwise +, viewed from the front: y right, z up)
    def mom(p, F):
        return (p[0] - lo[0]) * F[1] - (p[1] - lo[1]) * F[0]
    m_F = mom(cp, (0.0, Fz))
    m_U = mom(uo, eU)                       # moment of a unit tension in the UCA
    f_uca = -m_F / m_U
    # 2. force balance on the upright: F + f_uca eU + B = 0 (B = force from the lower ball joint on the upright)
    B = (-(0.0 + f_uca * eU[0]), -(Fz + f_uca * eU[1]))
    # 3. lower-wishbone node: -B (reaction) + f_lca eL + f_p eP = 0
    eL, eP = u(lo, li), u(pro, pri)
    f_lca, f_p = solve2(eL[0], eP[0], eL[1], eP[1], B[0], B[1])

    # comparison only: the 3D 6x6 solve at the same (static) pose
    import numpy as np
    import forces
    from common import load_vehicle
    from suspension import make_corner
    hp_all, meta = load_hardpoints()
    corner = make_corner(hp_all, meta, load_vehicle()["tyre"]["loaded_radius_mm"])
    members = forces.read_csv_rows(DATA / "members.csv")
    acts = {"Pushrod": "LCA_O"} if corner.pushrod_on == "lca" else {}
    f3d, _, _ = forces.solve_links(corner.pose(0.0), members, np.array([0.0, 0.0, Fz]), acts)
    names = [m["member"] for m in members]
    push3d = f3d[names.index("Pushrod")]
    lca3d = f3d[names.index("LCA front leg")] + f3d[names.index("LCA rear leg")]
    return {"Fz": Fz, "cp": cp, "pts": {"UCA_O": uo, "LCA_O": lo, "PR_O": pro, "PR_I": pri, "UCA pivot trace": ui,
                                   "LCA pivot trace": li},
            "f_uca": f_uca, "B": B, "f_lca": f_lca, "f_p": f_p, "push3d": push3d, "lca3d": lca3d}


def main():
    r = solve()
    Fz, cp, f_uca, B, f_lca, f_p, push3d, lca3d = (r[k] for k in ("Fz", "cp", "f_uca", "B", "f_lca", "f_p", "push3d", "lca3d"))

    print(f"Hand calc, LC1 3 g bump, front view (hardpoints {read_version(DATA / 'hardpoints.csv')})")
    print(f"  Fz = {Fz:.0f} N at the contact patch (y = {cp[0]:.0f}, z = 0)")
    print(f"  1. UCA force (moments about the lower ball joint): {f_uca:8.0f} N (tension +)")
    print(f"  2. Lower ball-joint force on the upright: By = {B[0]:.0f} N, Bz = {B[1]:.0f} N, |B| = {math.hypot(*B):.0f} N")
    print(f"  3. Node balance: LCA (front-view resultant) {f_lca:8.0f} N, pushrod {f_p:8.0f} N")
    print(f"  3D 6x6 solve, same static pose: pushrod {push3d:.0f} N; LCA legs sum {lca3d:.0f} N (legs are angled in plan, "
          f"so their sum is not the front-view resultant)")
    print(f"  Pushrod: hand calc vs 3D differs by {abs(f_p - push3d) / abs(push3d) * 100:.1f} %")
    return f_p, push3d


if __name__ == "__main__":
    main()
