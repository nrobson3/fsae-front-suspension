"""Two more checks of the upright FEA's method, on problems with closed-form answers (S-06g, S-06h).

  hole   a stress concentration: a thin 7075 plate 200 x 100 x 2 mm with a central 20 mm hole, pulled
         along x (a quarter is modelled, by symmetry). Peak sigma_xx at the hole edge against Kt (net section) = 3.000 - 3.140 x + 3.667 x^2
         - 1.527 x^3, x = d / W (Pilkey, Formulas for Stress, Strain and Structural Matrices, 2005; the
         fit to Howland's solution). t / r = 0.2, so plane stress holds. The peak is the raw
         integration-point stress, as the upright's is: points sit just inside the surface, so on a
         steep gradient this reads a little low, and the benchmark measures by how much.
  ring   the cup seat: the upright's housing ring (68 mm bore, 9 mm wall, 14.5 mm cup width) under a
         uniform bore pressure. Lame: hoop stress at the bore p (b^2 + a^2) / (b^2 - a^2) and radial
         growth a p / E ((b^2 + a^2) / (b^2 - a^2) + nu). With free ends sigma_r + sigma_theta is
         constant through the wall, so the axial strain is uniform and Lame is exact for a short ring.
Both use the upright's own chain: gmsh quadratic tetrahedra, scikit-fem, AMG solve, raw stress at the
integration points; the ring is held 3-2-1 (a self-balanced load) and its reactions must come out near zero.

Run: python benchmark_peaks.py   Writes data/fea_benchmarks.json and notes/fea_benchmarks.md (a few minutes)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fingerprint  # noqa: E402
from common import DATA, NOTES, ROOT  # noqa: E402
import fea  # noqa: E402

E, NU = 71.7e3, 0.33                     # 7075, as the upright
WORK = ROOT / ".tmp" / "fea"
TOL_HOLE_PCT, TOL_RING_PCT = 3.0, 1.0

# plate with a hole
PL, PW, PT, PD, P_SIGMA = 200.0, 100.0, 2.0, 20.0, 50.0          # mm, MPa gross stress (L / W = 2, as Heywood's)
# housing ring (data/vehicle.json hub: 68 mm cup OD, the upright's 9 mm wall, 14.5 mm cup width)
RA, RB, RL, RP = 34.0, 43.0, 14.5, 10.0                          # mm, MPa bore pressure


def kt_hole_net(x):
    return 3.000 - 3.140 * x + 3.667 * x ** 2 - 1.527 * x ** 3


def hole(size=1.75, local=0.35):
    """Quarter model (symmetry about x = 0 through the hole and about y = 0): fea._gmsh_mesh holds every
    element at no less than size / 5, so the global size is what lets the hole edge reach `local`."""
    import gmsh
    t0 = time.time()
    g = fea._gmsh_start()
    box = gmsh.model.occ.addBox(0, 0, -PT / 2, PL / 2, PW / 2, PT)
    cyl = gmsh.model.occ.addCylinder(0, 0, -PT, 0, 0, 2 * PT, PD / 2)
    gmsh.model.occ.cut([(3, box)], [(3, cyl)])
    gmsh.model.occ.synchronize()
    WORK.mkdir(parents=True, exist_ok=True)
    path = fea._gmsh_mesh(g, size, WORK / "bench_hole.msh", refine=[([np.array([0.0, 0.0, 0.0])], PD / 2 + 6.0, local)])
    m = fea.Model(fea.load_mesh(path), E, NU)
    m.fix(lambda x: np.abs(x[0]) < 1e-6, components=(0,))         # symmetry planes
    m.fix(lambda x: np.abs(x[1]) < 1e-6, components=(1,))
    m.fix_points([[PL / 2, 0.0, -PT / 2]], [(2,)])
    m.traction(lambda x: x[0] > PL / 2 - 1e-6, [P_SIGMA * PW / 2 * PT, 0.0, 0.0])
    m.solve("amg")
    sig, X = m.qp_stress()
    near = np.hypot(X[0], X[1]) < PD / 2 + 1.5
    peak = float(sig[0, 0][near].max())
    s_net = P_SIGMA * PW / (PW - PD)
    kt_ref = kt_hole_net(PD / PW)
    kt_fea = peak / s_net
    return {"dofs": int(m.K.shape[0]), "local_mm": local, "sigma_net_MPa": s_net, "peak_MPa": peak,
            "kt_fea": kt_fea, "kt_ref": kt_ref, "err_pct": (kt_fea / kt_ref - 1) * 100,
            "reaction_x_N": float(m.reactions()[0]), "seconds": time.time() - t0}


def ring(size=2.0):
    import gmsh
    from skfem import FacetBasis, LinearForm, asm
    t0 = time.time()
    g = fea._gmsh_start()
    outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, RL, RB)
    inner = gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, RL + 2, RA)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()
    WORK.mkdir(parents=True, exist_ok=True)
    path = fea._gmsh_mesh(g, size, WORK / "bench_ring.msh")
    m = fea.Model(fea.load_mesh(path), E, NU)

    def bore(x):
        return np.abs(np.hypot(x[0], x[1]) - RA) < 0.05

    # uniform pressure on the bore: traction p along the outward radial unit vector (fea.Model has no
    # pressure load, and adding one would change the FEA's fingerprint, so it is assembled here)
    fb = FacetBasis(m.mesh, m.ib.elem, facets=m.facets(bore))

    @LinearForm
    def press(v, w):
        r = np.sqrt(w.x[0] ** 2 + w.x[1] ** 2)
        return RP * (w.x[0] / r * v[0] + w.x[1] / r * v[1])

    m.f += asm(press, fb)
    # 3-2-1 on the end face z = 0: radial growth stays free at all three points
    m.fix_points([[RB, 0.0, 0.0], [-RB, 0.0, 0.0], [0.0, RB, 0.0]], [(1, 2), (1, 2), (0, 2)])
    m.solve("amg")
    react = np.abs(m.support_reactions()).max()
    sig, X = m.qp_stress()
    r_qp = np.hypot(X[0], X[1])
    th = np.arctan2(X[1], X[0])
    s_hoop = sig[0, 0] * np.sin(th) ** 2 + sig[1, 1] * np.cos(th) ** 2 - 2 * sig[0, 1] * np.sin(th) * np.cos(th)
    k = (RB ** 2 + RA ** 2) / (RB ** 2 - RA ** 2)
    hoop_ref = RP * k
    band = r_qp < RA + 0.4
    hoop_fea = float(s_hoop[band].mean())
    hoop_ref_band = float(np.mean(RP * RA ** 2 / (RB ** 2 - RA ** 2) * (1 + RB ** 2 / r_qp[band] ** 2)))
    xyz = m.node_xyz()
    on_bore = np.abs(np.hypot(xyz[0], xyz[1]) - RA) < 0.01
    ux, uy = m.u[m.comp[0]][on_bore], m.u[m.comp[1]][on_bore]
    ur = (ux * xyz[0][on_bore] + uy * xyz[1][on_bore]) / RA
    u_ref = RA * RP / E * (k + NU)
    return {"dofs": int(m.K.shape[0]), "hoop_bore_ref_MPa": hoop_ref, "hoop_band_ref_MPa": hoop_ref_band,
            "hoop_band_fea_MPa": hoop_fea, "hoop_err_pct": (hoop_fea / hoop_ref_band - 1) * 100,
            "u_bore_ref_um": u_ref * 1000, "u_bore_fea_um": float(ur.mean()) * 1000,
            "u_spread_um": float(ur.max() - ur.min()) * 1000,
            "u_err_pct": (float(ur.mean()) / u_ref - 1) * 100,
            "max_support_reaction_N": float(react), "load_N": RP * 2 * np.pi * RA * RL, "seconds": time.time() - t0}


def main():
    h, r = hole(), ring()
    load = P_SIGMA * PW / 2 * PT                                     # on the quarter model
    ok_h = abs(h["err_pct"]) <= TOL_HOLE_PCT and abs(h["reaction_x_N"] + load) < 1e-3 * load
    ok_r = (abs(r["hoop_err_pct"]) <= TOL_RING_PCT and abs(r["u_err_pct"]) <= TOL_RING_PCT
            and r["max_support_reaction_N"] < 1e-3 * r["load_N"])
    res = {"_inputs_sha": fingerprint.bench_inputs_hash(), "hole": h, "ring": r,
           "hole_kt_err_abs_pct": abs(h["err_pct"]),
           "ring_err_abs_pct": max(abs(r["hoop_err_pct"]), abs(r["u_err_pct"])), "ok": bool(ok_h and ok_r)}
    (DATA / "fea_benchmarks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["# FEA benchmarks: a stress concentration and the cup seat (S-06g, S-06h)", "",
          "Generated by `scripts/fea/benchmark_peaks.py`. Do not edit by hand. The upright's own chain: gmsh "
          "quadratic tetrahedra, scikit-fem, AMG solve, raw integration-point stress.", "",
          f"**Plate with a hole (S-06g):** 7075 plate {PL:.0f} x {PW:.0f} x {PT:g} mm (a quarter modelled), central {PD:.0f} mm hole, "
          f"{P_SIGMA:.0f} MPa gross tension, {h['local_mm']:g} mm elements at the hole, {h['dofs']:,} DOFs. Peak "
          f"sigma_xx {h['peak_MPa']:.1f} MPa on a net-section stress of {h['sigma_net_MPa']:.1f} MPa: Kt "
          f"{h['kt_fea']:.3f} against {h['kt_ref']:.3f} (Pilkey's fit to Howland, d/W {PD / PW:g}): "
          f"**{h['err_pct']:+.2f} %** (limit +/-{TOL_HOLE_PCT:g} %). The integration points sit just inside the "
          "surface, so on this gradient (about 7 sigma / r at the edge) the raw peak reads slightly low; the "
          "upright's peaks are read the same way.", "",
          f"**Housing ring under bore pressure (S-06h):** {2 * RA:.0f} mm bore, {RB - RA:g} mm wall, {RL:g} mm long, "
          f"{RP:g} MPa, {r['dofs']:,} DOFs, held 3-2-1 (largest support reaction {r['max_support_reaction_N']:.2e} N "
          f"on a {r['load_N'] / 1000:.1f} kN load).", "",
          "| Quantity | Lame | FEA | Error |", "|---|---|---|---|",
          f"| Hoop stress, mean over the 0.4 mm next to the bore [MPa] | {r['hoop_band_ref_MPa']:.2f} | "
          f"{r['hoop_band_fea_MPa']:.2f} | {r['hoop_err_pct']:+.2f} % |",
          f"| Radial growth of the bore [µm] | {r['u_bore_ref_um']:.3f} | {r['u_bore_fea_um']:.3f} | "
          f"{r['u_err_pct']:+.2f} % |", "",
          f"Hoop stress at the bore itself (Lame) {r['hoop_bore_ref_MPa']:.2f} MPa; the growth varies by "
          f"{r['u_spread_um']:.3f} µm round and along the bore. Limit +/-{TOL_RING_PCT:g} % (Lame is exact here).", "",
          f"**Result: {'PASS' if res['ok'] else 'FAIL'}.**"]
    (NOTES / "fea_benchmarks.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return res["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
