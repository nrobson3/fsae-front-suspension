"""Upright fatigue by Miner's rule over the duty spectrum (S-07c).

The infinite-life screen in upright_fea.py (S-07a) asks for unlimited life at the limit cornering load.
This check uses the cycles the upright actually sees in its design life (data/duty_loads.json):
  stress      the FEA in every duty state (same mesher, loads, support and exclusions as upright_fea.py),
              on the level-1 mesh (its governing peak is within 1 % of the finest mesh's)
  press fit   the cups' Lame hoop and radial stress at the tightest P6 fit (data/bearings.json) added to
              the mean stress wherever a cup sits
  cycle       amplitude = von Mises of half the stress-tensor range, mean = signed von Mises of the mean
              tensor (plus the press fit), Goodman to the design Ftu
  S-N line    Shigley's construction for aluminium (no endurance limit): f x Ftu at 1e3 cycles to the
              Marin-corrected fatigue strength at 5e8 (materials.json fatigue_basis); beyond 5e8 the line
              is continued
  margin      at every point, the factor k on all stresses that brings Miner's damage to 1 over the design
              life; target k >= 1.5. Also the life at k = 1.
Run: python upright_fatigue.py   Writes data/upright_fatigue.json, notes/upright_fatigue.md
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import upright_fea as U  # noqa: E402
import fingerprint  # noqa: E402
from common import DATA, NOTES, read_version, unit  # noqa: E402
from fea import Model  # noqa: E402

FAT_LEVEL = 1
TARGET = 1.5


def sn_line(mat):
    """(S at 1e3 cycles, exponent b, S at 5e8) of the S-N line: S = S1 (N / 1e3)^b."""
    S1 = mat["fatigue_f_1e3"] * mat["Ftu_MPa"]
    ka = 4.51 * mat["Ftu_typical_MPa"] ** -0.265
    Se = mat["fatigue_strength_5e8_MPa"] * ka * 0.9 * mat["fatigue_kd_temperature_assumed"] * 0.814
    b = math.log(Se / S1) / math.log(5e8 / 1e3)
    return S1, b, Se


def damage(k, amps, means, ns, S1, b, Sut):
    """Miner's damage at every point with all stresses scaled by k (arrays: cycles x points)."""
    D = np.zeros(amps.shape[1])
    for a, m, n in zip(amps, means, ns):
        mean = np.maximum(k * m, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sar = np.where(mean < Sut, k * a / (1.0 - mean / Sut), np.inf)
            N = np.where(sar > 0, 1e3 * (np.minimum(sar, 1e12) / S1) ** (1.0 / b), np.inf)
        N = np.where(sar >= S1, np.minimum(N, 1e3), N)          # at or above the 1e3 point: treated as 1e3 cycles
        D += n / N
    return D


def margin(amps, means, ns, S1, b, Sut, lo=0.05, hi=20.0, iters=40):
    """Per point, the stress factor k with damage(k) = 1 (bisection; capped to [lo, hi])."""
    klo = np.full(amps.shape[1], lo)
    khi = np.full(amps.shape[1], hi)
    for _ in range(iters):
        mid = np.sqrt(klo * khi)
        fail = damage(mid, amps, means, ns, S1, b, Sut) > 1.0
        khi = np.where(fail, mid, khi)
        klo = np.where(fail, klo, mid)
    return np.sqrt(klo * khi)


def fit_stress(X, faces, build, veh):
    """Lame hoop + radial stress of the tightest cup press fit, as tensors (3, 3, ...) at points X, zero
    outside the two cup zones of the housing ring."""
    brg = json.loads((DATA / "bearings.json").read_text())
    p = brg["fit"]["interface_pressure_MPa"]
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    a, bo = faces["seat_r"], build["R_h"]
    d = X - wc[:, None, None]
    ax = np.einsum("i,i...->...", s, d)
    radial = d - s[:, None, None] * ax
    r = np.linalg.norm(radial, axis=0)
    rib0, rib1 = faces["rib"]
    cw = veh["hub"]["cup_width_mm"]
    zone = (((ax > rib0 - cw) & (ax < rib0)) | ((ax > rib1) & (ax < rib1 + cw))) & (r >= a - 0.5) & (r <= bo)
    c = p * a ** 2 / (bo ** 2 - a ** 2)
    rr = np.maximum(r, a)
    s_t = np.where(zone, c * (1 + bo ** 2 / rr ** 2), 0.0)
    s_r = np.where(zone, c * (1 - bo ** 2 / rr ** 2), 0.0)
    er = radial / np.maximum(r, 1e-9)
    et = np.stack([s[1] * er[2] - s[2] * er[1], s[2] * er[0] - s[0] * er[2], s[0] * er[1] - s[1] * er[0]])
    return (np.einsum("i...,j...->ij...", et, et) * s_t + np.einsum("i...,j...->ij...", er, er) * s_r).astype(np.float32)


def main():
    faces, build, upl, veh, mat = U.setup()
    duty = json.loads((DATA / "duty_loads.json").read_text())
    reg, _ = U.regions(faces, build)
    sup_pts, sup_comp = U.supports(faces, build)
    sharp = np.array([p for e in build.get("sharp_edges", []) for p in e["points"]]).reshape(-1, 3)
    ver = read_version(DATA / "hardpoints.csv")
    t0 = time.time()
    inputs_sha = fingerprint.fatigue_inputs_hash()
    hot, _ = U.hot_spots(faces, build, upl, veh, mat, reg, sup_pts, sup_comp, sharp)
    m = U.build_model(FAT_LEVEL, faces, build, mat, reg, hot)
    print(f"level {FAT_LEVEL}: {m.K.shape[0]} dofs ({time.time() - t0:.0f} s)", flush=True)
    needed = sorted({c["a"] for c in duty["cycles"]} | {c["b"] for c in duty["cycles"]})
    sig, X = {}, None
    for name in needed:
        sg, X, _, _ = U.run_case(m, duty["states"][name], faces, veh, sup_pts, sup_comp)
        sig[name] = sg.astype(np.float32)
        print(f"{name}: peak {float(Model.von_mises_of(sg).max()):.0f} MPa ({time.time() - t0:.0f} s)", flush=True)
    near_sup, on_hole, near_sharp = U.masks(X, faces, m.support_xyz, sharp)
    body = ~near_sup & ~on_hole & ~near_sharp
    fit = fit_stress(X, faces, build, veh)
    S1, b, Se = sn_line(mat)
    Sut = mat["Ftu_MPa"]
    amps, means, ns = [], [], []
    for c in duty["cycles"]:
        sa, sb = sig[c["a"]], sig[c["b"]]
        amps.append(Model.von_mises_of((sa - sb) / 2).ravel())
        means.append(U.signed_vm((sa + sb) / 2 + fit).ravel())
        ns.append(c["n"])
    amps, means = np.array(amps), np.array(means)
    out = {}
    for label, mask in (("body", body), ("hole edges", on_hole & ~near_sup)):
        idx = np.flatnonzero(mask.ravel())
        k = margin(amps[:, idx], means[:, idx], ns, S1, b, Sut)
        i = int(np.argmin(k))
        j = idx[i]
        D1 = float(damage(1.0, amps[:, [j]], means[:, [j]], ns, S1, b, Sut)[0])
        contrib = [{"cycle": f"{c['a']} <-> {c['b']}", "n": c["n"], "amp_MPa": float(amps[q, j]), "mean_MPa": float(means[q, j]),
                    "damage_at_k1": float(damage(1.0, amps[[q]][:, [j]], means[[q]][:, [j]], [c["n"]], S1, b, Sut)[0])}
                   for q, c in enumerate(duty["cycles"])]
        out[label] = {"k_min": float(k[i]), "at": X.reshape(3, -1)[:, j].round(1).tolist(), "damage_at_k1": D1,
                      "life_km_at_k1": duty["life_km"] / D1 if D1 > 0 else float("inf"), "cycles": contrib}
        print(f"{label}: k_min {k[i]:.2f} at {out[label]['at']}, damage {D1:.2e} at k = 1", flush=True)
    res = {"_hardpoints": ver, "_inputs_sha": inputs_sha, "level": FAT_LEVEL, "dofs": m.K.shape[0], "life_km": duty["life_km"],
           "sn": {"S_1e3_MPa": S1, "b": b, "S_5e8_MPa": Se}, "fit_pressure_MPa": json.loads((DATA / "bearings.json").read_text())["fit"]["interface_pressure_MPa"],
           "state_peaks_MPa": {n: float(Model.von_mises_of(sig[n]).max()) for n in needed},
           "result": out, "fos_min": out["body"]["k_min"], "seconds": time.time() - t0}
    write_report(res, duty, mat)
    (DATA / "upright_fatigue.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def write_report(res, duty, mat):
    sn = res["sn"]
    md = [f"# Upright fatigue: Miner's rule over the duty spectrum (hardpoints {res['_hardpoints']})", "",
          "Generated by `scripts/fea/upright_fatigue.py`. Do not edit by hand.", "",
          f"Design life {res['life_km']:,.0f} km (`data/duty_loads.json`: duty states, cycle counts and their assumptions). Stress from "
          f"the FEA in each duty state on the level-{res['level']} mesh ({res['dofs']:,} DOFs; same loads, support and exclusions "
          f"as `upright_fea.py`), with the cups' press-fit stress ({res['fit_pressure_MPa']:.1f} MPa interface pressure, tightest P6 "
          "fit) added to the mean where a cup sits.", "",
          f"S-N line (Shigley's construction for aluminium): {sn['S_1e3_MPa']:.0f} MPa at 1e3 cycles to {sn['S_5e8_MPa']:.0f} MPa at "
          f"5e8 (exponent {sn['b']:.3f}); amplitude = von Mises of half the tensor range, mean = signed von Mises of the mean, "
          f"Goodman to {mat['Ftu_MPa']:g} MPa. {mat['fatigue_basis']}", "",
          "**Margin** = the factor on every stress that brings Miner's damage to 1 over the design life; target >= "
          f"{TARGET}.", "",
          "| Where | Margin | At [mm] | Damage at k = 1 | Life at k = 1 [km] |", "|---|---|---|---|---|"]
    for label, r in res["result"].items():
        md.append(f"| {label} | **{r['k_min']:.2f}** | {r['at']} | {r['damage_at_k1']:.2e} | {r['life_km_at_k1']:,.0f} |")
    md += ["", "Hole edges are pin-contact zones (for information; the holes are checked for bearing in `upright_fea.md`). "
           "Re-entrant corners the CAD kernel left sharp are excluded (singular), as in `upright_fea.md`, and checked by hand (S-14).",
           "", "## Cycles at the governing point", "",
           "| Cycle | Count | Amplitude [MPa] | Mean [MPa] | Damage at k = 1 |", "|---|---|---|---|---|"]
    for c in res["result"]["body"]["cycles"]:
        md.append(f"| {c['cycle']} | {c['n']:,.0f} | {c['amp_MPa']:.1f} | {c['mean_MPa']:.1f} | {c['damage_at_k1']:.2e} |")
    md += ["", "## Peak von Mises in each duty state [MPa]", "", "| State | Peak |", "|---|---|"]
    md += [f"| {n} | {v:.0f} |" for n, v in res["state_peaks_MPa"].items()]
    (NOTES / "upright_fatigue.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
