"""LC4 von Mises contour for the write-up (PLAN s17 step E), and the upright's hero image.

A re-solve of the critical case (LC4), re-meshed at the convergence study's level-1 element sizes (so a slightly
different DOF count), by the same method as upright_fea.py (its functions, not a copy). The surface is coloured, facet by facet, by the largest von Mises at the integration
points of the element under it. The zones the FEA keeps out of its peaks are grey: within 8 mm of a held point,
within 1.5 mm of a pin or bolt hole (pin contact: the holes are checked by bearing stress instead), within 2 mm of
a sharp corner the CAD kernel could not fillet (checked by hand, HC-7). It refuses to draw unless its own body
peak agrees with the stored level-1 convergence result within 1 %, and unless the stored FEA is current.
Nothing here writes upright_fea.json.
Run: python plot_lc4.py   Writes plots/fea02_lc4_von_mises.png, plots/hero_upright.png, data/fea_lc4_plot.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "drawings"))
import fingerprint  # noqa: E402
import upright_fea as U  # noqa: E402
from common import DATA, PLOTS, ROOT  # noqa: E402
from render import render  # noqa: E402

LEVEL = 1
CASE = "LC4"            # the file names and the write-up say LC4: the run stops if another case governs
VIEWS = (("outboard, from the front", (1.0, 1.6, 0.9)), ("inboard, from the rear", (-1.0, -1.6, 0.7)))
ALU = (0.74, 0.76, 0.80)


def _level_peak(fea):
    return next(c for c in fea["convergence"] if c["level"] == LEVEL)["peaks_MPa"]["_global"]


def _solve_sha():
    """The re-solve's own code (the FEA functions it calls are in the FEA's fingerprint): part of the cache key."""
    import hashlib
    import inspect
    return hashlib.sha256(inspect.getsource(_solve).encode()).hexdigest()[:8]


def solve():
    """The solve, cached in .tmp/fea against the FEA's input fingerprint and this file's _solve (a re-plot costs
    seconds, not a re-solve). On a cached solve the 1 % agreement is checked again and the stored FEA values it quotes
    are read afresh."""
    fea = json.loads((DATA / "upright_fea.json").read_text())
    if fea["_inputs_sha"] != fingerprint.fea_inputs_hash():
        raise RuntimeError("upright_fea.json is stale: re-run upright_fea.py first")
    if fea["critical_case"] != CASE:
        raise RuntimeError(f"{fea['critical_case']} now governs, not {CASE}: rename the plot and update the write-up")
    cache = U.WORK / f"lc4_plot_{fea['_inputs_sha']}_L{LEVEL}_{_solve_sha()}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        r = json.loads(str(z["meta"])) | {"V": z["V"], "F": z["F"], "val": z["val"]}
        r |= {"ref": _level_peak(fea), "base_peak": fea["cases"][CASE]["peaks_MPa"]["_global"], "sha": fea["_inputs_sha"]}
        if abs(r["peak"] / r["ref"] - 1) > 0.01:
            raise RuntimeError(f"the cached solve's peak {r['peak']:.1f} MPa differs from the stored level-{LEVEL} result")
        return r
    r = _solve(fea)
    U.WORK.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in r.items() if k not in ("V", "F", "val")}
    np.savez_compressed(cache, V=r["V"], F=r["F"], val=r["val"], meta=json.dumps(meta, default=float))
    return r


def _solve(fea):
    faces, build, upl, veh, mat = U.setup()
    reg, _ = U.regions(faces, build)
    sup_pts, sup_comp = U.supports(faces, build)
    sharp = np.array([p for e in build.get("sharp_edges", []) for p in e["points"]]).reshape(-1, 3)
    crit = fea["critical_case"]
    hot = [np.array(h["at"]) for h in fea["hot_spots"]]
    m = U.build_model(LEVEL, faces, build, mat, reg, hot)
    sig, X, _, _ = U.run_case(m, upl["cases"][crit], faces, veh, sup_pts, sup_comp)
    pk, vm, body = U.peaks(sig, X, reg, faces, m.support_xyz, sharp)
    ref = _level_peak(fea)
    if abs(pk["_global"] / ref - 1) > 0.01:
        raise RuntimeError(f"this solve's peak {pk['_global']:.1f} MPa differs from the stored level-{LEVEL} result {ref:.1f}")
    mesh = m.mesh
    bf = mesh.boundary_facets()
    tri = mesh.facets[:, bf].T
    el = mesh.f2t[0, bf]
    vm_el = np.where(body, vm, -1.0).max(axis=1)              # -1: every integration point of the element excluded
    val = vm_el[el]
    name = next(r["name"] for r in U._read_lcs() if r["id"] == crit).split(" (")[0]
    return {"crit": crit, "case_name": name, "V": mesh.p.T, "F": tri, "val": val,
            "peak": pk["_global"], "peak_at": pk["_global_at"], "base_peak": fea["cases"][crit]["peaks_MPa"]["_global"],
            "Fty": mat["Fty_MPa"], "dofs": int(m.K.shape[0]), "ref": ref, "sha": fea["_inputs_sha"]}


def colour(val, vmax):
    import matplotlib
    cmap = matplotlib.colormaps["turbo"]
    rgb = cmap(np.clip(val / vmax, 0, 1))[:, :3]
    rgb[val < 0] = (0.62, 0.62, 0.62)
    return rgb


def views(V, at):
    c = (V.min(0) + V.max(0)) / 2
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    return [(name, c + np.asarray(d) / np.linalg.norm(d) * 3 * span, c) for name, d in VIEWS]


def crop(img, pad=12, bg=1.0):
    """The image cut to the part plus a margin (the renderer fits the part's bounding box, not its outline)."""
    ink = np.abs(img - bg).max(axis=2) > 0.02
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0] - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1] - 1)
    return img[r0:r1 + 1, c0:c1 + 1]


def plot_contour(r):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    vmax = 10 * np.ceil(r["peak"] / 10)
    rgb = colour(r["val"], vmax)
    fig, axs = plt.subplots(1, 2, figsize=(11, 7.2), dpi=150, gridspec_kw={"wspace": 0.02})
    for ax, (name, eye, tgt) in zip(axs, views(r["V"], r["peak_at"])):
        ax.imshow(crop(render(r["V"], r["F"], rgb, eye, tgt, width=1000, height=1000, ssaa=2, spec=0.08)))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    cb = fig.colorbar(ScalarMappable(Normalize(0, vmax), "turbo"), ax=axs, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label("von Mises [MPa]")
    fig.suptitle(f"Upright, {r['crit']} ({r['case_name']}): von Mises, level-{LEVEL} mesh ({r['dofs']:,} DOF)\n"
                 f"Body peak {r['peak']:.0f} MPa ({r['base_peak']:.0f} on the base mesh) against a 7075-T7351 design yield of "
                 f"{r['Fty']:g} MPa. Grey: pin contact at the holes, held points, sharp corners (each checked separately)",
                 fontsize=9.5)
    out = PLOTS / "fea02_lc4_von_mises.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_hero():
    """The upright from the contour plot's two sides, side by side, in plain anodised grey."""
    import matplotlib.pyplot as plt
    import views as VW
    m = VW.mesh_solid(ROOT / "cad" / "upright" / "upright_L.step", size=1.0, curvature=24)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    fig, axs = plt.subplots(1, 2, figsize=(12, 7.6), dpi=150, gridspec_kw={"wspace": 0.0})
    for ax, (name, eye, tgt) in zip(axs, views(V, None)):
        ax.imshow(crop(render(V, F, ALU, eye, tgt, width=1100, height=1100, ssaa=2), pad=30))
        ax.axis("off")
    out = PLOTS / "hero_upright.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    return out, len(F)


def main():
    t0 = time.time()
    r = solve()                                   # first: a refused run rewrites neither image
    hero, nf = plot_hero()
    print(f"hero: {hero} ({nf} triangles, {time.time() - t0:.0f} s)", flush=True)
    out = plot_contour(r)
    (DATA / "fea_lc4_plot.json").write_text(json.dumps({
        "_fea_inputs_sha": r["sha"], "case": r["crit"], "level": LEVEL, "dofs": r["dofs"], "peak_MPa": r["peak"],
        "peak_at": r["peak_at"], "stored_level_peak_MPa": r["ref"], "base_mesh_peak_MPa": r["base_peak"]}, indent=1))
    print(f"lc4 contour: {out}; peak {r['peak']:.1f} MPa (stored level-{LEVEL} {r['ref']:.1f}); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
