"""Trade-off study: camber recovery in roll vs roll-centre migration in heave.

For each camber-recovery level, find the wishbone geometry with the least ground-referenced RC
migration over +/-25 mm heave, with the static RC held in 20-50 mm. Two frontiers:

  open   - inside the chassis pickup bounds only (the Week 1 study: 6 variables, as
           tune_hardpoints.stage_arms);
  rim    - the bounds the current design actually lives in: the wishbone legs must also clear the
           rim barrel by 6 mm (analytic) over heave x full steering lock, LCA pickups at y >= 195,
           LCA ball joint at z <= 132, LCA leg spread no narrower than now (7 variables, as
           tune_hardpoints.stage_arms_rim). This is the frontier that tests deviation DV-1.

Run: python tradeoff_rc_camber.py [--rim | --plot-only]   (minutes; one process per level)
Writes data/tradeoff_rc_camber[_rim].csv (stamped with the hardpoint version) and
plots/k08_tradeoff_rc_vs_camber.png (both frontiers + the current design point).
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import differential_evolution

from common import DATA, PLOTS, INK_2, SERIES, MUTED, figure, load_hardpoints, load_vehicle, read_version, style_axes, unit
from suspension import SolveError
from tune_hardpoints import BIG, arm_hp, arm_metrics

LEVELS = [40.0, 50.0, 60.0, 70.0, 80.0]
LO = np.array([180.0, 60.0, 190.0, 150.0, 108.0, 270.0])
HI = np.array([240.0, 130.0, 270.0, 300.0, 125.0, 292.0])

RIM_LEVELS = [60.0, 65.0, 70.0, 75.0]
RIM_LO = np.array([195.0, 60.0, 190.0, 150.0, 108.0, 268.0, 171.0])
RIM_HI = np.array([240.0, 130.0, 290.0, 300.0, 132.0, 292.0, 220.0])
MIN_CLEAR = 6.0     # analytic rim clearance, as the v1.5 optimiser (the CAD mesh sweep then measured 5.2 mm)


def _penalties(m, level):
    c = 10 * max(0.0, abs(m["recovery"] - level) - 2.5) ** 2
    c += 10 * max(0.0, abs(m["rc0"] - 35.0) - 15.0) ** 2
    return c


def solve_level(level):
    hp0, _ = load_hardpoints()

    def cost(x):
        try:
            m = arm_metrics(arm_hp(hp0, x))
        except (SolveError, np.linalg.LinAlgError, ValueError):
            return BIG
        return m["rc_mig"] / 15.0 + _penalties(m, level)

    r = differential_evolution(cost, list(zip(LO, HI)), seed=2, tol=1e-7, maxiter=40, popsize=10, polish=True)
    m = arm_metrics(arm_hp(hp0, r.x))
    return level, r.x, m, None


def rim_build(hp0, x, pr_offset):
    yLi, zLi, yUi, zUi, zLo, zUo, sL = x
    hp = arm_hp(hp0, [yLi, zLi, yUi, zUi, zLo, zUo])
    hp["LCA_F"][0], hp["LCA_R"][0] = sL, -sL
    hp["PR_O"] = hp["LCA_O"] + pr_offset * unit(hp["PR_I"] - hp["LCA_O"])
    return hp


def solve_level_rim(level):
    import rim_clearance as RCL
    hp0, _ = load_hardpoints()
    half_w = load_vehicle()["wheel"]["rim_width_in"] * 25.4 / 2
    pr_offset = float(np.linalg.norm(hp0["PR_O"] - hp0["LCA_O"]))
    rack_stop = _kin("rack_travel_at_stop")
    racks = (-rack_stop, -0.8 * rack_stop, 0.8 * rack_stop, rack_stop)
    meta = {"PR_O": {"body": "lca"}}

    def cost(x):
        try:
            hp = rim_build(hp0, x, pr_offset)
            m = arm_metrics(hp)
            w = RCL.sweep(hp, meta, half_w, racks=racks, zs=(-30.0, 30.0))
        except (SolveError, np.linalg.LinAlgError, ValueError):
            return BIG
        c = m["rc_mig"] / 15.0 + _penalties(m, level)
        c += 10 * max(0.0, abs(m["rc_roll"] - m["rc0"]) - 12.0) ** 2      # keep RC migration in roll inside 15 mm
        for k in ("lca_F", "lca_R", "uca_F", "uca_R"):
            c += 10.0 * max(0.0, MIN_CLEAR - w[k][0]) ** 2
        return c

    r = differential_evolution(cost, list(zip(RIM_LO, RIM_HI)), seed=5, tol=1e-7, maxiter=60, popsize=12, polish=True)
    hp = rim_build(hp0, r.x, pr_offset)
    m = arm_metrics(hp)
    w = RCL.sweep(hp, meta, half_w, racks=np.linspace(-rack_stop, rack_stop, 9), zs=(-30.0, -15.0, 0.0, 15.0, 30.0))
    return level, r.x, m, min(w[k][0] for k in ("lca_F", "lca_R", "uca_F", "uca_R"))


def _kin(metric):
    with open(DATA / "kinematics_summary.csv", newline="") as f:
        rows = {r[0]: r for r in csv.reader(f)}
    return float(rows[metric][1])


def main(rim):
    levels, fn, name = (RIM_LEVELS, solve_level_rim, "tradeoff_rc_camber_rim.csv") if rim else \
        (LEVELS, solve_level, "tradeoff_rc_camber.csv")
    with Pool(len(levels)) as pool:
        rows = pool.map(fn, levels)
    ver = read_version(DATA / "hardpoints.csv")
    cols = ["LCA_in_y", "LCA_in_z", "UCA_in_y", "UCA_in_z", "LCA_O_z", "UCA_O_z"] + (["LCA_spread_x"] if rim else [])
    with open(DATA / name, "w", newline="") as f:
        f.write(f"# hardpoints {ver}; {'rim-clearance-constrained frontier' if rim else 'chassis bounds only'}\n")
        w = csv.writer(f)
        w.writerow(["recovery_target_pct", "recovery_pct", "rc_static_mm", "rc_migration_pm25_mm", "fvsa_mm",
                    "rim_clearance_min_mm"] + cols)
        for level, x, m, clear in rows:
            w.writerow([level, f"{m['recovery']:.2f}", f"{m['rc0']:.2f}", f"{m['rc_mig']:.2f}", f"{m['fvsa']:.1f}",
                        "" if clear is None else f"{clear:.2f}"] + [f"{v:.2f}" for v in x])
            print(f"target {level:4.0f} %: recovery {m['recovery']:5.1f} %, RC {m['rc0']:5.1f} mm, "
                  f"migration {m['rc_mig']:5.1f} mm, FVSA {m['fvsa']:6.0f} mm"
                  + ("" if clear is None else f", rim clearance {clear:4.1f} mm") + f", x={np.round(x, 1)}")
    plot()


def _read(name):
    path = DATA / name
    if not path.exists():
        return None, []
    with open(path, newline="") as f:
        lines = list(f)
    stamp = lines[0].split()[2].rstrip(";") if lines and lines[0].startswith("# hardpoints") else "unstamped"
    return stamp, list(csv.DictReader(l for l in lines if not l.startswith("#")))


def plot():
    """k08 from the saved CSVs (--plot-only skips the solves): (a) best migration vs camber recovery,
    with and without the rim constraint; (b) best migration vs the upper pickups' width
    (probe_rc_vs_chassis_width.py), which is what actually limits the design."""
    fig, axs = figure(1, 2, w=5.2, h=3.8)
    ax = axs[0][0]
    top = 0.0
    for i, (name, label) in enumerate([("tradeoff_rc_camber.csv", "chassis bounds only"),
                                       ("tradeoff_rc_camber_rim.csv", "plus rim clearance at full lock")]):
        stamp, rows = _read(name)
        if not rows:
            continue
        rec = [float(r["recovery_pct"]) for r in rows]
        mig = [float(r["rc_migration_pm25_mm"]) for r in rows]
        top = max(top, max(mig))
        ax.plot(rec, mig, color=SERIES[i], marker="o", markersize=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=f"best possible: {label} ({stamp})")
    ver = read_version(DATA / "hardpoints.csv")
    rec0, mig0 = _kin("camber_recovery"), _kin("rc_migration_heave")
    top = max(top, mig0)
    ax.plot([rec0], [mig0], linestyle="none", marker="D", markersize=6, color=SERIES[7], label=f"design {ver}")
    ax.axhspan(0, 15, color="#0ca30c", alpha=0.08, linewidth=0, zorder=0)
    ax.axvspan(60, 80, color="#0ca30c", alpha=0.08, linewidth=0, zorder=0)
    ax.text(36.5, 15, "migration target", fontsize=7, color=INK_2, va="bottom")
    ax.text(60.5, top * 1.08, "recovery target", fontsize=7, color=INK_2, va="bottom")
    style_axes(ax, "(a) Best RC migration vs camber recovery\n(upper pickups free out to y = 270-290 mm)",
               "camber recovery in roll [%]", "min RC migration vs ground, +/-25 mm heave [mm]")
    ax.set_ylim(0, top * 1.2)
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="lower right")

    ax = axs[0][1]
    stamp, rows = _read("tradeoff_rc_chassis_width.csv")
    if rows:
        cap = [float(r["uca_pickup_y_max_mm"]) for r in rows]
        mig = [float(r["rc_migration_ground_pm25_mm"]) for r in rows]
        ax.plot(cap, mig, color=SERIES[1], marker="o", markersize=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=f"best possible with rim clearance ({stamp})")
        hp, _ = load_hardpoints()
        ax.plot([hp["UCA_F"][1]], [mig0], linestyle="none", marker="D", markersize=6, color=SERIES[7], label=f"design {ver}")
        ax.axvline(180.0, color=MUTED, linewidth=1.0, linestyle=":")
        ax.text(181.5, top * 1.12, "chassis rail", fontsize=7, color=INK_2, va="top")
        ax.axhspan(0, 15, color="#0ca30c", alpha=0.08, linewidth=0, zorder=0)
        ax.set_xlim(170, max(cap) + 10)
        ax.set_ylim(0, top * 1.2)
        ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="lower left")
    style_axes(ax, "(b) What limits it: upper pickup width\n(recovery anywhere in 60-80 %)",
               "upper wishbone chassis pickups, max y from centreline [mm]", "min RC migration vs ground [mm]")
    fig.tight_layout()
    fig.savefig(PLOTS / "k08_tradeoff_rc_vs_camber.png", dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    plot() if "--plot-only" in sys.argv else main("--rim" in sys.argv)
