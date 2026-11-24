"""Rebuild the analysis outputs from the data files and the committed CAD meshes (35-45 minutes; the fastener
sweep is most of it).

First the upright STEP, a CAD input that run_all does not regenerate: stamps.verify_upright regenerates it in a
scratch folder and stamps the committed files only if they are reproduced exactly (under a minute, and only when an
input or the STEP changed since the last check), so a STEP older than its inputs stops the run here.
Every script below stamps what it wrote (data/stamps.json: the content of each input it read and of each output);
the requirements matrix refuses any output whose stamp no longer matches.

Order matters: kinematics -> vehicle calcs (needs motion ratio, RC, ARB ratio) -> forces (needs the
load cases) -> upright interface loads -> hub bearing life and fits, ball-joint bolt bending, duty-spectrum loads ->
misalignment, hand calc, hand-calc packets -> CAD poses -> interference and fastener sweeps and mass budget (need cad/meshes built from
the current inputs) -> BOM with cost -> requirements matrix (reads everything above) -> DFMEA and test plan
(checks that every requirement not passing has a row that says what closes it) -> WRITEUP.md from its template
(every result read from the outputs; needs the FEA and its LC4 plot current: scripts/fea/plot_lc4.py).
The optimisation and trade studies (tune_hardpoints, tradeoff_rc_camber, probe_rc_vs_chassis_width)
and the upright FEA and fatigue (scripts/fea/upright_fea.py, about an hour; upright_fatigue.py, about 30
minutes) are run on their own; the matrix marks their items STALE if their inputs changed after they ran.

Run from the project root or scripts/:  python scripts/run_all.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bearings  # noqa: E402
import bolts  # noqa: E402
import bom  # noqa: E402
import dfmea  # noqa: E402
import duty_loads  # noqa: E402
import fasteners  # noqa: E402
import forces  # noqa: E402
import handcalc_lc1  # noqa: E402
import handcalcs  # noqa: E402
import interference  # noqa: E402
import kinematics  # noqa: E402
import mass_budget  # noqa: E402
import misalignment  # noqa: E402
import poses  # noqa: E402
import requirements_matrix  # noqa: E402
import rim_clearance  # noqa: E402
import stamps  # noqa: E402
import upright_loads  # noqa: E402
import vehicle_calcs  # noqa: E402
import writeup  # noqa: E402


def main():
    t0 = time.time()
    print("== upright STEP: reproduced from today's inputs (checked in a scratch folder when anything changed) ==")
    why = stamps.verify_upright()
    print("upright STEP: " + (f"reproduced exactly, stamp renewed ({'; '.join(why)})" if why else "stamp current"))
    print("\n== kinematics ==")
    kinematics.evaluate()
    print("\n== vehicle calcs ==")
    vehicle_calcs.main()
    print("\n== forces ==")
    forces.main()
    print("\n== upright interface loads (hub, caliper and upright free bodies) ==")
    upright_loads.main()
    print("\n== hub bearings: life and fits ==")
    bearings.main()
    print("\n== ball-joint and rod-end bolt bending ==")
    bolts.main()
    print("\n== duty-spectrum loads and cycles (for the fatigue check) ==")
    duty_loads.main()
    print("\n== joint misalignment ==")
    misalignment.main()
    print("\n== independent 2D hand calc, LC1 ==")
    handcalc_lc1.main()
    print("\n== hand-calc packets and answer keys (rocker, knee, spindle, ARB ends, steering arm, ear roots) ==")
    handcalcs.main()
    print("\n== CAD poses ==")
    poses.main()
    print("\n== rim clearance, analytic (the tightest rim pair) ==")
    rim_clearance.main([])
    print("\n== CAD interference sweep and mass budget (needs meshes exported from the current CAD) ==")
    interference.main()
    print("\n== fastener sweep: heads, nuts, inserts, tool access (S-18) ==")
    fasteners.main()
    mass_budget.main()
    print("\n== bill of materials with cost ==")
    bom.main()
    print("\n== requirements matrix ==")
    requirements_matrix.main()
    print("\n== DFMEA and test plan (every open item covered) ==")
    dfmea.main()
    print("\n== the write-up (every result from the outputs above) ==")
    writeup.main()
    print(f"\nrun_all: done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
