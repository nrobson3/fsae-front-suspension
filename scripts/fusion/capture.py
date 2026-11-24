"""Review images of the CAD assembly, at static and at solver poses (run from this folder).

    python capture.py            # writes plots/cad/r*.png

Poses come from the kinematic solver (poses.pose_transforms), converted to per-component
4x4 matrices in cm, so the images show exactly the geometry the analysis checked.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, load_hardpoints, load_vehicle  # noqa: E402
from poses import pose_transforms  # noqa: E402
from suspension import make_corner  # noqa: E402
from fusion_client import run_job  # noqa: E402

GROUP = {"Upright": "upright", "Hub, bearings, nut": "upright", "Brake disc": "upright", "Caliper (envelope)": "upright",
         "Wheel (Braid 10 in, envelope)": "upright", "Tyre (Hoosier 16x7.5-10, envelope)": "upright",
         "Upper wishbone": "uca", "Lower wishbone": "lca", "Tie rod": "tie_rod", "Pushrod": "pushrod", "Rocker": "rocker",
         "Damper body (Ohlins TTX25, envelope)": "damper_body", "Damper shaft": "damper_shaft", "ARB lever": "arb_lever",
         "Drop link": "drop_link"}


def pose_payload(z_left, z_right, rack):
    hp, meta = load_hardpoints()
    corner = make_corner(hp, meta, load_vehicle()["tyre"]["loaded_radius_mm"])
    T, _ = pose_transforms(corner, z_left, z_right, rack)
    out = {}
    for side in ("L", "R"):
        for name, grp in GROUP.items():
            M = np.array(T[f"{side}_{grp}"], float).copy()
            M[:3, 3] *= 0.1  # mm -> cm
            out[f"{side} {name}"] = M.ravel().tolist()
    bar = load_vehicle()["arb_bar"]                   # same component names as job_build.py
    for name, key in ((f"ARB bar ({bar['fitted']}, {bar['bars_mm'][bar['fitted']]:g} mm 4340)", "arb_bar"),
                      ("Rack bar", "rack_bar")):
        M = np.array(T[key], float).copy()
        M[:3, 3] *= 0.1
        out[name] = M.ravel().tolist()
    return out


def main():
    out_dir = str(ROOT / "plots" / "cad")
    jobs = [
        ({"views": [{"name": "r01_assembly_iso", "eye": [1500, 1700, 1050], "target": [-120, 0, 260]},
                    {"name": "r02_assembly_front", "eye": [2600, 0, 330], "target": [0, 0, 230]}],
          "hide": ["Driver"]}, None),
        ({"views": [{"name": "r03_corner_no_wheel", "eye": [750, 1150, 560], "target": [0, 420, 230]},
                    {"name": "r04_inside_wheel", "eye": [420, 300, 260], "target": [0, 575, 185]}],
          "hide": ["Driver", "Tyre", "Wheel"]}, None),
        ({"views": [{"name": "r05_full_lock_bump_top", "eye": [0, 0, 2600], "target": [0, 0, 250], "up": [1, 0, 0]}],
          "hide": ["Driver", "Tyre"]}, (25.0, 25.0, 25.3)),
        ({"views": [{"name": "r06_cockpit_check", "eye": [600, 1300, 900], "target": [-150, 0, 250]}],
          "hide": ["Tyre", "Wheel"]}, None),
        ({"views": [{"name": "r07_pushrod_bracket_and_caliper", "eye": [330, 300, 330], "target": [-10, 570, 170]}],
          "hide": ["Driver", "Tyre", "Wheel", "Ghost"]}, None),
    ]
    files = []
    for args, pose in jobs:
        args.update({"out_dir": out_dir, "width": 1600, "height": 1000})
        if pose:
            args["pose"] = pose_payload(*pose)
        files += run_job(str(HERE / "job_capture.py"), args, timeout=240)
    print("\n".join(files))


if __name__ == "__main__":
    main()
