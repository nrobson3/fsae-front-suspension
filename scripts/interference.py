"""Interference and clearance sweep of the CAD assembly over data/poses.json.

The CAD bodies are exported once at the static pose (cad/meshes, from
scripts/fusion/job_export_meshes.py; refused if built from other inputs or another upright STEP, and any mesh
file whose hash is not the one the index recorded: common.load_mesh_index, load_mesh). Each body belongs to a
rigid group that the kinematic solver poses (poses.py), so the CAD moves exactly as the analysis
says. For every pose, every pair of bodies in different groups is checked with FCL (collision +
minimum distance, with the nearest point recorded).

Pair rules are per BODY, not per part, so a designed contact cannot hide a clash elsewhere on
the same part (e.g. the upright may touch the wishbone's ball-joint node, never its legs):
  JOINTED  bodies that meet at a joint with designed small gaps: FAIL only on collision;
  OVERLAP  modelled overlaps by design (shaft inside damper body, rack bar inside its housing): skipped;
  INTRA    bodies in the SAME rigid group that still need clearance (rotating wheel and disc vs the
           caliper and upright; the driver's legs vs the steering column): checked once, own threshold;
  others   FAIL on collision, WARN under CLEARANCE_MM.
A body completely inside another (which mesh-to-mesh FCL reports as clear) is caught by an
inside-point test whenever one bounding box contains the other.
Run: python interference.py   Writes data/interference.csv and notes/interference.md
"""
import csv
import json
from fnmatch import fnmatch

import fcl
import numpy as np
import trimesh

import stamps
from common import DATA, NOTES, cad_inputs_hash, load_mesh, load_mesh_index, read_version
from poses import HEAVES, RACK_STEPS, ROLLS, WARPS

CLEARANCE_MM = 5.0      # moving parts vs links and chassis; basis: the stack-up in notes/requirements.md s6
PREFILTER_MM = 60.0

PART_GROUP = {  # part (component name without the side prefix) -> rigid group
    "Upright": "upright", "Hub, bearings, nut": "upright", "Brake disc": "upright", "Caliper (envelope)": "upright",
    "Wheel (Braid 10 in, envelope)": "upright", "Tyre (Hoosier 16x7.5-10, envelope)": "upright",
    "Upper wishbone": "uca", "Lower wishbone": "lca", "Tie rod": "tie_rod", "Pushrod": "pushrod", "Rocker": "rocker",
    "Damper body (Ohlins TTX25, envelope)": "damper_body", "Damper shaft": "damper_shaft", "ARB lever": "arb_lever",
    "Drop link": "drop_link", "Chassis tabs": "chassis",
}
CENTRAL = {"ARB bar": "arb_bar", "Rack bar": "rack_bar"}   # unsided parts that move; everything else unsided is static

GC = "Ghost chassis (assumed context)/*"
JOINTED = [
    ("Upright/*", "Upper wishbone/node"), ("Upright/*", "Lower wishbone/node"),
    ("Upright/*", "Tie rod/outer rod end"),
    ("Lower wishbone/pushrod bracket", "Pushrod/lower rod end"),
    ("Rocker/rocker", "Pushrod/upper rod end"), ("Rocker/rocker", "Damper shaft/damper shaft"),
    ("Rocker/rocker", "Drop link/upper rod end"), ("ARB lever/lever", "ARB bar/*"),
    ("ARB lever/lever", "Drop link/lower rod end"),              # beside the lever on its spacer (R-08)
    (GC, "Upper wishbone/rod end *"), (GC, "Lower wishbone/rod end *"),
    ("Rack bar/*", "Tie rod/inner rod end"),
    ("Chassis tabs/Upper wishbone*", "Upper wishbone/rod end *"), ("Chassis tabs/Lower wishbone*", "Lower wishbone/rod end *"),
    ("Chassis tabs/Rocker pivot bracket*", "Rocker/rocker"),
    ("Chassis tabs/Damper chassis tab*", "Damper body (Ohlins TTX25, envelope)/*"),
]
OVERLAP = [
    ("Damper body (Ohlins TTX25, envelope)/*", "Damper shaft/*"),
    ("Rack bar/rack bar", "Steering rack housing and column (envelope)/rack housing"),   # not the clevises: checked
    ("Rack bar/rack bar", "Steering rack housing and column (envelope)/steering column"),   # pinion meshes with the rack
    ("Chassis tabs/*", GC),                                              # welded to the chassis tubes
    ("Chassis tabs/Rack trunnion block", "Steering rack housing and column (envelope)/rack housing"),   # clamps it
]
INTRA = [  # (A, B, threshold mm; 0 = fail only on contact)
    ("Wheel (Braid 10 in, envelope)/*", "Caliper (envelope)/*", 3.0),
    ("Wheel (Braid 10 in, envelope)/*", "Upright/*", 3.0),
    ("Wheel (Braid 10 in, envelope)/*", "Brake disc/*", 3.0),
    ("Brake disc/*", "Upright/*", 3.0), ("Hub, bearings, nut/hub", "Caliper (envelope)/*", 3.0),
    ("Brake disc/*", "Caliper (envelope)/*", 0.0),
    ("Driver legs and pedal box (keep-out)/*", "Steering rack housing and column (envelope)/*", CLEARANCE_MM),
]


def _match(rules, a, b):
    for rule in rules:
        pa, pb = rule[0], rule[1]
        if (fnmatch(a, pa) and fnmatch(b, pb)) or (fnmatch(a, pb) and fnmatch(b, pa)):
            return rule
    return None


def part_of(component):
    """(side, part name, rigid-group key)."""
    if component.startswith(("L ", "R ")):
        side, name = component[0], component[2:]
        return side, name, f"{side}_{PART_GROUP[name]}"
    for prefix, grp in CENTRAL.items():
        if component.startswith(prefix):
            return "", prefix, grp
    return "", component, "static"


def load_bodies():
    bodies = []
    for fn, info in load_mesh_index().items():
        side, part, key = part_of(info["component"])
        m = load_mesh(fn, info, process=True)
        bvh = fcl.BVHModel()
        bvh.beginModel(len(m.vertices), len(m.faces))
        bvh.addSubModel(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
        bvh.endModel()
        bodies.append({"name": f"{info['component']} / {info['body']}", "id": f"{part}/{info['body']}", "side": side,
                       "key": key, "geom": bvh, "verts": np.asarray(m.vertices, float), "faces": np.asarray(m.faces)})
    return bodies


def classify(bodies):
    """-> moving pairs [(i, j, jointed)], intra pairs [(i, j, threshold)]"""
    moving, intra = [], []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = bodies[i], bodies[j]
            same_side = (a["side"] == b["side"]) or not a["side"] or not b["side"]
            if a["key"] == b["key"]:
                rule = _match(INTRA, a["id"], b["id"])
                if rule and same_side:
                    intra.append((i, j, rule[2]))
                continue
            if same_side and _match(OVERLAP, a["id"], b["id"]):
                continue
            moving.append((i, j, bool(same_side and _match(JOINTED, a["id"], b["id"]))))
    return moving, intra


def _inside(inner, outer):
    """True if the posed body `inner` lies inside the closed posed body `outer` (bounding boxes nested first)."""
    (iv, _), (ov, ofaces) = inner, outer
    if not (np.all(iv.min(0) >= ov.min(0)) and np.all(iv.max(0) <= ov.max(0))):
        return False
    m = trimesh.Trimesh(vertices=ov, faces=ofaces, process=False)
    return bool(m.is_watertight and m.contains(iv[:1])[0])


def _check(oa, ob, pa, pb):
    """Minimum gap (0 on contact or containment) and the midpoint of the nearest points.
    pa, pb = (posed vertices, faces)."""
    if fcl.collide(oa, ob, fcl.CollisionRequest(), fcl.CollisionResult()):
        return 0.0, None
    if _inside(pa, pb) or _inside(pb, pa):
        return 0.0, None
    dres = fcl.DistanceResult()
    d = fcl.distance(oa, ob, fcl.DistanceRequest(enable_nearest_points=True), dres)
    return d, 0.5 * (np.asarray(dres.nearest_points[0]) + np.asarray(dres.nearest_points[1]))


@stamps.producer("interference")
def main():
    poses = json.loads((DATA / "poses.json").read_text())
    ver = read_version(DATA / "hardpoints.csv")
    if poses.get("_hardpoints") != ver:
        raise RuntimeError(f"poses.json is for hardpoints {poses.get('_hardpoints')}, hardpoints.csv is {ver}: run poses.py")
    bodies = load_bodies()
    moving, intra = classify(bodies)
    worst = {}

    def posed(b, T):
        M = T.get(b["key"], np.eye(4))
        obj = fcl.CollisionObject(b["geom"], fcl.Transform(M[:3, :3], M[:3, 3]))
        return obj, (b["verts"] @ M[:3, :3].T + M[:3, 3], b["faces"])

    for i, j, thr in intra:                     # same rigid group: relative position never changes
        (oa, pa), (ob, pb) = posed(bodies[i], {}), posed(bodies[j], {})
        d, near = _check(oa, ob, pa, pb)
        worst[(i, j)] = (d, (0.0, 0.0, 0.0), "intra", thr, near)
    for pose in poses["poses"]:
        T = {k: np.array(v) for k, v in pose["T"].items()}
        objs, pv = zip(*(posed(b, T) for b in bodies))
        box = [(v.min(0), v.max(0)) for v, _ in pv]
        for i, j, jointed in moving:
            (amin, amax), (bmin, bmax) = box[i], box[j]
            if np.linalg.norm(np.maximum(0, np.maximum(amin - bmax, bmin - amax))) > PREFILTER_MM:
                continue
            d, near = _check(objs[i], objs[j], pv[i], pv[j])
            if (i, j) not in worst or d < worst[(i, j)][0]:
                worst[(i, j)] = (d, (pose["z_left"], pose["z_right"], pose["rack"]),
                                 "jointed" if jointed else "moving", 0.0 if jointed else CLEARANCE_MM, near)
    rows = []
    for (i, j), (d, at, kind, thr, near) in worst.items():
        status = "FAIL" if d <= 0 else ("WARN" if d < thr else "OK")
        rows.append({"a": bodies[i]["name"], "b": bodies[j]["name"], "min_mm": d, "kind": kind, "limit_mm": thr,
                     "z_left": at[0], "z_right": at[1], "rack": at[2], "status": status,
                     "near_x": "" if near is None else f"{near[0]:.1f}", "near_y": "" if near is None else f"{near[1]:.1f}",
                     "near_z": "" if near is None else f"{near[2]:.1f}"})
    if not rows:
        raise RuntimeError("no body pairs came within the pre-filter distance: check the meshes and poses")
    rows.sort(key=lambda r: (r["status"] != "FAIL", r["status"] != "WARN", r["min_mm"]))
    n_pose = len(poses["poses"])
    with open(DATA / "interference.csv", "w", newline="") as f:
        f.write(f"# hardpoints {ver}; cad {cad_inputs_hash()}; poses {n_pose}; pairs {len(moving) + len(intra)}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "min_mm": f"{r['min_mm']:.2f}"})
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    n_warn = sum(r["status"] == "WARN" for r in rows)
    md = [f"# Interference sweep (hardpoints {ver})", "",
          "Generated by `scripts/interference.py`. Do not edit by hand.", "",
          f"{n_pose} poses: heave {HEAVES[0]:+.0f}..{HEAVES[-1]:+.0f} mm every {HEAVES[1] - HEAVES[0]:g} mm x roll "
          f"{ROLLS[0]:+g}..{ROLLS[-1]:+g} deg every {ROLLS[1] - ROLLS[0]:g} deg x rack across its travel in {RACK_STEPS} "
          f"steps, plus warp {', '.join(f'({a:+.0f}, {b:+.0f})' for a, b in WARPS)} mm and their mirrors. "
          f"{len(bodies)} bodies; {len(moving)} pairs of "
          f"bodies that move relative to each other (skipped at a pose when their posed bounding boxes are more than "
          f"{PREFILTER_MM:.0f} mm apart) and {len(intra)} pairs inside one rigid group (rotating wheel and disc vs "
          "caliper and upright; driver vs steering column), checked once.", "",
          f"Rules are per body: jointed bodies (designed joints) fail only on contact; all other moving pairs warn under "
          f"{CLEARANCE_MM:.0f} mm; in-group pairs have their own limit. A body fully inside another counts as a collision.", "",
          f"**Result: {n_fail} collisions, {n_warn} clearance warnings.**", "",
          "| Status | Body A | Body B | Min gap [mm] | Limit [mm] | Kind | At z_L, z_R, rack [mm] | Nearest point x, y, z [mm] |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows[:50]:
        near = f"{r['near_x']}, {r['near_y']}, {r['near_z']}" if r["near_x"] else "-"
        md.append(f"| {r['status']} | {r['a']} | {r['b']} | {r['min_mm']:.1f} | {r['limit_mm']:.0f} | {r['kind']} | "
                  f"{r['z_left']:.0f}, {r['z_right']:.0f}, {r['rack']:.1f} | {near} |")
    (NOTES / "interference.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
