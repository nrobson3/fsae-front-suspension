"""S-18 fastener sweep: every bolt between designed parts can be fitted and tightened.

For each joint the bolt stack is built from the same data the CAD uses (the upright's build record for its
clevis and caliper bolts; the hardpoints and the CAD builders' rules for the pushrod, rocker and damper bolts):
  bodies   washers, hex heads and nuts (cylinders on their across-corners diameter: conservative), the bolt's
           protruding thread; each attached to the rigid group it moves with and checked against every CAD
           body and every other fastener over all the sweep's poses: FAIL on contact, WARN under WARN_MM
           against a part that moves relative to it
  inserts  a thread insert must lie in solid material: points on a ring MIN_WALL_MM outside its thread,
           over its length, must all be inside the upright (classified on its STEP's exact solid)
  tools    a socket (or hex bit) envelope straight along the axis from each end that is turned, and the bolt's
           own insertion path from the head side: with the wheel off, each must be clear at SOME pose (the
           corner is moved by hand during assembly); FAIL if blocked at every pose
The chassis-side joints (wishbone inboard rod ends, the rocker pivot, the damper's chassis eye, the inner tie-rod
joint in its rack-end clevis, the rack bracket bolts) come from scripts/chassis_tabs.py, the definition Fusion
builds them from. The drop link's lever end is bolted through the ARB blade (single shear, 24 mm washer); the ARB
bar's chassis bearings are not detailed.

Run: python fasteners.py   Writes data/fastener_checks.csv and notes/fasteners.md
"""
import csv
import json
import math

import fcl
import numpy as np
import trimesh

import chassis_tabs
import fingerprint
import interference as IF
import stamps
from common import DATA, NOTES, ROOT, cad_inputs_hash, load_hardpoints, load_mesh, load_mesh_index, read_version, unit

WARN_MM = 2.0
MIN_WALL_MM = 2.0
NO_TOOL_CHECK = ("Wheel (Braid 10 in, envelope)/*", "Tyre (Hoosier 16x7.5-10, envelope)/*")   # wheel off at assembly


def perp(a, hint):
    """Unit vector normal to a, nearest to hint (as fsae_cad.perp)."""
    a = unit(np.asarray(a, float))
    h = np.asarray(hint, float) - a * np.dot(hint, a)
    if np.linalg.norm(h) < 1e-6:
        h = np.array([1.0, 0, 0]) - a * a[0]
    return unit(h)


def cyl(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    T = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], (p1 - p0) / L)
    T[:3, 3] = (p0 + p1) / 2
    return trimesh.creation.cylinder(radius=r, height=L, transform=T, sections=24)


def end_bodies(face, out, size, kind, fast, role="turn", thread_mm=None):
    """Bodies stacked outward from `face` along unit `out`: kind 'head' (washer + head) or 'nut' (washer + nut +
    the bolt's thread past it: thread_mm from the bolt schedule, else two pitches). role 'turn': the tool is a straight socket (or hex bit) along the axis; 'hold': a ring
    spanner round the hex. Returns ([(label, mesh)], (tool label, tool mesh), outer end point)."""
    sz = fast["sizes"][size]
    gap = fast["seat_gap_mm"]
    p = face + out * gap
    bodies = []
    if sz["washer_t"]:                              # named by end: a joint has a washer at each, and names key the results
        bodies.append((f"washer ({kind} end)", cyl(p, p + out * sz["washer_t"], sz["washer_od"] / 2)))
        p = p + out * sz["washer_t"]
    if kind == "head":
        r = sz["head_af"] / np.sqrt(3) if sz["head"] == "hex" else sz["head_dia"] / 2
        h = sz["head_h"]
    else:
        r, h = sz["nut_af"] / np.sqrt(3), sz["nut_h"]
    bodies.append((kind, cyl(p, p + out * h, r)))
    base = p
    p = p + out * h
    if kind == "nut":
        past = thread_mm if thread_mm is not None else 2 * sz["pitch"]
        bodies.append(("thread end", cyl(p, p + out * past, sz["d"] / 2)))
        p = p + out * past
    if role == "hold":
        af = sz["head_af"] if kind == "head" else sz["nut_af"]
        tool = ("ring spanner", cyl(base + out * 0.01, base + out * h, fast["ring_spanner_r_per_af"] * af))
    else:
        label = "socket" if (kind == "nut" or sz["head"] == "hex") else "hex key"
        tool = (label, cyl(base + out * 0.01, p + out * fast["tool_length_mm"], sz["socket_od"] / 2))
    return bodies, tool, p


def joints(side):
    """[{name, group, size, ends: [(face point, outward unit, kind)], grip_mm, insert?}] at the static pose."""
    hp, _ = load_hardpoints()
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    ud = json.loads((DATA / "upright_design.json").read_text())
    members = {r["member"]: r for r in csv.DictReader(l for l in open(DATA / "members.csv") if not l.startswith("#"))}
    f = build["faces"]
    H = {h["name"]: h for h in f["holes"]}
    s = unit(np.array(f["spin"]))
    out = []

    def clevis(name, j, size, head_on, turn, insert=None):
        c1, c2 = np.array(H[f"{j}1"]["centre"]), np.array(H[f"{j}2"]["centre"])
        mid = (c1 + c2) / 2
        faces = []
        for c, hn in ((c1, f"{j}1"), (c2, f"{j}2")):
            o = unit(c - mid)
            t_out = H[hn]["half_len"] + (H[hn].get("boss_mm", 0.0))
            faces.append((c + o * t_out, o))
        ends = []
        for i, (fp, o) in enumerate(faces):
            kind = "insert" if insert and i == insert["ear"] else ("head" if i == head_on else "nut")
            ends.append((fp, o, kind, "turn" if kind == turn else "hold"))
        return {"name": name, "group": "upright", "size": size, "ends": ends, "bearing": True,
                "grip_mm": float(np.linalg.norm(faces[0][0] - faces[1][0])), "insert": insert}

    ins = ud["ball_joints"]["lower"].get("insert")
    if ins and H["LBJ1"].get("insert") != ins:
        raise RuntimeError(f"upright_build.json has insert {H['LBJ1'].get('insert')!r} at LBJ1, the design names {ins!r}: rebuild the upright")
    # head up on the ball joints and the tie rod; turned where a straight socket reaches (the hub is under the
    # upper joint's nut), held with a ring spanner at the other end. The lower joint's insert is in ear 0 (LBJ1,
    # the upper ear), its head on ear 1
    out.append(clevis("Upper ball joint (AN5)", "UBJ", "AN5", head_on=0, turn="head"))
    out.append(clevis(f"Lower ball joint (NAS6606 into a {ins} insert)" if ins else "Lower ball joint (NAS6606)", "LBJ",
                      "NAS6606", head_on=1, turn="head", insert={"ear": 0, "part": ins} if ins else None))
    out.append(clevis("Tie rod outer rod end (M8)", "TR", "M8", head_on=0, turn="nut"))
    lug_half = ud["caliper_mount"]["lug_t_mm"] / 2
    ear_t = json.loads((DATA / "vehicle.json").read_text())["brake"]["caliper"]["ear_thickness_mm"]
    for i, cb in enumerate(build["caliper_bolts"]):
        c = np.array(cb)
        out.append({"name": f"Caliper bolt {i + 1} (M8 socket head)", "group": "upright", "size": "M8_SHCS",
                    "lock": "safety wire", "engage_d": 2.0,     # drilled heads, wired in pairs; into the aluminium caliper
                    "tap_depth_mm": ear_t, "tap_through": True,  # the ear's full thread, no deeper: the disc is 8.7 mm on
                    "interface": f"Wilwood lists a 0.32 in mount hole: taken as tapped through the {ear_t:g} mm ear "
                                 "(a 3/8 tap drill is 0.332 in): confirm the thread on Wilwood's drawing",
                    "ends": [(c - s * lug_half, -s, "head", "turn")], "grip_mm": 2 * lug_half})
    # pushrod's lower rod end in the lower wishbone's bracket (fsae_parts.lca_pushrod_bracket)
    piv = unit(hp["LCA_R"] - hp["LCA_F"])
    b = perp(hp["PR_I"] - hp["PR_O"], piv)
    rod_w, gap = 12.0, ud["clevis"]["spacer_mm"]          # rod-end ball width: fsae_parts.ROD_END_W (Fusion-only module)
    t = float(members["Pushrod"]["tab_t_mm"])
    half = (rod_w + 2 * gap) / 2 + t
    out.append({"name": "Pushrod lower rod end (M8)", "group": "lca", "size": "M8", "bearing": True,
                "ends": [(hp["PR_O"] + b * half, b, "head", "turn"), (hp["PR_O"] - b * half, -b, "nut", "hold")],
                "grip_mm": 2 * half})                       # the tie rod is in the nut's socket path: held there
    # rocker: pushrod, damper and drop-link eyes between two 5 mm plates 26 mm apart (fsae_parts.rocker)
    ax = unit(hp["RKR_A"] - hp["RKR_P"])
    for key, name in (("PR_I", "Pushrod upper rod end (M8)"), ("DMP_R", "Damper rocker eye (M8)"),
                      ("ARB_R", "Drop link rocker end (M8)")):
        out.append({"name": name, "group": "rocker", "size": "M8", "bearing": True,
                    "ends": [(hp[key] + ax * 15.5, ax, "head", "hold"), (hp[key] - ax * 15.5, -ax, "nut", "turn")],
                    "grip_mm": 31.0})
    # the drop link's lever end: through the ARB blade along the bar's axis, single shear, the rod end captured by
    # a 24 mm washer outside its outer spacer (R-08's alternative). fsae_parts.arb_lever puts the blade's plane
    # vehicle.json arb_bar lever_offset_mm inboard of the ball centre: half the ball + a spacer + half the blade
    arb = unit(hp["ARB_A"] - hp["ARB_P"])
    bar = json.loads((DATA / "vehicle.json").read_text())["arb_bar"]
    blade_t, sp = bar["lever_blade_t_mm"], ud["clevis"]["spacer_mm"]
    if abs(float(np.dot(unit(hp["ARB_R"] - hp["ARB_L"]), arb))) > 0.05:
        raise RuntimeError("the drop link is not square to the ARB axis: the CAD's lever-end bolt (perp to the link) and "
                           "this joint's (along the bar) would differ")
    if abs(bar["lever_offset_mm"] - (rod_w / 2 + sp + blade_t / 2)) > 1e-6:
        raise RuntimeError("vehicle.json arb_bar lever_offset_mm is not half the ball + a spacer + half the blade")
    out.append({"name": "Drop link lever end (M8, single shear, 24 mm washer)", "group": "arb_lever", "size": "M8_capture",
                "bearing": True, "capture": "24 mm washer",
                "ends": [(hp["ARB_L"] + arb * (rod_w / 2 + sp), arb, "head", "turn"),
                         (hp["ARB_L"] - arb * (rod_w / 2 + sp + blade_t), -arb, "nut", "hold")],
                "grip_mm": rod_w + 2 * sp + blade_t})
    # chassis side: tabs, brackets and the rack-end clevis (one definition with the CAD build)
    for j in chassis_tabs.build()[3]:
        out.append({**j, "ends": [(np.asarray(fp, float), np.asarray(o, float), k, r) for fp, o, k, r in j["ends"]]})
    if side == "R":
        M = np.array([1.0, -1.0, 1.0])
        for j in out:
            j["ends"] = [(fp * M, o * M, k, r) for fp, o, k, r in j["ends"]]
    return out


def bvh(mesh):
    g = fcl.BVHModel()
    g.beginModel(len(mesh.vertices), len(mesh.faces))
    g.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
    g.endModel()
    return g


def step_solid(step_path):
    """(inside(points) -> bool array, volume mm^3) for the one solid in a STEP file, classified by OpenCASCADE
    through gmsh on the exact B-rep: a tessellated mesh can't be trusted for this (Fusion's left upright mesh
    came out as two shells sharing an edge, so ray-parity tests on it are unreliable). Leaves gmsh initialised."""
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        raise RuntimeError(f"{step_path.name}: expected one solid, found {len(vols)}")
    tag = vols[0][1]

    def inside(pts):
        return np.array([gmsh.model.isInside(3, tag, [float(x) for x in p]) > 0 for p in pts])
    return inside, gmsh.model.occ.getMass(3, tag)


def insert_check(j, fast, inside):
    """Points on a ring MIN_WALL_MM outside the insert's thread, over its length: how many are not in solid
    material. inside(points) -> bool array (step_solid's classifier, or a watertight mesh's contains)."""
    part = fast["inserts"][j["insert"]["part"]]
    fp, o, _, _ = j["ends"][j["insert"]["ear"]]
    a = -o                                                          # installed from the outer face, inward
    r = part["countersink_mm"] / 2 + MIN_WALL_MM
    e1 = perp(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    e2 = np.cross(a, e1)
    pts = [fp + a * z + r * (np.cos(th) * e1 + np.sin(th) * e2)
           for z in np.linspace(0.3, part["length_mm"] - 0.3, 6) for th in np.radians(np.arange(0, 360, 15))]
    ok = np.asarray(inside(np.array(pts)), bool)
    return int((~ok).sum()), len(pts)


@stamps.producer("fasteners")
def main():
    ver = read_version(DATA / "hardpoints.csv")
    fast = json.loads((DATA / "fasteners.json").read_text())
    poses = json.loads((DATA / "poses.json").read_text())
    if poses.get("_hardpoints") != ver:
        raise RuntimeError("poses.json is for other hardpoints: run poses.py")
    if not poses["poses"]:
        raise RuntimeError("poses.json has no poses")
    cad = IF.load_bodies()
    index = load_mesh_index()
    fb, tools, rows = [], [], []                     # fastener bodies, tool envelopes
    inserts = []
    for side in ("L", "R"):
        ups = [fn for fn, i in index.items() if i["component"] == f"{side} Upright" and i["body"] == "upright"]
        if len(ups) != 1:
            raise RuntimeError(f"expected one {side} upright body in the mesh index, found {len(ups)}")
        # inserts are checked on the upright's STEP (exact); it must be the solid the CAD meshes were made from (the
        # mesh index records the STEP's hash; the volumes are compared as well)
        inside, vol = step_solid(ROOT / "cad" / "upright" / f"upright_{side}.step")
        mesh_vol = load_mesh(ups[0], index[ups[0]], process=True).volume
        if abs(vol - mesh_vol) > 0.005 * mesh_vol:
            raise RuntimeError(f"upright_{side}.step ({vol:.0f} mm^3) is not the meshed upright ({mesh_vol:.0f} mm^3): rebuild")
        for j in joints(side):
            key = j.get("key") or f"{side}_{j['group']}"     # the rack-end clevis moves with the (unsided) rack bar
            for fp, o, kind, role in j["ends"]:
                if kind == "insert":
                    bad, n = insert_check(j, fast, inside)
                    inserts.append({"joint": f"{side} {j['name']}", "outside": bad, "points": n})
                    continue
                bodies, (tlabel, tool), tip = end_bodies(np.asarray(fp, float), np.asarray(o, float), j["size"], kind, fast,
                                                         role, schedule(j, fast)["past_mm"] if kind == "nut" else None)
                for label, m in bodies:
                    fb.append({"name": f"{side} {j['name']}: {label}", "joint": f"{side} {j['name']}", "key": key,
                               "geom": bvh(m), "verts": np.asarray(m.vertices), "faces": np.asarray(m.faces)})
                tools.append({"name": f"{side} {j['name']}: {tlabel} on the {kind} ({'turned' if role == 'turn' else 'held'})",
                              "joint": f"{side} {j['name']}", "key": key, "mesh": tool})
                sc = schedule(j, fast)
                if kind == "head" and sc.get("tip_past_mm") and sc["tip_past_mm"] > fast["seat_gap_mm"] + 0.1:
                    # (a tip within the seat gap of the far face, like the caliper bolts' 0.2 mm, is not a body)
                    d_ = fast["sizes"][j["size"]]["d"]   # the bolt's tip beyond a part tapped through (the caliper's ear)
                    far = np.asarray(fp, float) - np.asarray(o, float) * (j["grip_mm"] + j["tap_depth_mm"])
                    m = cyl(far - np.asarray(o, float) * fast["seat_gap_mm"], far - np.asarray(o, float) * sc["tip_past_mm"], d_ / 2)
                    fb.append({"name": f"{side} {j['name']}: thread tip (past the tapped part)", "joint": f"{side} {j['name']}",
                               "key": key, "geom": bvh(m), "verts": np.asarray(m.vertices), "faces": np.asarray(m.faces)})
                if kind == "head":                  # the bolt has to come out (and go in) along its axis from the head side:
                    sz = fast["sizes"][j["size"]]   # its head and washer sweep the bolt's length beyond the seat
                    L = schedule(j, fast)["length_mm"]
                    r = max(sz["washer_od"] / 2 if sz["washer_t"] else 0.0,
                            sz["head_af"] / np.sqrt(3) if sz["head"] == "hex" else sz["head_dia"] / 2)
                    reach = 0.3 + sz["washer_t"] + sz["head_h"] + L
                    tools.append({"name": f"{side} {j['name']}: bolt insertion path ({L:.0f} mm)", "joint": f"{side} {j['name']}",
                                  "key": key, "mesh": cyl(np.asarray(fp) + np.asarray(o) * 0.3,
                                                          np.asarray(fp) + np.asarray(o) * reach, r)})
    import gmsh
    gmsh.finalize()
    names = [b["name"] for b in fb] + [t["name"] for t in tools]
    if len(set(names)) != len(names):               # names key the results: a repeat would hide one body's result
        raise RuntimeError(f"repeated fastener names: {sorted({n for n in names if names.count(n) > 1})}")
    for t in tools:
        t.update(geom=bvh(t["mesh"]), verts=np.asarray(t["mesh"].vertices), faces=np.asarray(t["mesh"].faces))

    def posed(b, T):
        M = T.get(b["key"], np.eye(4))
        return (fcl.CollisionObject(b["geom"], fcl.Transform(M[:3, :3], M[:3, 3])),
                (b["verts"] @ M[:3, :3].T + M[:3, 3], b["faces"]))

    def side_of(name):
        return name[0] if name[:2] in ("L ", "R ") else ""

    # fastener bodies vs CAD bodies and vs other joints' fasteners
    others = cad + fb
    for b in others + tools:
        lo, hi = b["verts"].min(0), b["verts"].max(0)
        b["corners"] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def box(b, T):                                  # conservative AABB of the posed body
        M = T.get(b["key"], np.eye(4))
        c = b["corners"] @ M[:3, :3].T + M[:3, 3]
        return c.min(0), c.max(0)

    def apart(ba, bb, margin):
        (amin, amax), (bmin, bmax) = ba, bb
        return np.linalg.norm(np.maximum(0, np.maximum(amin - bmax, bmin - amax))) > margin

    def relevant(f, b):
        return not (b.get("joint") == f["joint"] or (side_of(b["name"]) and side_of(b["name"]) != f["name"][0]))

    worst = {}
    ident = {}
    for f in fb:                                    # same rigid group: once
        for b in others:
            if b is f or b["key"] != f["key"] or not relevant(f, b) or apart(box(f, ident), box(b, ident), WARN_MM + 0.5):
                continue
            (fo, fpv), (bo, bpv) = posed(f, ident), posed(b, ident)
            d, near = IF._check(fo, bo, fpv, bpv)
            pair = (f["name"], b["name"])
            if pair not in worst or d < worst[pair][0]:
                worst[pair] = (d, None, "same group", near)
    moving = [(f, b) for f in fb for b in others if b is not f and b["key"] != f["key"] and relevant(f, b)]
    for pose in poses["poses"]:
        T = {k: np.array(v) for k, v in pose["T"].items()}
        at = (pose["z_left"], pose["z_right"], pose["rack"])
        for f, b in moving:
            if apart(box(f, T), box(b, T), WARN_MM + 0.5):
                continue
            (fo, fpv), (bo, bpv) = posed(f, T), posed(b, T)
            d, near = IF._check(fo, bo, fpv, bpv)
            pair = (f["name"], b["name"])
            if pair not in worst or d < worst[pair][0]:
                worst[pair] = (d, at, "moving", near)
    # tools: clear at some pose (the static pose first), collision tests only
    order = sorted(poses["poses"], key=lambda p: abs(p["z_left"]) + abs(p["z_right"]) + abs(p["rack"]))
    tool_best = {}
    for t in tools:
        cands = [b for b in others if relevant(t, b) and not any(fnmatchname(b["name"], pat) for pat in NO_TOOL_CHECK)]
        first_block = None
        for pose in order:
            T = {k: np.array(v) for k, v in pose["T"].items()}
            to, tpv = posed(t, T)
            hit = None
            for b in cands:
                if apart(box(t, T), box(b, T), 0.01):
                    continue
                bo, _ = posed(b, T)
                if fcl.collide(to, bo, fcl.CollisionRequest(), fcl.CollisionResult()):
                    hit = b["name"]
                    break
            at = (pose["z_left"], pose["z_right"], pose["rack"])
            if hit is None:                         # no surface crossing; a tool wholly inside a body still reads 0
                gaps = [(IF._check(to, posed(b, T)[0], tpv, posed(b, T)[1])[0], b["name"]) for b in cands
                        if not apart(box(t, T), box(b, T), 10.0)]
                inside = [n for g, n in gaps if g <= 0]
                if not inside:
                    tool_best[t["name"]] = (min(g for g, _ in gaps) if gaps else np.inf, at, None)
                    break
                hit = inside[0]
            first_block = first_block or (at, hit)
        else:
            tool_best[t["name"]] = (0.0, first_block[0], first_block[1])

    for (a, b), (d, at, kind, near) in worst.items():
        status = "FAIL" if d <= 0 else ("WARN" if kind == "moving" and d < WARN_MM else "OK")
        rows.append({"check": "fastener body", "item": a, "against": b, "min_mm": d, "status": status,
                     "pose": f"{at[0]:.0f}, {at[1]:.0f}, {at[2]:.1f}" if at else "all (same rigid group)"})
    for name, (gap, at, blocker) in tool_best.items():
        status = "FAIL" if blocker else "OK"
        rows.append({"check": "tool / insertion path", "item": name, "against": blocker or "-",
                     "min_mm": gap if np.isfinite(gap) else 99.0, "status": status,
                     "pose": (f"blocked at every pose (e.g. {at[0]:.0f}, {at[1]:.0f}, {at[2]:.1f})" if blocker
                              else f"clear at {at[0]:.0f}, {at[1]:.0f}, {at[2]:.1f}")})
    # R-08: every rod end and spherical bearing in double shear, i.e. held by two faces on opposite sides
    for side in ("L", "R"):
        for j in joints(side):
            if not j.get("bearing"):
                continue
            outs = [np.asarray(o, float) for _, o, _, _ in j["ends"]]
            ok = len(outs) >= 2 and min(float(a @ b) for a in outs for b in outs) < -0.9
            cap = j.get("capture")
            if cap and fast["sizes"][j["size"]]["washer_od"] <= fast["rod_end_ball_d_mm"]:
                cap = None                          # a washer no larger than the ball captures nothing
            rows.append({"check": DOUBLE_SHEAR, "item": f"{side} {j['name']}", "against": "-", "min_mm": float(len(outs)),
                         "status": "OK" if ok or cap else "FAIL",
                         "pose": f"single shear, captured by a {cap}" if cap else ("held both sides" if ok else "single shear")})
    # R-09: every joint positively locked, and a bolt length that leaves 2 threads past each nut (the sweep above
    # modelled each bolt's real thread past its nut, from the same schedule)
    for side in ("L", "R"):
        for j in joints(side):
            lk, sc = locking(j, fast), schedule(j, fast)
            rows.append({"check": LOCKING, "item": f"{side} {j['name']}", "against": lk or "none", "min_mm": 0.0,
                         "status": "OK" if lk in fast["locking_allowed"] else "FAIL", "pose": "FSAE T.8.3 (fasteners.json)"})
            tol = fast["length_tol_mm"]
            if sc["past_mm"] is not None:
                ok, pose = sc["past_mm"] >= 2 * fast["sizes"][j["size"]]["pitch"] - tol, "thread past the nut"
            else:
                ok, pose = sc["engage_mm"] >= sc["engage_need_mm"] - tol, "thread engaged"
                if sc.get("tap_depth_mm") is not None and not j.get("tap_through") and sc["engage_mm"] > sc["tap_depth_mm"] + tol:
                    ok, pose = False, "bottoms in a blind hole"
            status = "OK" if ok else "FAIL"
            if ok and j.get("interface"):
                status, pose = "WARN", f"interface: {j['interface']}"
            rows.append({"check": BOLT_LENGTH, "item": f"{side} {j['name']}", "against": sc["text"],
                         "min_mm": sc["past_mm"] if sc["past_mm"] is not None else sc["engage_mm"],
                         "status": status, "pose": pose})
    for r in inserts:
        rows.append({"check": INSERT_CHECK, "item": r["joint"], "against": "upright",
                     "min_mm": float(r["points"] - r["outside"]), "status": "FAIL" if r["outside"] else "OK",
                     "pose": f"{r['outside']} of {r['points']} points outside the material ({MIN_WALL_MM:g} mm wall)"})
    if not rows or not inserts:
        raise RuntimeError("fastener sweep produced no checks (or no insert check)")
    rows.sort(key=lambda r: (r["status"] != "FAIL", r["status"] != "WARN", r["check"], r["min_mm"]))
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    n_warn = sum(r["status"] == "WARN" for r in rows)
    with open(DATA / "fastener_checks.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# hardpoints {ver}; cad {cad_inputs_hash()}; inputs {fingerprint.fastener_inputs_hash()}; "
                 f"poses {len(poses['poses'])}; fails {n_fail}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "min_mm": f"{r['min_mm']:.2f}"})
    seen_joints = sorted({j["name"] for j in joints("L")})
    md = [f"# Fastener sweep, S-18 (hardpoints {ver})", "",
          "Generated by `scripts/fasteners.py`. Do not edit by hand.", "",
          f"Joints (each side): {'; '.join(seen_joints)}. Hex heads and nuts are cylinders on their across-corners "
          "diameter, so a clear result is conservative. Hardware and tool sizes: `data/fasteners.json`.", "",
          f"Checks over the sweep's {len(poses['poses'])} poses: fastener bodies against every CAD body and every other "
          f"joint's fasteners (FAIL on contact; WARN under {WARN_MM:g} mm against a part that moves relative to it); "
          "each tool and each bolt's insertion path must be clear at some pose with the wheel off; thread inserts must "
          f"sit in solid material ({MIN_WALL_MM:g} mm round the insert). The chassis-side joints come from "
          "`scripts/chassis_tabs.py`.", "",
          f"**Result: {n_fail} FAIL, {n_warn} WARN.**", "",
          "| Status | Check | Item | Against | Min gap [mm] | Pose (z_L, z_R, rack) |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "OK" or r["check"] != "fastener body":
            md.append(f"| {r['status']} | {r['check']} | {r['item']} | {r['against']} | {r['min_mm']:.1f} | {r['pose']} |")
    (NOTES / "fasteners.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"fastener sweep: {n_fail} FAIL, {n_warn} WARN over {len(rows)} checks")
    for r in rows:
        if r["status"] != "OK":
            print(f"  {r['status']}: {r['item']} vs {r['against']} ({r['min_mm']:.1f} mm, {r['pose']})")
    return n_fail


INSERT_CHECK = "insert in solid material"
DOUBLE_SHEAR = "double shear"
LOCKING = "positive locking"
BOLT_LENGTH = "bolt length"


def locking(j, fast):
    """How the joint is locked: a nut is an all-metal prevailing-torque nut (BOM J7, J3N), a thread insert the
    screw-locking Heli-Coil (fasteners.json), anything else must say ('lock')."""
    kinds = [k for _, _, k, _ in j["ends"]]
    if "nut" in kinds:
        return "all-metal prevailing-torque nut"
    if "insert" in kinds and fast["inserts"][j["insert"]["part"]].get("locking_version"):
        return "screw-locking insert"
    return j.get("lock")


def schedule(j, fast):
    """The bolt for a joint: under-head length on the size's length steps. Through a nut: the shortest that
    leaves 2 pitches past it (past_mm = the real thread past the nut). Into a thread insert: the bolt reaches the
    insert's outer end (within 1 mm). Into a tapped part: grip + engage_d x d (1.5 in steel, the joint says 2 in
    aluminium), or the whole thread of a bought part whose depth is known (tap_depth_mm), with the tip past it."""
    sz = fast["sizes"][j["size"]]
    kinds = [k for _, _, k, _ in j["ends"]]
    stack = j["grip_mm"] + sz["washer_t"] * sum(k in ("head", "nut") for k in kinds)
    step = sz["length_step_mm"]
    tol = fast["length_tol_mm"]
    if "nut" in kinds:
        need = stack + sz["nut_h"] + 2 * sz["pitch"]
        L = math.ceil((need - tol) / step) * step
        return {"length_mm": L, "past_mm": L - stack - sz["nut_h"], "engage_mm": None, "engage_need_mm": None,
                "text": f"{j['size']} x {L:.1f} mm"}
    if "insert" in kinds:
        need = stack - 1.0
        L = math.ceil((need - tol) / step) * step
        ins = fast["inserts"][j["insert"]["part"]]["length_mm"]
        return {"length_mm": L, "past_mm": None, "engage_mm": ins - max(0.0, stack - L), "engage_need_mm": ins - 1.0,
                "text": f"{j['size']} x {L:.1f} mm"}
    e_need = j.get("engage_d", fast["thread_engagement_d"]) * sz["d"]
    depth = j.get("tap_depth_mm")
    if depth is not None:
        e_need = min(e_need, depth)                 # a bought part's thread: engage all of it
    L = math.ceil((stack + e_need - tol) / step) * step
    return {"length_mm": L, "past_mm": None, "engage_mm": L - stack, "engage_need_mm": e_need, "tap_depth_mm": depth,
            "tip_past_mm": max(0.0, L - stack - depth) if depth is not None else None,
            "text": f"{j['size']} x {L:.1f} mm"}


def sweep_clear(path=DATA / "fastener_checks.csv"):
    """(clear, reason). Clear only if the sweep's file exists, was made from the current inputs
    (fingerprint.fastener_inputs_hash), has rows, no FAIL and an OK insert check on each side: anything
    missing or old reads as not clear, so a drawing hold cannot lift by accident."""
    import re
    if not path.exists():
        return False, "fastener_checks.csv missing: run fasteners.py"
    with open(path, encoding="utf-8") as f:
        head = f.readline()
        rows = list(csv.DictReader(f))
    m = re.search(r"inputs ([0-9a-f]{16})", head)
    if not m or m.group(1) != fingerprint.fastener_inputs_hash():
        return False, "fastener_checks.csv is not from the current inputs: run fasteners.py"
    if not rows:
        return False, "fastener_checks.csv has no checks"
    fails = [r["item"] for r in rows if r["status"] == "FAIL"]
    if fails:
        return False, f"{len(fails)} FAIL: {fails[0]}"
    ins = [r for r in rows if r["check"] == INSERT_CHECK]
    if {r["item"][:2] for r in ins if r["status"] == "OK"} != {"L ", "R "}:
        return False, "no OK insert check on both sides"
    return True, f"{len(rows)} checks, 0 FAIL"


def fnmatchname(name, pat):
    from fnmatch import fnmatch
    comp, _, body = name.partition(" / ")
    part = comp[2:] if comp[:2] in ("L ", "R ") else comp
    return fnmatch(f"{part}/{body}", pat)


if __name__ == "__main__":
    main()
