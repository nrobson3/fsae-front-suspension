"""Chassis-side mounts for the inboard joints, as primitives (standard library only, so Fusion can import it).

One definition feeds the Fusion build (fsae_parts.chassis_tabs), the fastener sweep (fasteners.py) and the
offline pre-checks, so they cannot drift apart. From data/chassis_tabs.json, the hardpoints, members.csv and the
ghost chassis (cockpit_envelope.json), LEFT side:
  wishbone tabs    a double-shear tab pair at each inboard rod end, normal to its bolt axis (as fsae_parts.wishbone
                   sets it), running to the nearest chassis tube
  rocker bracket   two plates outside the rocker's pivot boss, to the top rail
  damper tabs      a pair at the damper's chassis eye, on an assumed cross tube under it
  rack clevis      a custom clevis on the Kaz rack's 1/4-28 extension round the inner tie-rod joint (it moves with the rack)
  rack bracket     a trunnion block on the rack tube and a plate from it to the lower side rail
Primitives: {"kind": "plate", p1, p2, r, t, normal} (fsae_cad.plate), {"kind": "tube", a, b, ro, ri}. Joints: {name, key, size, ends: [(face, outward unit, kind, role)], grip_mm}.
"""
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROD_END_W, ROD_END_HEAD_R = 12.0, 12.0          # fsae_parts' AM-M8T envelope
BOLT_HOLE_R = 4.05                               # M8 clearance in the tab plates
ROCKER_HALF = 15.5                               # fsae_parts.rocker: plates at RKR_P +/-13, 5 mm thick, boss to +/-15.5


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit(a):
    n = math.sqrt(dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def perp(a, hint=(0.0, 0.0, 1.0)):
    """As fsae_cad.perp: a unit vector normal to a, as close as possible to hint."""
    a = unit(a)
    h = sub(hint, mul(a, dot(hint, a)))
    if math.sqrt(dot(h, h)) < 1e-6:
        h = sub((1.0, 0.0, 0.0), mul(a, a[0]))
    return unit(h)


def mirror(p):
    return (p[0], -p[1], p[2])


def load(root=ROOT):
    def rd(rel):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return json.load(f)
    hp = {}
    with open(os.path.join(root, "data", "hardpoints.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            hp[r["id"]] = (float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]))
    with open(os.path.join(root, "data", "members.csv"), encoding="utf-8") as f:
        members = {r["member"]: r for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    return hp, members, rd("data/cockpit_envelope.json"), rd("data/upright_design.json"), rd("data/chassis_tabs.json")


def tubes(ck, cfg):
    """Ghost-chassis tube axes [(a, b)] on the left, with the cross members and the assumed cross tube."""
    n = ck["chassis_nodes_left"]
    out = [(tuple(n[a]), tuple(n[b])) for a, b in ck["chassis_members_left"]]
    out += [(tuple(n[a]), mirror(n[a])) for (a,) in ck["chassis_cross_members"]]
    return out + [cross_tube(ck, cfg), rocker_rail(cfg)]


def cross_tube(ck, cfg):
    """The assumed cross tube under the dampers' chassis eyes, between the top rails (y of top_front)."""
    c = cfg["context_cross_tube"]
    y = ck["chassis_nodes_left"]["top_front"][1]
    return ((c["x_mm"], y, c["z_mm"]), (c["x_mm"], -y, c["z_mm"]))


def rocker_rail(cfg):
    """The assumed longitudinal tube below the rocker pivot (left side)."""
    c = cfg["context_rocker_rail"]
    return ((c["x_mm"][0], c["y_mm"], c["z_mm"]), (c["x_mm"][1], c["y_mm"], c["z_mm"]))


def nearest(p, segs):
    """(distance, nearest point, segment) to the nearest segment's axis."""
    best = None
    for a, b in segs:
        ab = sub(b, a)
        t = max(0.0, min(1.0, dot(sub(p, a), ab) / dot(ab, ab)))
        q = add(a, mul(ab, t))
        d = math.sqrt(dot(sub(p, q), sub(p, q)))
        if best is None or d < best[0]:
            best = (d, q, (a, b))
    return best


def pair(name, centre, b, half_gap, t, r, target, tube_r=0.0):
    """Two plates normal to b at +/-(half_gap + t/2) from centre, each from the bolt to target projected into its
    plane, stopping tube_r short of it (on the surface of a tube whose axis passes through target, where the weld
    is); and the faces the bolt's stack sits on (outer faces, outward normals)."""
    plates, faces = [], []
    for sg in (1.0, -1.0):
        off = mul(b, sg * (half_gap + t / 2))
        c0 = add(centre, off)
        tp = add(target, mul(b, dot(sub(c0, target), b)))       # the target in this plate's plane
        if tube_r:
            tp = add(tp, mul(unit(sub(c0, tp)), tube_r))
        plates.append({"kind": "plate", "name": f"{name} {'A' if sg > 0 else 'B'}", "p1": c0, "p2": tp, "r": r, "t": t,
                       "normal": b, "hole_r": BOLT_HOLE_R})
        faces.append((add(centre, mul(b, sg * (half_gap + t))), mul(b, sg)))
    return plates, faces


def tube_r(ck, cfg, seg):
    """Outside radius of the chassis tube along seg: an assumed tube's own, or the ghost chassis's."""
    if seg == cross_tube(ck, cfg):
        return cfg["context_cross_tube"]["od_mm"] / 2
    if seg == rocker_rail(cfg):
        return cfg["context_rocker_rail"]["od_mm"] / 2
    return ck["chassis_tube_od_mm"] / 2


def rail_side_held(faces, to_rail):
    """A bolt's ends: the nut on the face towards the rail, held by a ring spanner (short), and the head on the
    other face, turned with a socket along the axis away from the rail. The wishbone bolts tilt towards their
    rail (normal to the leg, for the rod end's misalignment), so a socket on the rail side hit it (sweep)."""
    (fa, na), (fb, nb) = faces
    if dot(na, to_rail) > dot(nb, to_rail):
        (fa, na), (fb, nb) = (fb, nb), (fa, na)
    return [(fa, na, "head", "turn"), (fb, nb, "nut", "hold")]


def build(root=ROOT):
    """-> (static primitives, rack-bar primitives, context tubes, joints) for the LEFT side."""
    hp, members, ck, ud, cfg = load(root)
    segs = tubes(ck, cfg)
    spacer = ud["clevis"]["spacer_mm"]
    static, rack, joints = [], [], []
    # ---- wishbone inboard rod ends
    w = cfg["wishbone_tabs"]
    for pre, leg, mname in (("UCA", "F", "UCA front leg"), ("UCA", "R", "UCA rear leg"),
                            ("LCA", "F", "LCA front leg"), ("LCA", "R", "LCA rear leg")):
        pin, o = hp[f"{pre}_{leg}"], hp[f"{pre}_O"]
        piv = unit(sub(hp[f"{pre}_R"], hp[f"{pre}_F"]))
        b = perp(sub(o, pin), piv)
        t = float(members[mname]["tab_t_mm"])
        _, q, seg = nearest(pin, segs)
        name = f"{'Upper' if pre == 'UCA' else 'Lower'} wishbone {'front' if leg == 'F' else 'rear'} tab"
        plates, faces = pair(name, pin, b, ROD_END_W / 2 + spacer, t, w["r_mm"][pre], q, tube_r(ck, cfg, seg))
        static += plates
        joints.append({"name": f"{'Upper' if pre == 'UCA' else 'Lower'} wishbone {'front' if leg == 'F' else 'rear'} "
                       "inboard rod end (M8)", "group": "chassis", "size": "M8", "bearing": True,
                       "ends": rail_side_held(faces, unit(sub(q, pin))), "grip_mm": ROD_END_W + 2 * spacer + 2 * t})
    # ---- rocker pivot bracket
    rb = cfg["rocker_bracket"]
    ax = unit(sub(hp["RKR_A"], hp["RKR_P"]))
    piv = hp["RKR_P"]                                  # the rocker is centred on RKR_P (fsae_parts.rocker)
    _, q, seg = nearest(piv, [rocker_rail(cfg)])
    plates, faces = pair("Rocker pivot bracket", piv, ax, ROCKER_HALF + rb["spacer_mm"], rb["t_mm"], rb["r_mm"], q,
                         tube_r(ck, cfg, seg))
    static += plates
    joints.append({"name": "Rocker pivot (M8)", "group": "chassis", "size": "M8",
                   "ends": [(faces[0][0], faces[0][1], "head", "hold"), (faces[1][0], faces[1][1], "nut", "turn")],
                   "grip_mm": 2 * (ROCKER_HALF + rb["spacer_mm"] + rb["t_mm"])})
    # ---- damper chassis eye on the assumed cross tube
    dt = cfg["damper_tabs"]
    _, q, seg = nearest(hp["DMP_C"], [cross_tube(ck, cfg)])
    q = (q[0], q[1] - dt["lean_mm"], q[2])                                 # leaning away from the damper body
    plates, faces = pair("Damper chassis tab", hp["DMP_C"], ax, dt["eye_width_mm"] / 2 + dt["spacer_mm"], dt["t_mm"], dt["r_mm"], q,
                         tube_r(ck, cfg, seg))
    for pl in plates:            # narrow next to the eye (the damper body is 9 mm away), wide from step_mm down to the tube
        e = unit(sub(pl["p2"], pl["p1"]))
        pm = add(pl["p1"], mul(e, dt["step_mm"]))
        static.append({**pl, "name": pl["name"].replace("tab", "tab root"), "p1": pm, "r": dt["r_root_mm"], "hole_r": 0.0})
        pl["p2"] = pm
    static += plates
    joints.append({"name": "Damper chassis eye (M8)", "group": "chassis", "size": "M8", "bearing": True,
                   "ends": [(faces[0][0], faces[0][1], "head", "hold"), (faces[1][0], faces[1][1], "nut", "turn")],
                   "grip_mm": dt["eye_width_mm"] + 2 * dt["spacer_mm"] + 2 * dt["t_mm"]})
    # ---- rack-end clevis round the inner tie-rod joint (moves with the rack bar)
    rc = cfg["rack_clevis"]
    tie = sub(hp["TR_O"], hp["TR_I"])
    bt = perp(tie, (0.0, 0.0, 1.0))                                        # as job_build sets the inner rod end's bolt
    toward = sub(hp["TR_I"], (0.0, rc["reach_mm"], 0.0))                   # back along the rack, inboard
    plates, faces = pair("Rack clevis ear", hp["TR_I"], bt, rc["slot_mm"] / 2, rc["ear_t_mm"], rc["r_mm"], toward)
    rack += plates
    joints.append({"name": "Tie rod inner rod end (M8, rack-end clevis)", "key": "rack_bar", "group": "rack_bar", "size": "M8",
                   "bearing": True,
                   "ends": [(faces[0][0], faces[0][1], "head", "hold"), (faces[1][0], faces[1][1], "nut", "turn")],
                   "grip_mm": rc["slot_mm"] + 2 * rc["ear_t_mm"]})
    # ---- rack bracket: trunnion block on the rack tube, plate to the lower side rail
    rk = cfg["rack_brackets"]
    tr = hp["TR_I"]
    bc = (tr[0], rk["y_mm"], tr[2])
    col = rk["collar"]
    static.append({"kind": "tube", "name": "Rack trunnion block", "a": sub(bc, (0.0, col["length_mm"] / 2, 0.0)),
                   "b": add(bc, (0.0, col["length_mm"] / 2, 0.0)), "ro": col["ro_mm"], "ri": col["ri_mm"]})
    t = rk["plate_t_mm"]
    foot = (bc[0], bc[1], bc[2] - col["ro_mm"])                             # the block's underside
    half = rk["bolt_spacing_mm"] / 2
    pad_c = add(foot, (0.0, 0.0, -t / 2))
    # a horizontal pad under the block takes the bolts; a vertical web runs from it to the lower side rail
    static.append({"kind": "plate", "name": "Rack bracket pad", "p1": sub(pad_c, (half, 0.0, 0.0)),
                   "p2": add(pad_c, (half, 0.0, 0.0)), "r": rk["plate_r_mm"], "t": t, "normal": (0.0, 0.0, 1.0)})
    rail = [s for s in segs if abs(s[0][2] - s[1][2]) < 1e-6 and s[0][2] < 100.0]   # the lower side rails
    _, q, seg = nearest(foot, rail)
    q = add(q, mul(unit(sub(foot, q)), ck["chassis_tube_od_mm"] / 2))   # on the rail's surface
    static.append({"kind": "plate", "name": "Rack bracket web", "p1": add(foot, (0.0, 0.0, -t - rk["plate_r_mm"])), "p2": q,
                   "r": rk["plate_r_mm"], "t": t, "normal": (1.0, 0.0, 0.0)})
    for i in range(rk["bolts"]):
        p = add(foot, (((i - (rk["bolts"] - 1) / 2) * rk["bolt_spacing_mm"]), 0.0, -t))
        joints.append({"name": f"Rack bracket bolt {i + 1} (M8, into the trunnion block)", "group": "chassis", "size": "M8",
                       "ends": [(p, (0.0, 0.0, -1.0), "head", "turn")], "grip_mm": t,
                       "lock": "safety wire", "engage_d": 2.0,     # drilled heads wired in pairs; into the aluminium block
                       "interface": "Kaz publishes no trunnion block drawing: it needs M8 threads at least 2 d deep, "
                                    "clear of its bore (the modelled collar is an envelope)"})
    context = [{"kind": "tube", "name": "Damper cross tube (assumed)", "a": cross_tube(ck, cfg)[0], "b": cross_tube(ck, cfg)[1],
                "ro": cfg["context_cross_tube"]["od_mm"] / 2,
                "ri": cfg["context_cross_tube"]["od_mm"] / 2 - cfg["context_cross_tube"]["wall_mm"]}]
    rr = cfg["context_rocker_rail"]
    for f in (lambda p: p, mirror):                    # one each side
        a, b = rocker_rail(cfg)
        context.append({"kind": "tube", "name": "Rocker mount rail (assumed)", "a": f(a), "b": f(b),
                        "ro": rr["od_mm"] / 2, "ri": rr["od_mm"] / 2 - rr["wall_mm"]})
    return static, rack, context, joints


def mirrored(prims):
    """The right side's primitives."""
    out = []
    for p in prims:
        q = dict(p)
        for k in ("p1", "p2", "center", "a", "b"):
            if k in q:
                q[k] = mirror(q[k])
        for k in ("normal", "ldir", "wdir"):
            if k in q:
                q[k] = mirror(q[k])
        out.append(q)
    return out
