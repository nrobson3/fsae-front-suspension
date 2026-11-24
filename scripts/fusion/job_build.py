"""Fusion job: build the suspension assembly from data/hardpoints.csv (run via fusion_client.py).

ARGS: {"project": <project folder>, "tag": <doc tag>, "stages": [...], "fresh": bool}
Stages: "context" (skeleton sketch, ghost chassis, cockpit envelope, rack + column).
More stages are added as the build grows. Returns per-component bounding boxes and body
reports for verification against the hardpoints.
"""
import csv
import importlib
import json
import os
import sys

import adsk.core
import adsk.fusion

sys.path.insert(0, os.path.dirname(__file__))
import fsae_cad as C  # noqa: E402

importlib.reload(C)

GREY = ["Paint - Enamel Glossy (Grey)", "Plastic - Matte (Grey)", "Steel - Satin"]
DRIVER = ["Plastic - Translucent Matte (Yellow)", "Plastic - Matte (Yellow)", "Paint - Enamel Glossy (Yellow)"]
ALU = ["Aluminum - Satin", "Aluminum - Anodized Glossy (Grey)", "Aluminum"]
STEEL = ["Steel - Satin", "Steel"]


CFG = {}   # vehicle.json and members.csv, read once in main() for the part builders


def read_hp(path):
    pts, meta = {}, {}
    with open(path, newline="") as f:
        rows = [l for l in f if not l.lstrip().startswith("#")]
    for r in csv.DictReader(rows):
        pts[r["id"]] = (float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]))
        meta[r["id"]] = r
    return pts, meta


def mirror(p):
    return (p[0], -p[1], p[2])


def stage_context(des, root, hp, ck, report):
    # ---- skeleton: every hardpoint (both corners) and the links between them
    occ = C.new_component(root, "Skeleton (from hardpoints.csv)")
    comp = occ.component
    sk = comp.sketches.add(comp.xYConstructionPlane)
    sk.name = "hardpoints"
    pts = {}
    for side, f in (("L", lambda p: p), ("R", mirror)):
        for k, p in hp.items():
            sp = sk.sketchPoints.add(sk.modelToSketchSpace(C.P(f(p))))
            pts[(side, k)] = sp
    links = [("UCA_O", "UCA_F"), ("UCA_O", "UCA_R"), ("LCA_O", "LCA_F"), ("LCA_O", "LCA_R"), ("TR_O", "TR_I"),
             ("PR_O", "PR_I"), ("DMP_R", "DMP_C"), ("ARB_R", "ARB_L"), ("UCA_O", "LCA_O"), ("WC", "WA"),
             ("RKR_P", "PR_I"), ("RKR_P", "DMP_R"), ("RKR_P", "ARB_R"), ("ARB_P", "ARB_L")]
    for side in ("L", "R"):
        for a, b in links:
            sk.sketchCurves.sketchLines.addByTwoPoints(pts[(side, a)], pts[(side, b)])
    report["skeleton_points"] = sk.sketchPoints.count - 1  # minus the sketch origin
    report["skeleton_lines"] = sk.sketchCurves.sketchLines.count

    # ---- ghost chassis (context, not designed)
    nodes = ck["chassis_nodes_left"]
    r = ck["chassis_tube_od_mm"] / 2
    bodies = []
    for f in (lambda p: tuple(p), lambda p: mirror(p)):
        for a, b in ck["chassis_members_left"]:
            bodies.append(C.tube(f(nodes[a]), f(nodes[b]), r, r - 1.65))
    for (a,) in ck["chassis_cross_members"]:
        bodies.append(C.tube(tuple(nodes[a]), mirror(nodes[a]), r, r - 1.65))
    for p in chassis_tabs_build()[2]:                  # the assumed cross tube under the dampers' eyes (D24)
        bodies.append(C.tube(p["a"], p["b"], p["ro"], p["ri"]))
    occ = C.new_component(root, "Ghost chassis (assumed context)")
    C.commit(occ.component, [C.union(bodies)], ["ghost chassis"])
    C.paint(des, occ.component, GREY)

    # ---- driver lower-leg envelope + pedal box keep-out
    dl = ck["driver_left"]
    bodies = []
    for f in (lambda p: tuple(p), lambda p: mirror(p)):
        for part in ("foot", "shin"):
            s = dl[part]
            a, b = f(s["from"]), f(s["to"])
            bodies.append(C.union([C.cyl(a, b, s["r"]), C.sphere(a, s["r"]), C.sphere(b, s["r"])]))
    pb = dl["pedal_box"]
    ctr = ((pb["x"][0] + pb["x"][1]) / 2, 0.0, (pb["z"][0] + pb["z"][1]) / 2)
    bodies.append(C.box(ctr, (1, 0, 0), (0, 1, 0), pb["x"][1] - pb["x"][0], pb["y"][1] - pb["y"][0], pb["z"][1] - pb["z"][0]))
    occ = C.new_component(root, "Driver legs and pedal box (keep-out)")
    C.commit(occ.component, bodies, ["left foot", "left shin", "right foot", "right shin", "pedal box"])
    C.paint(des, occ.component, DRIVER)

    # ---- steering rack housing (envelope) and column
    tl, tr = hp["TR_I"], mirror(hp["TR_I"])
    house_l, house_r = C.add(tl, (0, -45, 0)), C.add(tr, (0, 45, 0))   # 45 mm of rack + inner joint at each end
    sc = ck["steering_column"]
    bodies = [C.tube(house_l, house_r, 18.0, 13.0),
              C.cyl(tuple(sc["pinion"]), tuple(sc["wheel_centre"]), sc["od_mm"] / 2)]
    occ = C.new_component(root, "Steering rack housing and column (envelope)")
    C.commit(occ.component, bodies, ["rack housing", "steering column"])
    C.paint(des, occ.component, STEEL)


RUBBER = ["Rubber - Black", "Plastic - Matte (Black)", "Paint - Enamel Glossy (Black)"]
DARK = ["Aluminum - Anodized Glossy (Black)", "Paint - Enamel Glossy (Black)", "Plastic - Matte (Black)"]
RED = ["Paint - Enamel Glossy (Red)", "Plastic - Glossy (Red)", "Aluminum - Anodized Glossy (Red)"]
CHROME = ["Chrome", "Steel - Polished", "Steel - Satin"]


def chassis_tabs_build():
    """(static, rack, context, joints) from scripts/chassis_tabs.py: the one definition the sweeps use too."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import chassis_tabs
    importlib.reload(chassis_tabs)
    return chassis_tabs.build(CFG["project"])


def corner_hp(hp, side):
    return hp if side == "L" else {k: mirror(p) for k, p in hp.items()}


def stage_corner(des, root, hp, side, report):
    importlib.reload(sys.modules.get("fsae_parts") or __import__("fsae_parts"))
    import fsae_parts as F
    F.configure(CFG["veh"], CFG["members"], CFG["ud"])
    h = corner_hp(hp, side)
    k = F.kingpin(h)

    def put(name, bodies, names, look):
        occ = C.new_component(root, f"{side} {name}")
        C.commit(occ.component, bodies, names)
        C.paint(des, occ.component, look)
        return occ

    put("Wheel (Braid 10 in, envelope)", [F.wheel(h)], ["wheel"], ALU)
    put("Tyre (Hoosier 16x7.5-10, envelope)", [F.tyre(h)], ["tyre"], RUBBER)
    hb, nut = F.hub(h)
    put("Hub, bearings, nut", [hb, nut] + F.bearings(h), ["hub", "axle nut", "inner bearing", "outer bearing"], STEEL)
    put("Brake disc", [F.disc(h)], ["disc"], CHROME)
    put("Caliper (envelope)", [F.caliper(h)], ["caliper"], RED)
    # the detailed upright is generated as code (scripts/upright/upright_geom.py) and read from its STEP
    step = os.path.join(CFG["project"], "cad", "upright", f"upright_{side}.step")
    bodies = sorted(C.step_bodies(root, step), key=lambda b: -b.volume)
    if len(bodies) != 1:
        raise RuntimeError(f"{step}: expected 1 solid (the one-piece upright, v4), got {len(bodies)}")
    put("Upright", bodies, ["upright"], ALU)
    node, legs, ends = F.wishbone(h, "UCA")
    put("Upper wishbone", [node, legs] + ends, ["node", "legs", "rod end front", "rod end rear"], DARK)
    node, legs, ends = F.wishbone(h, "LCA")
    put("Lower wishbone", [node, legs, F.lca_pushrod_bracket(h)] + ends,
        ["node", "legs", "pushrod bracket", "rod end front", "rod end rear"], DARK)
    tie = C.sub(h["TR_O"], h["TR_I"])
    body, ends = F.link(h["TR_I"], h["TR_O"], F.TUBE_TIE, C.perp(tie, (0, 0, 1)), C.perp(tie, k))
    put("Tie rod", [body] + ends, ["tube", "inner rod end", "outer rod end"], STEEL)
    push = C.sub(h["PR_I"], h["PR_O"])
    piv_l = C.unit(C.sub(h["LCA_R"], h["LCA_F"]))
    rk = C.unit(C.sub(h["RKR_A"], h["RKR_P"]))
    body, ends = F.link(h["PR_O"], h["PR_I"], F.TUBE_PUSH, C.perp(push, piv_l), C.perp(push, rk))
    put("Pushrod", [body] + ends, ["tube", "lower rod end", "upper rod end"], DARK)
    put("Rocker", [F.rocker(h)], ["rocker"], ALU)
    db, ds = F.damper(h)
    put("Damper body (Ohlins TTX25, envelope)", [db], ["damper body"], RED)
    put("Damper shaft", [ds], ["damper shaft"], CHROME)
    put("ARB lever", [F.arb_lever(h)], ["lever"], STEEL)
    drop = C.sub(h["ARB_R"], h["ARB_L"])
    arb_ax = C.unit(C.sub(h["ARB_A"], h["ARB_P"]))              # lever end: bolt along the bar, through the blade (R-08)
    body, ends = F.link(h["ARB_L"], h["ARB_R"], (4.0, 0.0), C.perp(drop, arb_ax), C.perp(drop, rk))
    put("Drop link", [body] + ends, ["rod", "lower rod end", "upper rod end"], STEEL)
    import chassis_tabs as T
    static = chassis_tabs_build()[0]
    prims = static if side == "L" else T.mirrored(static)
    put("Chassis tabs", [F.prim_body(p) for p in prims], [p["name"] for p in prims], STEEL)


def stage_arb_bar(des, root, hp):
    import fsae_parts as F
    F.configure(CFG["veh"], CFG["members"], CFG["ud"])
    bar = CFG["veh"]["arb_bar"]
    occ = C.new_component(root, f"ARB bar ({bar['fitted']}, {bar['bars_mm'][bar['fitted']]:g} mm 4340)")
    C.commit(occ.component, [F.arb_bar(hp)], ["ARB bar"])
    C.paint(des, occ.component, STEEL)


def stage_rack_bar(des, root, hp):
    import fsae_parts as F
    F.configure(CFG["veh"], CFG["members"], CFG["ud"])
    occ = C.new_component(root, "Rack bar")
    import chassis_tabs as T
    rack = chassis_tabs_build()[1]                    # the custom rack-end clevis at each end (D24)
    clevis = [C.union([F.prim_body(p) for p in prims]) for prims in (rack, T.mirrored(rack))]
    C.commit(occ.component, [F.rack_bar(hp)] + clevis, ["rack bar", "rack clevis ears L", "rack clevis ears R"])
    C.paint(des, occ.component, CHROME)


def main(args):
    proj = args["project"]
    hp, meta = read_hp(os.path.join(proj, "data", "hardpoints.csv"))
    with open(os.path.join(proj, "data", "cockpit_envelope.json")) as f:
        ck = json.load(f)
    with open(os.path.join(proj, "data", "vehicle.json")) as f:
        veh = json.load(f)
    with open(os.path.join(proj, "data", "members.csv"), newline="") as f:
        members = list(csv.DictReader(l for l in f if not l.lstrip().startswith("#")))
    with open(os.path.join(proj, "data", "upright_design.json")) as f:
        ud = json.load(f)
    CFG.update(veh=veh, members=members, ud=ud, project=proj)
    tag = args.get("tag", "fsae-front-suspension")
    if args.get("fresh", True):
        doc, des = C.fresh_doc(tag)
    else:
        doc, des = C.use_doc(tag)
    doc.activate()
    root = des.rootComponent
    report = {"doc": doc.name, "hardpoint_version": None, "rim_half_width_mm": veh["wheel"]["rim_width_in"] * 25.4 / 2}
    with open(os.path.join(proj, "data", "hardpoints.csv")) as f:
        for line in f:
            if line.startswith("# version:"):
                report["hardpoint_version"] = line.split(":", 1)[1].strip().split()[0]
    if report["hardpoint_version"]:
        des.attributes.add("fsae", "hardpoints", report["hardpoint_version"])
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import fingerprint
    importlib.reload(fingerprint)            # Fusion keeps modules between runs
    report["inputs_sha"] = fingerprint.cad_inputs_hash(proj)
    des.attributes.add("fsae", "inputs_sha", report["inputs_sha"])
    stages = args.get("stages", ["context"])
    if "context" in stages:
        stage_context(des, root, hp, ck, report)
    for side in ("L", "R"):
        if f"corner_{side}" in stages:
            stage_corner(des, root, hp, side, report)
    if "arb_bar" in stages:
        stage_arb_bar(des, root, hp)
    if "rack_bar" in stages:
        stage_rack_bar(des, root, hp)
    report["components"] = {}
    for i in range(root.occurrences.count):
        comp = root.occurrences.item(i).component
        report["components"][comp.name] = {"bbox_mm": C.bbox_mm(comp), "bodies": C.body_report(comp)}
    return report
