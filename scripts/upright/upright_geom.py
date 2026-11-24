"""The detailed upright as code: OpenCASCADE solids built with gmsh from the hardpoints and
data/upright_design.json, written as STEP for the Fusion assembly AND meshed directly for the FEA,
so the CAD and the FEA can never disagree.

Coordinates: the vehicle frame at static ride height (the CAD frame), mm. The left upright is built;
the right is its mirror in y = 0.

One solid (v4; v1-v3 had a separately bolted upper clevis over a camber shim pack, which the fastener sweep
showed could be neither fitted nor adjusted: S-18, D20): bearing housing (two 68 mm cup seats either side of a
central rib), upper post with the upper ball-joint ears, lower yoke web with the lower ball-joint ears (the upper
one bossed for a thread insert), steering-arm ears, caliper plate. Camber is set at the upper wishbone's
chassis rod ends. Every clevis ear is a 6 mm plate round its bolt, normal to the bolt.

    python upright_geom.py      writes cad/upright/upright_{L,R}.step (CAD) and upright_L_fea.step (the same solid)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DATA, ROOT, load_hardpoints, load_vehicle, unit  # noqa: E402

OUT = ROOT / "cad" / "upright"


def frame(hp):
    """Non-spinning wheel frame: s spin axis (outboard), u up, r rear, t = s x r."""
    s = unit(hp["WA"] - hp["WC"])
    u = unit(np.array([0.0, 0.0, 1.0]) - s * s[2])
    r = unit(np.cross(s, u))
    if r[0] > 0:
        r = -r
    return s, u, r, unit(np.cross(s, r))


def perp(a, hint):
    a = unit(a)
    h = np.asarray(hint, float) - a * np.dot(hint, a)
    return unit(h)


class Builder:
    def __init__(self):
        import gmsh
        self.g = gmsh
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("upright")
        self.occ = gmsh.model.occ

    def pt(self, p):
        return self.occ.addPoint(*map(float, p))

    def slot_face(self, c0, c1, radius, normal, plane_pt):
        """Slot-shaped plane face from c0 to c1 (radius `radius`) in the plane through plane_pt normal to `normal`."""
        n = unit(normal)
        c0, c1 = (np.asarray(c, float) - n * np.dot(np.asarray(c, float) - plane_pt, n) for c in (c0, c1))
        d = unit(c1 - c0)
        w = unit(np.cross(n, d))
        P = [self.pt(c0 + w * radius), self.pt(c0 - d * radius), self.pt(c0 - w * radius),
             self.pt(c1 - w * radius), self.pt(c1 + d * radius), self.pt(c1 + w * radius)]
        C0, C1 = self.pt(c0), self.pt(c1)
        o = self.occ
        curves = [o.addCircleArc(P[0], C0, P[1]), o.addCircleArc(P[1], C0, P[2]), o.addLine(P[2], P[3]),
                  o.addCircleArc(P[3], C1, P[4]), o.addCircleArc(P[4], C1, P[5]), o.addLine(P[5], P[0])]
        return o.addPlaneSurface([o.addCurveLoop(curves)])

    def stadium(self, c0, c1, radius, thick, normal):
        """Slot-shaped plate from c0 to c1 (radius `radius`), `thick` along `normal`, centred on the plane through c0."""
        return self.plate([(c0, c1, radius)], [], thick, normal)

    def plate(self, slots, polygons, thick, normal):
        """One flat plate, `thick` along `normal`, centred on the plane through the first slot's start: the
        union of slots (c0, c1, radius) and polygons (corner lists), fused as faces BEFORE extruding, so the
        plate's faces carry no internal seams for a later fillet to trip on."""
        n = unit(normal)
        base = np.asarray(slots[0][0], float) - n * thick / 2
        o = self.occ
        faces = [(2, self.slot_face(c0, c1, rad, n, base)) for c0, c1, rad in slots]
        for corners in polygons:
            P = [self.pt(np.asarray(c, float) - n * np.dot(np.asarray(c, float) - base, n)) for c in corners]
            loop = o.addCurveLoop([o.addLine(P[i], P[(i + 1) % len(P)]) for i in range(len(P))])
            faces.append((2, o.addPlaneSurface([loop])))
        if len(faces) > 1:
            faces, _ = o.fuse(faces[:1], faces[1:])
        ext = o.extrude([f for f in faces if f[0] == 2], *(n * thick))
        return [e for e in ext if e[0] == 3]

    def cylinder(self, c, axis, r, half_len):
        a = unit(axis)
        p = np.asarray(c, float) - a * half_len
        return [(3, self.occ.addCylinder(*p, *(a * 2 * half_len), r))]

    def fuse(self, objs):
        out, _ = self.occ.fuse(objs[:1], objs[1:])
        return out

    def cut(self, objs, tools):
        out, _ = self.occ.cut(objs, tools)
        return out


def build(side="L"):
    hp, _ = load_hardpoints()
    veh = load_vehicle()
    ud = json.loads((DATA / "upright_design.json").read_text())
    members = {r.split(",")[0]: r.split(",") for r in (DATA / "members.csv").read_text().splitlines()
               if r and not r.startswith("#") and not r.startswith("member,")}
    s, u, r, t = frame(hp)
    wc = hp["WC"]
    k = unit(hp["UCA_O"] - hp["LCA_O"])
    hub, cal = veh["hub"], veh["brake"]["caliper"]
    hz, cl = ud["housing"], ud["clevis"]
    ear_t, ear_r, spacer = cl["ear_t_mm"], cl["ear_r_mm"], cl["spacer_mm"]
    on = lambda a, du=0.0, dr=0.0, dt=0.0: wc + s * a + u * du + r * dr + t * dt      # noqa: E731
    ax = lambda p: float(np.dot(p - wc, s))                                             # noqa: E731

    B = Builder()
    B.wc_ax = float(np.dot(wc, s))
    occ = B.occ
    # ---- bearing housing
    w = hub["width_mm"]
    a0 = hub["row_centres_axial_mm"][0] - w / 2 - hz["end_land_mm"]
    a1 = hub["row_centres_axial_mm"][1] + w / 2 + hz["end_land_mm"]
    rib0, rib1 = hub["row_centres_axial_mm"][0] + w / 2, hub["row_centres_axial_mm"][1] - w / 2
    R_h = hub["od_mm"] / 2 + hz["wall_mm"]
    housing = [(3, occ.addCylinder(*on(a0), *(s * (a1 - a0)), R_h))]

    # ---- upper post and the upper ball-joint ears, one piece with the upright (v4: camber is set at the upper
    # wishbone's chassis rod ends; the bolted clevis + shims could not be fitted or adjusted, S-18 / D20)
    ubj = hp["UCA_O"]
    up_ubj = float(np.dot(ubj - wc, u))
    x_ubj = float(np.dot(ubj - wc, r))
    ucanode_h = float(members["UCA front leg"][3]) + 6.0
    bj_u = ud["ball_joints"]["upper"]
    node_r_u = bj_u["od_in"] * 25.4 / 2 + 4.5
    post_in = a0                                            # flush with the housing's inboard end, clear of the UBJ node
    assert post_in > ax(ubj) + node_r_u + 3.0, "post inboard face too close to the upper ball-joint node"
    post_t, post_w = ud["post"]["t_mm"], ud["post"]["half_width_mm"]
    post_mid = post_in + post_t / 2
    uc = ud["upper_clevis"]
    post = B.stadium(on(post_mid, R_h - 12, x_ubj), on(post_mid, up_ubj + 22, x_ubj), post_w, post_t, s)
    ears_u, plates_u = [], []
    for sg in (1.0, -1.0):
        c0 = ubj + k * sg * (ucanode_h / 2 + spacer + ear_t / 2)
        dt, geoms = ear(B, c0, k, s, uc["ear_r_mm"], ear_t, ud["clevis"]["inboard_edge_upper_mm"], post_in + post_t)
        ears_u += dt
        plates_u.append(geoms)

    # ---- lower yoke web and the lower ball-joint ears
    lbj = hp["LCA_O"]
    x_lbj = float(np.dot(lbj - wc, r))
    lcanode_h = float(members["LCA front leg"][3]) + 6.0
    bj_l = ud["ball_joints"]["lower"]
    node_r_l = bj_l["od_in"] * 25.4 / 2 + 4.5
    web_in = ax(lbj) + node_r_l + 3.0
    web_out = web_in + ud["lower_web"]["t_mm"]
    # a flat-bottomed web (rounded rectangle) ending below_lower_ear_mm under the lower ear's bottom face: full
    # width wherever an ear joins it (a round end narrow enough to end that close would leave the ear overhanging it)
    hw, rc = ud["lower_web"]["half_width_mm"], ud["lower_web"]["corner_r_mm"]
    up_ear2 = float(np.dot(lbj - k * (lcanode_h / 2 + spacer + ear_t / 2) - wc, u))
    y_bot = up_ear2 - ear_t / 2 - ud["lower_web"]["below_lower_ear_mm"]
    wm = (web_in + web_out) / 2
    web = B.plate([(on(wm, y_bot + rc, x_lbj - (hw - rc)), on(wm, y_bot + rc, x_lbj + (hw - rc)), rc)],
                  [[on(wm, -(R_h - 12), x_lbj - hw), on(wm, -(R_h - 12), x_lbj + hw),
                    on(wm, y_bot + rc, x_lbj + hw), on(wm, y_bot + rc, x_lbj - hw)]], web_out - web_in, s)
    ears_l, plates_l = [], []
    grow_l = ud["clevis"]["lower_upper_ear_grow_mm"]
    ib = ud["clevis"]["lbj_insert_boss"]
    for sg in (1.0, -1.0):
        c0 = lbj + k * sg * (lcanode_h / 2 + spacer + ear_t / 2)
        dt, geoms = ear(B, c0, k, s, ear_r, ear_t, ud["clevis"]["inboard_edge_lower_mm"], web_out,
                        grow=grow_l if sg > 0 else 0.0, clip=(wc, a0, a1))
        if sg > 0:      # the thread insert's boss: round about the bolt (= steering) axis, on top of the upper ear
            dt = B.fuse(dt + B.cylinder(c0 + k * (ear_t / 2 + ib["height_mm"] / 2 - 0.5), k, ib["dia_mm"] / 2,
                                        ib["height_mm"] / 2 + 0.5))
        ears_l += dt
        plates_l.append(geoms)

    # ---- steering-arm ears (normal to the tie-rod bolt), running back into the housing
    tro = hp["TR_O"]
    tie = unit(hp["TR_O"] - hp["TR_I"])
    b_tie = unit(np.cross(np.cross(tie, k), tie))           # bolt normal to the tie rod, nearest the kingpin
    sa = ud["steering_arm"]
    ears_s, plates_s = [], []
    for sg in (1.0, -1.0):
        c0 = tro + b_tie * sg * (sa["rod_end_w_mm"] / 2 + spacer + ear_t / 2)
        radial = (c0 - wc) - s * ax(c0)
        c1 = wc + s * ax(c0) + unit(radial) * (R_h - 14.0)
        ears_s += B.stadium(c0, c1, ear_r, ear_t, b_tie)
        plates_s.append([(c0, c1, ear_r, ear_t, b_tie)])

    # ---- caliper lugs on the PS-1 bolt pattern (axial bolts)
    cm = ud["caliper_mount"]
    disc_ax = hub["disc_axial_mm"]
    r_bolt = veh["brake"]["disc_od_mm_assumed"] / 2 - cal["mount_below_disc_od_mm"]
    mount_face = disc_ax - cal["mount_offset_mm"]
    lug_mid = mount_face - cm["lug_t_mm"] / 2
    if cm.get("flush_inboard"):
        assert abs(mount_face - cm["lug_t_mm"] - a0) < 0.05, "caliper plate no longer flush with the housing end: update lug_t_mm"
    cal_bolts = [on(lug_mid, 0.0, r_bolt, sg * cal["bolt_spacing_mm"] / 2) for sg in (1.0, -1.0)]
    roots = [on(lug_mid, 0.0, R_h - 12, sg * cm["root_half_spacing_mm"]) for sg in (1.0, -1.0)]
    slots = [(root, bolt, cm["lug_r_mm"]) for root, bolt in zip(roots, cal_bolts)]
    fill = []
    if cm.get("bridge"):
        # one plate: the bridge between the bolt bosses and the fill between the two lugs (the four slot
        # centres, which lie inside the slots, so the union has no slivers), so the lugs work as one deep
        # beam and the housing junction is one edge loop with no narrow gap for the fillet to choke on
        slots.append((cal_bolts[0], cal_bolts[1], cm["lug_r_mm"]))
        fill = [[roots[0], cal_bolts[0], cal_bolts[1], roots[1]]]
    lugs = B.plate(slots, fill, cm["lug_t_mm"], s)

    # ---- assemble: each arm is fused and its junction with the housing filleted straight away
    # (the largest of 8, 6, 5, 3 mm that OCC accepts), before the next one, because OCC refuses many fillets
    # at once; then each group of clevis ears, whose roots get the machining radius (3 mm, else 2 mm): an
    # unfilleted re-entrant corner is a stress singularity, whose FEA peak grows without limit as the mesh
    # is refined
    report = {"fillets": []}
    # where the mass is: each feature's own volume before it is fused (overlaps with the housing counted in
    # the housing; the bores are cut later, so the housing figure here is the ring outside the 68 mm bore)
    dens = json.loads((DATA / "materials.json").read_text())["7075_T7351_plate"]["density_kg_m3"] * 1e-9
    bore = B.cylinder(on((a0 + a1) / 2), s, hub["od_mm"] / 2, (a1 - a0) / 2 + 1)
    ring, _ = occ.cut(occ.copy(housing), bore)
    report["feature_mass_kg"] = {"housing ring": sum(occ.getMass(d, t) for d, t in ring) * dens}
    for name, feat in (("caliper plate", lugs), ("post", post), ("lower web", web), ("lower clevis ears", ears_l),
                       ("steering-arm ears", ears_s), ("upper clevis ears", ears_u)):
        cp = occ.copy(feat)
        outside, _ = occ.cut(cp, occ.copy(housing))
        report["feature_mass_kg"][name] = sum(occ.getMass(d, t) for d, t in outside) * dens
        occ.remove(outside, recursive=True)
    occ.remove(ring, recursive=True)
    occ.synchronize()
    ear_radii = (ud["fillets"]["min_mm"], 2.0)
    root_r = ud["fillets"]["load_root_mm"]
    radii = sorted({root_r, 6.0, 5.0, ud["fillets"]["min_mm"]}, reverse=True)
    radii = [x for x in radii if x <= root_r]
    upright, seen = housing, []
    for name, feat in (("caliper lugs", lugs), ("post", post), ("lower web", web)):
        upright = B.fuse(upright + feat)
        occ.synchronize()
        upright, seen = fillet_new_junctions(B, upright, seen, name, wc, s, R_h, a0, a1, radii, report)
    for name, ears, plates in (("upper clevis ears", ears_u, plates_u), ("lower ball-joint ears", ears_l, plates_l),
                               ("steering-arm ears", ears_s, plates_s)):
        upright = B.fuse(upright + ears)
        occ.synchronize()
        for i, plate in enumerate(plates):
            upright = fillet_edges(B, upright, plate_junction_edges(B, upright, plate), f"{name[:-1]} {i + 1}", ear_radii, report)

    # ---- holes
    tools = [(3, occ.addCylinder(*on(a0 - 1), *(s * (rib0 - a0 + 1)), hub["od_mm"] / 2)),
             (3, occ.addCylinder(*on(rib1), *(s * (a1 + 1 - rib1)), hub["od_mm"] / 2)),
             (3, occ.addCylinder(*on(a0 - 2), *(s * (a1 - a0 + 4)), hz["rib_bore_mm"] / 2))]
    tools += B.cylinder(lbj, k, bj_l["bolt_d_mm"] / 2 + 0.05, 45.0)
    lbj_up = lbj + k * (lcanode_h / 2 + spacer + ear_t / 2)                # upper ear: the insert's tap drill through ear + boss
    tools += B.cylinder(lbj_up + k * ib["height_mm"] / 2, k, ib["tap_drill_mm"] / 2, (ear_t + ib["height_mm"]) / 2 + 1.0)
    tools += B.cylinder(tro, b_tie, sa["bolt_d_mm"] / 2 + 0.05, 30.0)
    for bolt in cal_bolts:
        tools += B.cylinder(bolt, s, cm["bolt_hole_mm"] / 2, 20.0)
    tools += B.cylinder(ubj, k, bj_u["bolt_d_mm"] / 2 + 0.05, 40.0)
    faces = {"spin": s.tolist(), "wc": wc.tolist(), "seat_r": hub["od_mm"] / 2, "rib_r": hz["rib_bore_mm"] / 2,
             "seats": [[a0, rib0], [rib1, a1]], "rib": [rib0, rib1], "holes": []}
    for name, centre, axis, rad, ears in (
            ("UBJ", ubj, k, bj_u["bolt_d_mm"] / 2 + 0.05, [ubj + k * sg * (ucanode_h / 2 + spacer + ear_t / 2) for sg in (1, -1)]),
            ("LBJ", lbj, k, bj_l["bolt_d_mm"] / 2 + 0.05, [lbj + k * sg * (lcanode_h / 2 + spacer + ear_t / 2) for sg in (1, -1)]),
            ("TR", tro, b_tie, sa["bolt_d_mm"] / 2 + 0.05, [tro + b_tie * sg * (sa["rod_end_w_mm"] / 2 + spacer + ear_t / 2) for sg in (1, -1)])):
        for i, e in enumerate(ears):
            hole = {"name": f"{name}{i + 1}", "joint": name, "centre": e.tolist(), "axis": unit(axis).tolist(),
                    "radius": rad, "half_len": ear_t / 2}
            if name == "LBJ" and i == 0:          # the insert's ear: tap drill, and the boss on its outer face
                hole.update(radius=ib["tap_drill_mm"] / 2, boss_mm=ib["height_mm"], insert=bj_l["insert"])
            faces["holes"].append(hole)
    for i, bolt in enumerate(cal_bolts):
        faces["holes"].append({"name": f"bolt{i + 1}", "joint": f"bolt{i + 1}", "centre": bolt.tolist(), "axis": s.tolist(),
                               "radius": cm["bolt_hole_mm"] / 2, "half_len": cm["lug_t_mm"] / 2})
    upright = B.cut(upright, tools)
    occ.synchronize()


    if side == "R":
        occ.mirror(upright, 0, 1, 0, 0)
        occ.synchronize()
    geo = {"upright": upright}
    report.update({"faces": faces, "R_h": R_h, "axial": [a0, a1], "rib": [rib0, rib1], "post_axial": [post_in, post_in + post_t],
                   "web_axial": [web_in, web_out], "caliper_bolts": [b.tolist() for b in cal_bolts]})
    return B, geo, report


def ear(B, bolt, k, s, radius, thick, inboard_edge, face_axial, grow=0.0, inset=1.0, clip=None):
    """Clevis ear (plate normal to the bolt axis k, centred on `bolt`) running outboard from the bolt into the
    plate or web whose outboard face is at `face_axial` (measured from the wheel centre plane along s),
    ending `inset` mm inside it: a round end exactly touching that face would leave a tangent sliver that
    defeats the root fillets. The ear is a slot of radius `radius`, shifted outboard so the edge distance on
    the inboard side is only `inboard_edge` (the large ball-joint loads push the bolt outboard or sideways).
    `grow` adds a boss |grow| thick on the side the ear points to along k (away from the joint), only within
    clip = (wc, lo, hi), the axial range of the housing, where the ear would otherwise graze it; the face the
    joint bears on stays where it was. Returns the solid and the plate geometries [(c0, c1, r, thick, normal)]."""
    reach = perp(k, s)                                     # s made normal to k: outboard, in the ear's plane
    rs = float(np.dot(reach, s))
    wc_ax = float(np.dot(bolt, s))                         # absolute axial coordinates: compare differences only
    start = bolt + reach * (radius - inboard_edge)
    length = (face_axial - inset - radius * rs - (wc_ax - B.wc_ax)) / rs
    end = bolt + reach * max(length, radius - inboard_edge + 0.1)
    plate = B.stadium(start, end, radius, thick, k)
    geoms = [(start, end, radius, thick, k)]
    if grow:
        off = k * (thick / 2 + grow / 2)
        boss = B.stadium(start + off, end + off, radius, abs(grow), k)
        c, lo, hi = clip
        slab = [(3, B.occ.addCylinder(*(c + s * lo), *(s * (hi - lo)), 500.0))]
        boss, _ = B.occ.intersect(boss, slab)
        plate = B.fuse(plate + boss)
        geoms.append((start + off, end + off, radius, abs(grow), k))
    return plate, geoms


def junction_edges(B, dimtags, wc, s, R_h, a0, a1):
    """{edge tag: midpoint} for edges lying on the housing's outer cylinder, other than its end circles:
    the junctions of the post, web, lugs and ears with the housing (a junction may run out to an end face,
    as the caliper plate's does, being flush with the inboard end)."""
    g = B.g
    out = {}
    surfaces = g.model.getBoundary(dimtags, combined=False, oriented=False)
    for d, e in g.model.getBoundary(surfaces, combined=False, oriented=False):
        e = abs(e)
        if len(set(g.model.getAdjacencies(1, e)[0])) < 2:          # the cylinder's own seam: one face both sides
            continue
        lo, hi = g.model.getParametrizationBounds(1, e)
        pts = np.array(g.model.getValue(1, e, list(np.linspace(lo[0], hi[0], 7)))).reshape(-1, 3)
        rel = pts - wc
        axial = rel @ s
        radial = np.linalg.norm(rel - np.outer(axial, s), axis=1)
        on_end = np.all(np.abs(axial - a0) < 0.05) or np.all(np.abs(axial - a1) < 0.05)   # the housing's own end circles
        if np.all(np.abs(radial - R_h) < 0.3) and np.all((axial > a0 - 0.05) & (axial < a1 + 0.05)) and not on_end:
            out[e] = pts[3]
    return out


def plate_junction_edges(B, dimtags, plates, tol=0.05):
    """Edges where something else meets one of the flat slot plates (c0, c1, radius, thick, normal): edges
    lying inside a plate face, or along its side wall strictly within its thickness. The plate's own outline
    lies on both boundaries at once and is not picked. These are the re-entrant corners at the ear roots."""
    g = B.g
    out = []
    surfaces = g.model.getBoundary(dimtags, combined=False, oriented=False)
    edges = {abs(e) for _, e in g.model.getBoundary(surfaces, combined=False, oriented=False)}
    for e in sorted(edges):
        lo, hi = g.model.getParametrizationBounds(1, e)
        pts = np.array(g.model.getValue(1, e, list(np.linspace(lo[0], hi[0], 7)))).reshape(-1, 3)
        for c0, c1, rad, thick, n in plates:
            n = unit(n)
            c0 = np.asarray(c0, float)
            c1 = np.asarray(c1, float) - n * np.dot(np.asarray(c1, float) - c0, n)
            h = (pts - c0) @ n
            seg = c1 - c0
            tt = np.clip(((pts - c0) @ seg) / max(seg @ seg, 1e-12), 0.0, 1.0)
            q = pts - n * h[:, None]                                  # projected into the plate's mid-plane
            d = np.linalg.norm(q - (c0 + np.outer(tt, seg)), axis=1)
            on_face = (np.abs(np.abs(h) - thick / 2) < tol) & (d < rad - tol)
            on_wall = (np.abs(d - rad) < tol) & (np.abs(h) < thick / 2 - tol)
            on_plate = (d < rad + tol) & (np.abs(h) < thick / 2 + tol)      # an end may sit on the plate's outline
            if np.all((on_face | on_wall)[1:-1]) and np.all(on_plate[[0, -1]]) and is_concave(B, dimtags, e):
                out.append(e)
                break
    return out


def is_concave(B, dimtags, e, eps=0.3, n=16):
    """True if edge e is a re-entrant (inside) corner of the solid: on a small circle round the edge, normal
    to it, more than 60 % of the points are inside the material (about 75 % at a 90 deg inside corner,
    25 % at an outside corner, 50 % where two faces meet flush), at three stations along the edge."""
    g = B.g
    vols = [t for d, t in dimtags if d == 3]
    lo, hi = g.model.getParametrizationBounds(1, e)
    for f in (0.25, 0.5, 0.75):
        u = lo[0] + f * (hi[0] - lo[0])
        m = np.array(g.model.getValue(1, e, [u]))
        tan = unit(np.array(g.model.getDerivative(1, e, [u])))
        e1 = unit(np.cross(tan, [1.0, 0.0, 0.0]) if abs(tan[0]) < 0.9 else np.cross(tan, [0.0, 1.0, 0.0]))
        e2 = np.cross(tan, e1)
        inside = 0
        for a in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
            p = m + eps * (np.cos(a) * e1 + np.sin(a) * e2)
            inside += any(g.model.isInside(3, v, p.tolist()) for v in vols)
        if inside / n <= 0.6:
            return False
    return True


def fillet_edges(B, dimtags, edges, name, radii, report):
    """Fillet the given edges together with the largest radius OCC accepts, on a copy so a refusal never
    costs the part; the outcome is recorded in report["fillets"]. OCC sometimes fails inside the fillet
    with an access violation, which corrupts the model (the next fillet can hang), so probe builds (each in
    its own process, see main) try every radius of the features not yet decided, stopping at the first
    access violation; the real build only tries the radius that worked (PLAN).
    Edges left sharp are recorded in report["sharp_edges"] (sampled every ~1 mm), so the FEA can keep the
    singular stress at a sharp re-entrant corner out of its peaks."""
    g, occ = B.g, B.occ
    entry = {"feature": name, "edges": len(edges), "radius_mm": None}
    probing = PROBING and name not in PLAN
    if not probing:
        radii = [PLAN[name]] if PLAN.get(name) else []
        if edges and not radii:
            entry["refused"] = ["no radius worked in the probe build"]
    for rad in radii:
        if not edges:
            break
        trial = occ.copy(dimtags)
        occ.synchronize()
        try:
            out = occ.fillet([t for d, t in trial if d == 3], edges_of(B, trial, edges, dimtags), [rad], removeVolume=True)
            occ.synchronize()
            occ.remove(dimtags, recursive=True)
            occ.synchronize()
            dimtags = [(3, t) for d, t in out if d == 3]
            entry["radius_mm"] = rad
            break
        except Exception as ex:
            occ.remove([x for x in trial if x in g.model.getEntities(3)], recursive=True)
            occ.synchronize()
            entry.setdefault("refused", []).append(f"{rad} mm: {str(ex)[:50]}")
            if probing and "access violation" in str(ex):
                print("PROBE " + json.dumps([name, None]), flush=True)       # the model is corrupt: stop here
                print("ABORT", flush=True)
                os._exit(3)
    report["fillets"].append(entry)
    if probing:
        print("PROBE " + json.dumps([name, entry["radius_mm"]]), flush=True)
    if edges and entry["radius_mm"] is None:
        pts = []
        for e in edges:
            lo, hi = g.model.getParametrizationBounds(1, e)
            ends = np.array(g.model.getValue(1, e, [lo[0], hi[0]])).reshape(2, 3)
            n = max(2, int(np.ceil(np.linalg.norm(ends[1] - ends[0]) / 1.0)) + 1)
            pts += np.array(g.model.getValue(1, e, list(np.linspace(lo[0], hi[0], n)))).reshape(-1, 3).round(2).tolist()
        report.setdefault("sharp_edges", []).append({"feature": name, "points": pts})
    return dimtags


def fillet_new_junctions(B, dimtags, seen, name, wc, s, R_h, a0, a1, radii, report):
    """Fillet the junction edges on the housing cylinder that are new since the last feature (a closed
    loop round one arm's root), on a copy so a refusal never costs the part. Returns the body and the
    midpoints of every junction edge seen so far (fillet boundaries also lie on the cylinder)."""
    g, occ = B.g, B.occ
    cand = junction_edges(B, dimtags, wc, s, R_h, a0, a1)
    new = [e for e, m in cand.items() if all(np.linalg.norm(m - q) > 1e-3 for q in seen)]
    entry = {"feature": name, "edges": len(new), "radius_mm": None}
    for rad in radii:
        if not new:
            break
        trial = occ.copy(dimtags)
        occ.synchronize()
        try:
            out = occ.fillet([t for d, t in trial if d == 3], edges_of(B, trial, new, dimtags), [rad], removeVolume=True)
            occ.synchronize()
            occ.remove(dimtags, recursive=True)
            occ.synchronize()
            dimtags = [(3, t) for d, t in out if d == 3]
            entry["radius_mm"] = rad
            break
        except Exception as ex:
            occ.remove([x for x in trial if x in g.model.getEntities(3)], recursive=True)
            occ.synchronize()
            entry.setdefault("refused", []).append(f"{rad} mm: {str(ex)[:50]}")
    report["fillets"].append(entry)
    seen = seen + [m for m in junction_edges(B, dimtags, wc, s, R_h, a0, a1).values()]
    return dimtags, seen


def edges_of(B, copy_dimtags, edges, original):
    """The copy's edges matching the original's selected edges (same geometry, new tags)."""
    g = B.g
    def mids(dts, wanted=None):
        out = {}
        for d, e in g.model.getBoundary(g.model.getBoundary(dts, combined=False, oriented=False), combined=False, oriented=False):
            e = abs(e)
            if wanted is not None and e not in wanted:
                continue
            lo, hi = g.model.getParametrizationBounds(1, e)
            out[e] = np.array(g.model.getValue(1, e, [0.5 * (lo[0] + hi[0])]))
        return out
    want = mids(original, set(edges))
    have = mids(copy_dimtags)
    sel = []
    for p in want.values():
        best = min(have.items(), key=lambda kv: np.linalg.norm(kv[1] - p))
        if np.linalg.norm(best[1] - p) < 1e-6:
            sel.append(best[0])
    if len(set(sel)) != len(set(edges)):
        raise ValueError(f"matched {len(set(sel))} of {len(set(edges))} edges on the copy")
    return sorted(set(sel))


PLAN = {}              # {feature: radius or None} decided by the probe builds
PROBING = False        # True in a probe build: features not in PLAN are tried at every radius


def probe():
    """Build the left upright trying every fillet radius, printing which radius each ear root took as it goes
    (PROBE lines). Windows' crash dialog is switched off, so an OCC access violation ends the process instead
    of waiting for a click."""
    global PLAN, PROBING
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    PLAN, PROBING = json.loads(sys.argv[sys.argv.index("--probe") + 1]), True
    _, _, rep = build("L")
    print("PLAN " + json.dumps({f["feature"]: f["radius_mm"] for f in rep["fillets"]}))


def main():
    global PLAN
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    mat = json.loads((DATA / "materials.json").read_text())["7075_T7351_plate"]
    PLAN = {}
    for attempt in range(12):
        try:
            res = subprocess.run([sys.executable, __file__, "--probe", json.dumps(PLAN)], capture_output=True, text=True,
                                 timeout=900)
            stdout, stderr = res.stdout, res.stderr
        except subprocess.TimeoutExpired as ex:
            stdout = ex.stdout.decode() if isinstance(ex.stdout, bytes) else (ex.stdout or "")
            stderr = "probe timed out"
        PLAN.update(dict(json.loads(l[6:]) for l in stdout.splitlines() if l.startswith("PROBE ")))
        if any(l.startswith("PLAN ") for l in stdout.splitlines()):
            break
        if "ABORT" not in stdout:
            raise RuntimeError("fillet probe build failed:\n" + stderr[-2000:])
        print(f"fillet probe {attempt + 1}: stopped after an OCC access violation; {len(PLAN)} features decided", flush=True)
    else:
        raise RuntimeError("fillet probes did not finish")
    summary = {}
    for side in ("L", "R"):
        B, geo, rep = build(side)
        g = B.g
        masses = {k: sum(B.occ.getMass(d, t) for d, t in v) * mat["density_kg_m3"] * 1e-9 for k, v in geo.items()}
        g.write(str(OUT / f"upright_{side}.step"))
        if side == "L":
            g.write(str(OUT / "upright_L_fea.step"))       # one solid: the FEA meshes the same part
            summary = {"masses_kg": masses, "total_kg": sum(masses.values()), **rep}
        g.finalize()
    (OUT / "upright_build.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("masses_kg", "total_kg", "fillets")}, indent=1, default=float))


if __name__ == "__main__":
    probe() if "--probe" in sys.argv else main()
