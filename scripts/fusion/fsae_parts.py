"""Part builders for one suspension corner (run INSIDE Fusion; mm in, temp bodies out).

Every builder takes `h`, one corner's hardpoints (the right corner passes mirrored points), so
geometry follows data/hardpoints.csv exactly. Directions are derived from points, never typed,
so they mirror correctly. Dimensions that are not hardpoints are named constants below with
their basis (catalogue data or a packaging assumption).
"""
import math

import adsk.core
import adsk.fusion

import fsae_cad as C

# ---------------- catalogue / packaging constants (mm) ----------------
# Values marked (data) are overwritten by configure() from data/vehicle.json and data/members.csv,
# so a change there reaches the CAD; the defaults only keep the module importable.
RIM_BEAD_R = 127.0          # 10 in rim
RIM_HALF_W = 76.2           # (data) half the rim width between flanges
RIM_WALL = 6.0              # barrel wall (assumed; Braid inner profile not published)
RIM_FLANGE_R = 136.0
WHEEL_OFFSET = 35.0         # (data) ET35: mounting face outboard of the wheel centre plane
TYRE_R = 205.75             # (data) Hoosier 43075 unloaded diameter 411.5
TYRE_TREAD_HALF = 92.5      # (data) 185 mm tread
TYRE_BULGE_HALF = 96.5      # section width about 7.6 in on a 6 in rim (8.4 in is quoted on an 8 in rim)
TYRE_BULGE_R = 165.0        # sidewall widest at about this radius
TYRE_SHOULDER = 18.0
BRG_OD, BRG_ID, BRG_W = 68.0, 40.0, 19.0   # (data: vehicle.json hub) SKF 32008 X tapered rollers, back to back
BRG_AX = (-15.0, 6.0)       # (data) bearing centres along the spin axis from the wheel centre (+ outboard)
HOUSING = (-26.5, 17.5, 40.0)  # (data-derived) upright bearing housing: axial from, to, outer radius
DISC_AX, DISC_T, DISC_RO, DISC_RI = 14.0, 4.0, 90.0, 56.0   # (data: OD) floating disc, 4 mm outboard of the LBJ clevis ears
# floating mount (D23, S-16): a 7075 carrier ring in the disc's plane, 2 mm inside the disc, joined to it by titanium
# bobbins across the gap, and a carrier hat out to the hub flange. The solid steel hat of v1.7 (56-60 mm) passed
# the disc's heat straight into the hub (HC-8)
CARRIER = (50.0, 54.0)      # carrier radii: one tube from the disc plane to the hub flange, 3.8 mm clear of the upright
BOBBIN_R, BOBBIN_D, N_BOBBINS = 56.0, 11.0, 6   # 50.5-61.5 mm: not tangent to the carrier's 50 mm bore
BOBBIN_HEAD = (0.5, 1.0)    # bobbin heads proud of the disc, inboard / outboard (the inboard side faces the upright)
HUB_FLANGE = (21.0, 35.0, 60.0)  # axial from, to, radius (4 mm of flange outside the M12 studs at r 50)
STUD_PCD_R, STUD_R, STUD_L, N_STUDS = 50.0, 6.0, 32.0, 4
# Caliper envelope from the Wilwood PS-1 drawing (vehicle.json brake.caliper; set by configure()): a body
# from the pads' inner edge to the bridge beyond the disc edge, centred on the disc, plus two mounting ears
# on the inboard side around the lug bolts.
CALIPER = {"r_in": 65.0, "r_out": 113.9, "tangential": 96.3, "ax_from": DISC_AX - 17.15, "ax_to": DISC_AX + 17.15,
           "r_bolt": 59.1, "bolt_half_spacing": 31.75, "mount_face_ax": DISC_AX - 21.1, "ear_t": 10.4}
BJ_HOUSING_R = {"UCA": 14.0, "LCA": 14.8}   # (data: upright_design.json ball joints) node outer radius = bearing OD / 2 + 4.5
BJ_BORE_R = {"UCA": 9.525, "LCA": 10.32}     # (data) COM-5 upper (0.750 in OD), COM-6 lower (0.8125 in OD)
BJ_HOUSING_H = {"UCA": 22.0, "LCA": 25.0}  # (data) node height = leg tube OD + 6 mm so the legs weld inside it
EAR_T, EAR_GAP = 6.0, 5.0   # upright clevis ear thickness; misalignment spacer each side: the ear must clear the node
                            # rim (r 14) beyond the COM-5's 12 deg rating (14 sin 12 = 2.9 mm) and the legs leaving the node
ROD_END_HEAD_R, ROD_END_W, ROD_END_SHANK = 12.0, 12.0, 40.0  # Aurora AM-M8 envelope (shank: data)
TUBE_LEG = (15.875 / 2, 15.875 / 2 - 0.889)    # (data) members.csv, as (outer, inner radius)
TUBE_LCA = (19.05 / 2, 19.05 / 2 - 0.889)      # (data)
TUBE_TIE = (15.875 / 2, 15.875 / 2 - 1.245)    # (data)
TUBE_PUSH = (19.05 / 2, 19.05 / 2 - 0.889)     # (data)
ARB_BAR_R, ARB_END_R, ARB_HUB_WALL = 6.2, 11.0, 6.0   # (data) vehicle.json arb_bar: fitted body, splined ends, hub
ARB_HUB_HALF, ARB_RUNOUT = 8.0, 5.0                   # (data) lever hub half-length; the upset starts this far inboard
ARB_BAR_STUB = 30.0         # bar runs 30 mm outboard of ARB_P (its chassis bearing), past the lever hub
ARB_BLADE_T, ARB_BLADE_W, ARB_LEVER_OFFSET = 8.0, 25.0, 15.0   # (data) vehicle.json arb_bar: lever t, w; plane inboard of ARB_L
PUSH_TAB_T = 3.0            # (data) members.csv tab thickness of the pushrod's LCA clevis


def configure(veh, members, ud=None):
    """Set the (data) constants from vehicle.json (dict), members.csv (rows as dicts) and upright_design.json."""
    global RIM_HALF_W, WHEEL_OFFSET, TYRE_R, TYRE_TREAD_HALF, ROD_END_SHANK, BJ_HOUSING_H
    global TUBE_LEG, TUBE_LCA, TUBE_TIE, TUBE_PUSH, ARB_BAR_R, ARB_END_R, ARB_HUB_WALL, PUSH_TAB_T, ARB_BLADE_T, ARB_BLADE_W
    global ARB_LEVER_OFFSET, ARB_HUB_HALF, ARB_RUNOUT
    RIM_HALF_W = veh["wheel"]["rim_width_in"] * 25.4 / 2
    WHEEL_OFFSET = float(veh["wheel"]["offset_mm"])
    TYRE_R = veh["tyre"]["unloaded_diameter_mm"] / 2
    TYRE_TREAD_HALF = veh["tyre"]["tread_width_mm"] / 2
    m = {r["member"]: r for r in members}

    def tube(name):
        od, wall = float(m[name]["tube_od_mm"]), float(m[name]["tube_wall_mm"])
        return od / 2, od / 2 - wall

    TUBE_LEG, TUBE_LCA, TUBE_TIE, TUBE_PUSH = tube("UCA front leg"), tube("LCA front leg"), tube("Tie rod"), tube("Pushrod")
    ROD_END_SHANK = float(m["Pushrod"]["rod_end_shank_mm"])
    PUSH_TAB_T = float(m["Pushrod"]["tab_t_mm"])
    BJ_HOUSING_H = {"UCA": 2 * TUBE_LEG[0] + 6.0, "LCA": 2 * TUBE_LCA[0] + 6.0}
    bar = veh["arb_bar"]
    ARB_BAR_R, ARB_END_R = bar["bars_mm"][bar["fitted"]] / 2, bar["end_od_mm"] / 2
    ARB_HUB_WALL = float(bar["lever_hub_wall_mm"])
    ARB_HUB_HALF, ARB_RUNOUT = float(bar["lever_hub_half_len_mm"]), float(bar["upset_runout_mm"])
    ARB_BLADE_T, ARB_BLADE_W = float(bar["lever_blade_t_mm"]), float(bar["lever_blade_w_mm"])
    ARB_LEVER_OFFSET = float(bar["lever_offset_mm"])
    global DISC_RO, CALIPER, DISC_AX, BRG_OD, BRG_ID, BRG_W, BRG_AX, HOUSING, BJ_HOUSING_R, BJ_BORE_R, EAR_T, EAR_GAP
    if ud:
        bj = ud["ball_joints"]
        BJ_BORE_R = {"UCA": bj["upper"]["od_in"] * 25.4 / 2, "LCA": bj["lower"]["od_in"] * 25.4 / 2}
        BJ_HOUSING_R = {k: v + 4.5 for k, v in BJ_BORE_R.items()}
        EAR_T, EAR_GAP = ud["clevis"]["ear_t_mm"], ud["clevis"]["spacer_mm"]
    hub = veh["hub"]
    BRG_OD, BRG_ID, BRG_W = hub["od_mm"], hub["bore_mm"], hub["width_mm"]
    BRG_AX = tuple(hub["row_centres_axial_mm"])
    HOUSING = (BRG_AX[0] - BRG_W / 2 - 2.0, BRG_AX[1] + BRG_W / 2 + 2.0, BRG_OD / 2 + 6.0)   # shoulder / ring land, 6 mm wall
    DISC_AX = hub["disc_axial_mm"]
    DISC_RO = veh["brake"]["disc_od_mm_assumed"] / 2
    cal = veh["brake"]["caliper"]
    CALIPER = {"r_in": DISC_RO - cal["pad_height_mm_estimated"], "r_out": DISC_RO + cal["bridge_beyond_disc_od_mm"],
               "tangential": cal["length_mm"], "ax_from": DISC_AX - cal["width_axial_mm"] / 2,
               "ax_to": DISC_AX + cal["width_axial_mm"] / 2, "r_bolt": DISC_RO - cal["mount_below_disc_od_mm"],
               "bolt_half_spacing": cal["bolt_spacing_mm"] / 2, "mount_face_ax": DISC_AX - cal["mount_offset_mm"],
               "ear_t": cal["ear_thickness_mm"]}


def spin(h):
    return C.unit(C.sub(h["WA"], h["WC"]))


def kingpin(h):
    return C.unit(C.sub(h["UCA_O"], h["LCA_O"]))


def wheel_frame(h):
    """(axis s, up u, rear r) unit vectors of the non-spinning wheel frame."""
    s = spin(h)
    u = C.perp(s, (0.0, 0.0, 1.0))
    r = C.unit(C.cross(s, u))
    if C.dot(r, (-1.0, 0.0, 0.0)) < 0:
        r = C.mul(r, -1.0)
    return s, u, r


def on_axis(h, ax, radial=(0.0, 0.0, 0.0)):
    return C.add(C.add(h["WC"], C.mul(spin(h), ax)), radial)


def torus(center, axis, R, r):
    t = C.tm()
    b = t.createTorus(adsk.core.Point3D.create(0, 0, 0), C.V(axis), R * C.CM, r * C.CM)
    m = adsk.core.Matrix3D.create()
    m.translation = adsk.core.Vector3D.create(center[0] * C.CM, center[1] * C.CM, center[2] * C.CM)
    t.transform(b, m)
    return b


def rod_end(center, bolt_axis, shank_dir):
    """Aurora AM-M8 envelope: head ring around the bolt axis + threaded shank along shank_dir."""
    head = C.ring(center, bolt_axis, ROD_END_HEAD_R, 5.0, ROD_END_W)   # bore drawn 1 mm oversize: the ball turns with the bolt
    d = C.unit(shank_dir)
    shank = C.cyl(C.add(center, C.mul(d, ROD_END_HEAD_R - 1.0)), C.add(center, C.mul(d, ROD_END_SHANK)), 4.0)
    return C.union([head, shank])


def clevis(center, bolt_axis, toward, gap, reach=22.0, r=11.0):
    """Two ear plates normal to bolt_axis, either side of a joint of width `gap`, running `reach`
    mm from the bolt towards `toward`; returns one body (ears joined by a web at the far end)."""
    b = C.unit(bolt_axis)
    d = C.unit(C.sub(C.sub(toward, center), C.mul(b, C.dot(C.sub(toward, center), b))))
    off = gap / 2 + EAR_T / 2
    ears = []
    for sgn in (1.0, -1.0):
        c0 = C.add(center, C.mul(b, sgn * off))
        ears.append(C.plate(c0, C.add(c0, C.mul(d, reach)), r, EAR_T, b))
    web_c = C.add(center, C.mul(d, reach))
    web = C.box(web_c, b, C.unit(C.cross(b, d)), 2 * off + EAR_T, 2 * r, 2 * r * 0.6)
    body = C.union(ears + [web])
    return C.cut(body, C.cyl(C.add(center, C.mul(b, off + 10)), C.sub(center, C.mul(b, off + 10)), 4.0))


# ---------------- wheel group ----------------
def wheel(h):
    s, u, r = wheel_frame(h)
    c = h["WC"]
    barrel = C.tube(C.sub(c, C.mul(s, RIM_HALF_W)), C.add(c, C.mul(s, RIM_HALF_W)), RIM_BEAD_R, RIM_BEAD_R - RIM_WALL)
    flanges = [C.ring(C.add(c, C.mul(s, sg * (RIM_HALF_W + 4))), s, RIM_FLANGE_R, RIM_BEAD_R - RIM_WALL, 8.0) for sg in (1, -1)]
    face = C.add(c, C.mul(s, WHEEL_OFFSET))
    centre = C.tube(face, C.add(face, C.mul(s, 10.0)), RIM_BEAD_R - RIM_WALL + 1, 30.0)
    for i in range(5):  # five lightening windows between the spokes
        a = 2 * math.pi * i / 5 + math.pi / 5
        rad = C.add(C.mul(u, math.cos(a) * 82), C.mul(r, math.sin(a) * 82))
        p = C.add(face, rad)
        centre = C.cut(centre, C.cyl(C.sub(p, C.mul(s, 2)), C.add(p, C.mul(s, 12)), 24.0))
    for i in range(N_STUDS):
        a = 2 * math.pi * i / N_STUDS + math.pi / 4
        rad = C.add(C.mul(u, math.cos(a) * STUD_PCD_R), C.mul(r, math.sin(a) * STUD_PCD_R))
        p = C.add(face, rad)
        centre = C.cut(centre, C.cyl(C.sub(p, C.mul(s, 2)), C.add(p, C.mul(s, 12)), STUD_R + 0.5))
    return C.union([barrel] + flanges + [centre])


def tyre(h):
    """Tyre envelope on the rim: tread 185 mm, sidewalls bulging to about 7.6 in at r = 165 mm and
    pinched to the rim flanges at the bead (a full-width block down to the bead overstates it)."""
    s_, _, _ = wheel_frame(h)
    c = h["WC"]
    bead_half = RIM_HALF_W + 8.0
    body = C.cyl(C.sub(c, C.mul(s_, TYRE_TREAD_HALF - TYRE_SHOULDER)), C.add(c, C.mul(s_, TYRE_TREAD_HALF - TYRE_SHOULDER)), TYRE_R)
    tori = [torus(C.add(c, C.mul(s_, sg * (TYRE_TREAD_HALF - TYRE_SHOULDER))), s_, TYRE_R - TYRE_SHOULDER, TYRE_SHOULDER)
            for sg in (1, -1)]
    # sidewall: cone from the flange (bead_half, r 140) out to the bulge, then to the shoulder
    walls = []
    for sg in (1, -1):
        a0, a1 = C.add(c, C.mul(s_, sg * bead_half)), C.add(c, C.mul(s_, sg * TYRE_BULGE_HALF))
        walls.append(C.cyl(C.sub(c, C.mul(s_, 0.0)), a0, TYRE_R - TYRE_SHOULDER))
        walls.append(C.cyl(a0, a1, TYRE_BULGE_R + 25.0, TYRE_BULGE_R))
    body = C.union([body] + tori + walls)
    body = C.cut(body, C.cyl(C.sub(c, C.mul(s_, bead_half + 30)), C.add(c, C.mul(s_, bead_half + 30)), RIM_BEAD_R))
    for sg in (1, -1):  # beyond the flanges the tyre starts at the sidewall, not at the bead
        a0, a1 = C.add(c, C.mul(s_, sg * bead_half)), C.add(c, C.mul(s_, sg * (TYRE_BULGE_HALF + 5)))
        body = C.cut(body, C.cyl(a0, a1, TYRE_BULGE_R - 20.0))
    return body


def hub(h):
    s, u, r = wheel_frame(h)
    f0, f1, fr = HUB_FLANGE
    flange = C.cyl(on_axis(h, f0), on_axis(h, f1), fr)
    spindle = C.tube(on_axis(h, HOUSING[0] - 12), on_axis(h, f0 + 0.5), BRG_ID / 2, 13.0)
    nut = C.tube(on_axis(h, HOUSING[0] - 22), on_axis(h, HOUSING[0] - 10), 17.0, BRG_ID / 2 - 6)
    studs = []
    for i in range(N_STUDS):
        a = 2 * math.pi * i / N_STUDS + math.pi / 4
        rad = C.add(C.mul(u, math.cos(a) * STUD_PCD_R), C.mul(r, math.sin(a) * STUD_PCD_R))
        studs.append(C.cyl(on_axis(h, f1 - 3, rad), on_axis(h, f1 + STUD_L, rad), STUD_R))
    body = C.union([flange, spindle] + studs)
    return body, nut


def bearings(h):
    s = spin(h)
    return [C.ring(on_axis(h, a), s, BRG_OD / 2, BRG_ID / 2, BRG_W) for a in BRG_AX]


def disc(h):
    """Floating disc (one envelope body): friction ring, the carrier (ring and hat as one tube), and the bobbins."""
    s, u, r = wheel_frame(h)
    ring = C.ring(on_axis(h, DISC_AX), s, DISC_RO, DISC_RI, DISC_T)
    carrier = C.tube(on_axis(h, DISC_AX - DISC_T / 2), on_axis(h, HUB_FLANGE[0]), CARRIER[1], CARRIER[0])   # ring + hat as one
    bobbins = []
    for i in range(N_BOBBINS):
        a = 2 * math.pi * i / N_BOBBINS
        rad = C.add(C.mul(u, math.cos(a) * BOBBIN_R), C.mul(r, math.sin(a) * BOBBIN_R))
        bobbins.append(C.cyl(on_axis(h, DISC_AX - DISC_T / 2 - BOBBIN_HEAD[0], rad),
                             on_axis(h, DISC_AX + DISC_T / 2 + BOBBIN_HEAD[1], rad), BOBBIN_D / 2))
    return C.union([ring, carrier] + bobbins)


def caliper(h):
    """Wilwood PS-1 envelope behind the axle: a body from the pads' inner edge to the bridge beyond the
    disc edge, with a slot the disc runs through, and two lug-mount ears (axial bolts) on its inboard side."""
    s, u, r = wheel_frame(h)
    cfg = CALIPER
    tang = C.unit(C.cross(s, r))
    r_mid, depth = (cfg["r_in"] + cfg["r_out"]) / 2, cfg["r_out"] - cfg["r_in"]
    body = C.box(on_axis(h, (cfg["ax_from"] + cfg["ax_to"]) / 2, C.mul(r, r_mid)), tang, s,
                 cfg["tangential"], cfg["ax_to"] - cfg["ax_from"], depth)
    s_lo, s_hi = cfg["r_in"] - 2.0, DISC_RO + 3.0       # slot open at the bottom, 3 mm above the disc edge
    slot = C.box(on_axis(h, DISC_AX, C.mul(r, (s_lo + s_hi) / 2)), tang, s, cfg["tangential"] + 2, DISC_T + 3, s_hi - s_lo)
    # the outer face is an arc at r_out (the drawing gives an outside radius), not the corners of a box
    beyond = C.ring(on_axis(h, (cfg["ax_from"] + cfg["ax_to"]) / 2), s, cfg["r_out"] + 80.0, cfg["r_out"],
                    cfg["ax_to"] - cfg["ax_from"] + 2.0)
    parts = [C.cut(C.cut(body, slot), beyond)]
    a0, a1 = cfg["mount_face_ax"], cfg["mount_face_ax"] + cfg["ear_t"]
    for sg in (1.0, -1.0):
        c = C.add(C.mul(r, cfg["r_bolt"]), C.mul(tang, sg * cfg["bolt_half_spacing"]))
        ear = C.box(on_axis(h, (a0 + a1) / 2, c), tang, s, 18.0, a1 - a0, 2 * (cfg["r_in"] + 2 - cfg["r_bolt"]))
        parts.append(C.cut(ear, C.cyl(on_axis(h, a0 - 1, c), on_axis(h, a1 + 1, c), 4.05)))
    return C.union(parts)


# ---------------- upright ----------------
def upright(h):
    s, u, r = wheel_frame(h)
    k = kingpin(h)
    a0, a1, ro = HOUSING
    housing = C.tube(on_axis(h, a0), on_axis(h, a1), ro, BRG_OD / 2)
    parts = [housing]
    for bj, side in (("UCA_O", 1.0), ("LCA_O", -1.0)):
        p = h[bj]
        gap = BJ_HOUSING_H[bj[:3]] + 2 * EAR_GAP
        root = on_axis(h, -8.0, C.mul(u, side * (ro - 6)))     # where the arm leaves the housing
        parts.append(C.strut(root, C.add(p, C.mul(C.unit(C.sub(root, p)), 28.0)), 22.0, 16.0, up=s))  # ends inside the clevis web
        parts.append(clevis(p, k, root, gap, reach=24.0, r=12.0))   # web 3 mm clear of the node; LBJ ear tips 4 mm inboard of the disc
    # steering arm to the tie-rod rod end; bolt normal to the tie rod, closest to the kingpin
    tie = C.sub(h["TR_O"], h["TR_I"])
    b_tie = C.perp(tie, k)
    root = on_axis(h, -6.0, C.mul(C.unit(C.sub((h["TR_O"][0], h["WC"][1], h["TR_O"][2]), h["WC"])), ro - 6))
    parts.append(C.strut(root, C.add(h["TR_O"], C.mul(C.unit(C.sub(root, h["TR_O"])), 24.0)), 18.0, 14.0, up=s))
    parts.append(clevis(h["TR_O"], b_tie, root, ROD_END_W + 2 * EAR_GAP, reach=20.0, r=11.0))
    # caliper lugs: the caliper's ears bolt axially onto their inboard faces (PS-1 lug mount)
    cfg = CALIPER
    tang = C.unit(C.cross(s, r))
    for sg in (1.0, -1.0):
        bolt = C.add(C.mul(r, cfg["r_bolt"]), C.mul(tang, sg * cfg["bolt_half_spacing"]))
        lug_end = on_axis(h, cfg["mount_face_ax"] - 6.0, bolt)
        lug_root = on_axis(h, cfg["mount_face_ax"] - 6.0, C.add(C.mul(r, ro - 5), C.mul(tang, sg * 18)))
        lug = C.strut(lug_root, C.add(lug_end, C.mul(C.unit(C.sub(lug_end, lug_root)), 8.0)), 16.0, 12.0, up=s)
        parts.append(C.cut(lug, C.cyl(on_axis(h, cfg["mount_face_ax"] - 14, bolt), on_axis(h, cfg["mount_face_ax"] + 1, bolt), 4.05)))
    return C.union(parts)


# ---------------- wishbones and links ----------------
def wishbone(h, prefix):
    """Weldment: node housing at the ball joint (axis = kingpin) + two legs ending at rod-end
    shanks. Returns (node body, legs body, [rod-end bodies]). Node and legs stay separate bodies
    so the interference sweep can tell the designed joint (upright clevis around the node) from
    a clash with the legs."""
    o = h[f"{prefix}_O"]
    k = kingpin(h)
    hh = BJ_HOUSING_H[prefix]
    node = C.tube(C.sub(o, C.mul(k, hh / 2)), C.add(o, C.mul(k, hh / 2)), BJ_HOUSING_R[prefix], BJ_BORE_R[prefix])
    legs, rod_ends = [], []
    piv = C.unit(C.sub(h[f"{prefix}_R"], h[f"{prefix}_F"]))
    for leg in ("F", "R"):
        pin = h[f"{prefix}_{leg}"]
        d = C.unit(C.sub(o, pin))
        tube_start = C.add(pin, C.mul(d, ROD_END_SHANK))
        tube_end = C.add(o, C.mul(d, -BJ_HOUSING_R[prefix] + 2))
        legs.append(C.tube(tube_start, tube_end, *(TUBE_LCA if prefix == "LCA" else TUBE_LEG)))
        bolt = C.perp(d, piv)
        rod_ends.append(rod_end(pin, bolt, d))
    return node, C.union(legs), rod_ends


def lca_pushrod_bracket(h):
    """Double-shear clevis for the pushrod rod end on the lower-wishbone node: two plates in the
    pushrod's plane (bolt normal to the pushrod, along the LCA pivot axis), each running from the
    pickup out to a knee and down to the side of the node, welded there at mid-height.

    The pushrod line passes through the ball-joint centre, but it leaves the node through its top
    face, where the upright's clevis ear sits; so the plates cannot follow the line into the node.
    The knee keeps them 24 mm from the kingpin axis at the ear's height (about 5 mm clear of the
    ear disc) and costs a local bending moment in the plates (pushrod force x the knee's offset
    from the line), checked by hand in Week 3."""
    k = kingpin(h)
    o, p = h["LCA_O"], h["PR_O"]
    piv = C.unit(C.sub(h["LCA_R"], h["LCA_F"]))
    b = C.perp(C.sub(h["PR_I"], p), piv)                  # bolt axis
    rho = C.perp(k, C.sub(p, o))                          # from the node axis towards the pickup

    def in_plane(q):                                      # into the plates' mid-plane (through PR_O, normal b)
        return C.sub(q, C.mul(b, C.dot(C.sub(q, p), b)))

    knee = in_plane(C.add(o, C.add(C.mul(rho, 27.0), C.mul(k, 10.0))))   # 27 mm: clear of the detailed upright's ears
    foot = in_plane(C.add(o, C.mul(rho, BJ_HOUSING_R["LCA"] - 3.0)))   # 3 mm into the node side, at mid-height (clear of both ears)
    t = PUSH_TAB_T
    off = (ROD_END_W + 2 * EAR_GAP) / 2 + t / 2
    plates = []
    for sg in (1.0, -1.0):
        o_ = C.mul(b, sg * off)
        boss = C.cyl(C.add(C.add(p, o_), C.mul(b, -t / 2)), C.add(C.add(p, o_), C.mul(b, t / 2)), 12.0)  # e/D 1.5 at the bolt
        plates += [boss, C.plate(C.add(p, o_), C.add(knee, o_), 8.0, t, b), C.plate(C.add(knee, o_), C.add(foot, o_), 8.0, t, b)]
    br = C.union(plates)
    return C.cut(br, C.cyl(C.add(p, C.mul(b, off + 10)), C.sub(p, C.mul(b, off + 10)), 4.0))


def link(p_a, p_b, tube_r, bolt_a, bolt_b):
    """Tube with a rod end at each end (tie rod, pushrod, drop link)."""
    d = C.unit(C.sub(p_b, p_a))
    t0, t1 = C.add(p_a, C.mul(d, ROD_END_SHANK)), C.sub(p_b, C.mul(d, ROD_END_SHANK))
    body = C.tube(t0, t1, *tube_r)
    return body, [rod_end(p_a, bolt_a, d), rod_end(p_b, bolt_b, C.mul(d, -1.0))]


# ---------------- inboard: rocker, damper, ARB ----------------
def rocker(h):
    ax = C.unit(C.sub(h["RKR_A"], h["RKR_P"]))
    piv = h["RKR_P"]
    half = 13.0
    plates = []
    for sg in (1.0, -1.0):
        o = C.mul(ax, sg * half)
        for k, rr in (("PR_I", 11.0), ("DMP_R", 11.0), ("ARB_R", 9.0)):
            plates.append(C.plate(C.add(piv, o), C.add(h[k], o), rr, 5.0, ax))
        plates.append(C.plate(C.add(h["PR_I"], o), C.add(h["DMP_R"], o), 11.0, 5.0, ax))
    boss = C.tube(C.add(piv, C.mul(ax, half + 2.5)), C.sub(piv, C.mul(ax, half + 2.5)), 16.0, 11.0)
    pins = [C.cyl(C.add(h[k], C.mul(ax, half + 2.5)), C.sub(h[k], C.mul(ax, half + 2.5)), 3.9) for k in ("PR_I", "DMP_R", "ARB_R")]  # M8 bolts
    return C.union(plates + [boss] + pins)


def damper(h):
    """Ohlins TTX25 envelope: body + spring hang from the chassis eye; shaft from the rocker eye."""
    c, r_ = h["DMP_C"], h["DMP_R"]
    d = C.unit(C.sub(r_, c))
    ax = C.unit(C.sub(h["RKR_A"], h["RKR_P"]))
    eye_c = C.ring(c, ax, 9.0, 4.0, 8.0)
    body = C.cyl(C.add(c, C.mul(d, 9.0)), C.add(c, C.mul(d, 112.0)), 20.0)
    spring = C.tube(C.add(c, C.mul(d, 40.0)), C.add(c, C.mul(d, 108.0)), 28.0, 22.0)
    perch_lo = C.cyl(C.add(c, C.mul(d, 36.0)), C.add(c, C.mul(d, 42.0)), 28.0)  # lower spring perch joins spring and body
    body_part = C.union([eye_c, C.cyl(C.add(c, C.mul(d, 8.5)), C.add(c, C.mul(d, 12.0)), 7.0), body, perch_lo, spring])
    eye_r = C.ring(r_, ax, 9.0, 4.0, 8.0)
    shaft = C.cyl(C.sub(r_, C.mul(d, 9.0)), C.sub(r_, C.mul(d, 95.0)), 6.0)
    perch = C.cyl(C.sub(r_, C.mul(d, 40.0)), C.sub(r_, C.mul(d, 46.0)), 28.0)
    return body_part, C.union([eye_r, C.cyl(C.sub(r_, C.mul(d, 8.5)), C.sub(r_, C.mul(d, 12.0)), 7.0), shaft, perch])


def arb_lever(h):
    """The lever (blade) on the bar, ARB_LEVER_OFFSET inboard of the drop link's ball so the rod end sits beside it
    on its spacer, bolted through the blade along the bar's axis (R-08: single shear, 24 mm washer)."""
    ax = C.unit(C.sub(h["ARB_A"], h["ARB_P"]))
    off = C.mul(ax, -ARB_LEVER_OFFSET)
    p, l_ = C.add(h["ARB_P"], off), C.add(h["ARB_L"], off)
    bore = ARB_END_R + 0.1                                     # 0.1 mm clear of the bar's end (splined in reality)
    hub_ = C.tube(C.add(p, C.mul(ax, ARB_HUB_HALF)), C.sub(p, C.mul(ax, ARB_HUB_HALF)), bore + ARB_HUB_WALL, bore)
    blade = C.plate(p, l_, ARB_BLADE_W / 2, ARB_BLADE_T, ax)
    lever = C.union([hub_, blade])
    lever = C.cut(lever, C.cyl(C.add(p, C.mul(ax, 20)), C.sub(p, C.mul(ax, 20)), bore))  # bar bore
    return C.cut(lever, C.cyl(C.add(l_, C.mul(ax, 5)), C.sub(l_, C.mul(ax, 5)), 4.05))   # M8 hole for the drop link


def arb_bar(hp):
    """The fitted solid torsion bar (D26): its body between the lever hubs, and the splined ends (upset) from 5 mm
    inboard of each hub to the chassis bearing ARB_BAR_STUB outboard of ARB_P (left hardpoints; the right is the
    mirror)."""
    ax = C.unit(C.sub(hp["ARB_A"], hp["ARB_P"]))
    m = lambda q: (q[0], -q[1], q[2])                          # noqa: E731
    e_in = C.sub(hp["ARB_P"], C.mul(ax, ARB_LEVER_OFFSET + ARB_HUB_HALF + ARB_RUNOUT))
    e_out = C.add(hp["ARB_P"], C.mul(ax, ARB_BAR_STUB))
    body_a = C.add(e_in, C.mul(ax, 1.0))                       # 1 mm into each end: one solid
    return C.union([C.cyl(body_a, m(body_a), ARB_BAR_R), C.cyl(e_in, e_out, ARB_END_R), C.cyl(m(e_in), m(e_out), ARB_END_R)])


def prim_body(p):
    """A chassis_tabs primitive as a body: a plate (with its bolt hole) or a tube."""
    if p["kind"] == "plate":
        body = C.plate(p["p1"], p["p2"], p["r"], p["t"], p["normal"])
        if p.get("hole_r"):
            n = C.mul(C.unit(p["normal"]), p["t"])
            body = C.cut(body, C.cyl(C.sub(p["p1"], n), C.add(p["p1"], n), p["hole_r"]))
        return body
    return C.tube(p["a"], p["b"], p["ro"], p["ri"])


def rack_bar(hp):
    """Rack bar envelope between the inner tie-rod joints; it slides with the rack (poses.py rack_bar)."""
    tl = hp["TR_I"]
    end = (tl[0], tl[1] - (ROD_END_HEAD_R + 1.0), tl[2])      # 1 mm short of the inner rod-end head
    return C.cyl(end, (end[0], -end[1], end[2]), 9.0)
