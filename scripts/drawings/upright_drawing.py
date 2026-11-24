"""GD&T drawing of the upright (FS1-UPR-001), generated from the same STEP the CAD and FEA use.

Datum reference frame: A = the common axis of the two Ø68 cup seats; B = the inboard face of the central
rib; C = the lower ball-joint hole (lower ear). Origin at A and B; +Z outboard along A; -Y through C.
Views (third angle, A1): FRONT from outboard (along -Z), RIGHT (along -X), SECTION A-A on the plane X = 0
seen along -X, all 1:1; DETAIL D (the rib, 4:1) and an isometric (1:2). The compound-angle holes are
given in a hole table (basic coordinates and axis directions in the A|B|C frame), as is usual for a
3+2-axis machined part; surfaces without a dimension are held by a general profile to the 3D model.
Run: python upright_drawing.py   Writes drawings/FS1-UPR-001.{dxf,pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sheet as S  # noqa: E402
import views as V  # noqa: E402
from common import DATA, ROOT  # noqa: E402

OUT = ROOT / "drawings"
NUMBER = "FS1-UPR-001"
FV = np.array([185.0, 345.0])      # front: right = +X, up = +Y
RV = np.array([400.0, 345.0])      # right: right = -Z, up = +Y
SV = np.array([565.0, 345.0])      # section A-A: right = -Z, up = +Y
IV = np.array([160.0, 118.0])      # isometric, 1:2
DV = np.array([420.0, 118.0])      # detail D, 4:1
TX = 640.0                         # left edge of the table / notes column


def fastener_hold():
    """True unless the fastener sweep (S-18) ran on the current inputs with no FAIL and both H1 inserts in
    solid material (fasteners.sweep_clear, which treats a missing or old result as not clear)."""
    import fasteners
    clear, why = fasteners.sweep_clear()
    if not clear:
        print(f"drawing HOLD: {why}")
    return not clear


def frame_of(build):
    f = build["faces"]
    s, wc = np.array(f["spin"]), np.array(f["wc"])
    H = {h["name"]: h for h in f["holes"]}
    O = wc + s * f["rib"][0]
    c = np.array(H["LBJ2"]["centre"]) - O
    c -= s * c.dot(s)
    Y = -c / np.linalg.norm(c)
    Z = s
    X = np.cross(Y, Z)
    return O, X, Y, Z


def to_drf(p, O, X, Y, Z):
    d = np.asarray(p, float) - O
    return np.array([d @ X, d @ Y, d @ Z])


def dir_drf(a, X, Y, Z):
    a = np.asarray(a, float)
    return np.array([a @ X, a @ Y, a @ Z])


def draw_lines(msp, segs, origin, layer="OUTLINE", k=1.0):
    for pl in V.chain([np.asarray(s) for s in segs]):
        S.pline(msp, pl * k + origin, layer)


def section_regions(mesh, polys, kept_side, to3d):
    """Material regions of the cut face: the faces formed by the section linework whose interior is in the solid."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    faces = list(polygonize(unary_union([LineString(p) for p in polys if len(p) > 2])))
    keep = []
    for f in faces:
        rp = np.array(f.representative_point().coords[0])
        if mesh.contains([to3d(rp) + kept_side * 0.05])[0]:
            keep.append(f)
    return keep


def wall(mesh, p, direction):
    """First surface point of `mesh` from p along `direction` (p inside a hole: the hole wall)."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    loc, _, _ = mesh.ray.intersects_location([p], [d])
    if not len(loc):
        raise ValueError(f"no wall from {p} along {d}")
    return loc[np.argmin(np.linalg.norm(loc - p, axis=1))]


def detail(msp, segs, regions, c, r, k, at):
    """Detail view: section linework and hatching inside the circle (c, r), scaled k, centred at `at`."""
    from shapely.geometry import LineString, Point, Polygon
    disk = Point(c).buffer(r, 128)
    for pl in V.chain([np.asarray(s) for s in segs]):
        g = LineString(pl).intersection(disk)
        for part in getattr(g, "geoms", [g]):
            if isinstance(part, LineString) and part.length > 1e-6:
                S.pline(msp, (np.array(part.coords) - c) * k + at)
    for reg in regions:
        g = reg.intersection(disk)
        for part in getattr(g, "geoms", [g]):
            if isinstance(part, Polygon) and part.area > 1e-6:
                S.hatch_polygon(msp, (np.array(part.exterior.coords) - c) * k + at,
                                [(np.array(h.coords) - c) * k + at for h in part.interiors])
    S.circle(msp, at, r * k, "THIN")


def fcf_stack(msp, x, y, frames, h=3.0, right=False):
    """FCFs stacked downwards from the top-left (or top-right if right=True) corner (x, y). Returns the bottom y."""
    for kind, tol, datums in frames:
        y -= 2.0 * h
        x0 = x - S.fcf_width(tol, datums, h) if right else x
        S.fcf(msp, x0, y, kind, tol, datums, h)
    return y


def main():
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    veh = json.loads((DATA / "vehicle.json").read_text())
    ud = json.loads((DATA / "upright_design.json").read_text())
    ins, boss = ud["ball_joints"]["lower"]["insert"], ud["clevis"]["lbj_insert_boss"]
    ins = json.loads((DATA / "fasteners.json").read_text())["inserts"][ins].get("locking_version", ins)   # R-09
    hold = fastener_hold()
    ubj_d = 2 * {h['name']: h for h in build['faces']['holes']}['UBJ1']['radius']
    Hh = {h['name']: h for h in build['faces']['holes']}
    gap_u = float(np.linalg.norm(np.subtract(Hh['UBJ1']['centre'], Hh['UBJ2']['centre']))) - 2 * Hh['UBJ1']['half_len']
    gap_l = float(np.linalg.norm(np.subtract(Hh['LBJ1']['centre'], Hh['LBJ2']['centre']))) - 2 * Hh['LBJ1']['half_len']
    f = build["faces"]
    O, X, Y, Z = frame_of(build)
    rib0 = f["rib"][0]
    Zb = lambda ax: ax - rib0                    # noqa: E731  axial coordinate (from the wheel centre) -> Z from B
    zr = lambda z: -z                            # noqa: E731  Z -> view right coordinate in the right / section views
    mesh = V.to_frame(V.mesh_solid(ROOT / "cad" / "upright" / "upright_L.step"), O, X, Y, Z)
    if not mesh.is_watertight:
        print("warning: drawing mesh is not watertight; section shading may be incomplete")
    H = {h["name"]: h for h in f["holes"]}
    P = {n: to_drf(H[n]["centre"], O, X, Y, Z) for n in H}
    A = {n: dir_drf(H[n]["axis"], X, Y, Z) for n in H}
    doc, msp = S.new_doc()
    S.frame(msp)

    # ---- views
    vis_f, _ = V.view_lines(mesh, [0, 0, -1], [0, 1, 0])
    vis_r, _ = V.view_lines(mesh, [-1, 0, 0], [0, 1, 0])
    vis_s, _, polys = V.section(mesh, [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0])
    d_iso = -np.array([0.75, 0.55, 1.0])
    vis_i, _ = V.view_lines(mesh, d_iso, [0, 1, 0])
    draw_lines(msp, vis_f, FV)
    draw_lines(msp, vis_r, RV)
    draw_lines(msp, vis_s, SV)
    pts = vis_i.reshape(-1, 2)
    draw_lines(msp, vis_i, IV - 0.5 * (pts.min(0) + pts.max(0)) / 2, k=0.5)
    regions = section_regions(mesh, polys, np.array([-1.0, 0, 0]), lambda q: np.array([0.0, q[1], -q[0]]))
    for reg in regions:
        S.hatch_polygon(msp, np.array(reg.exterior.coords) + SV, [np.array(h.coords) + SV for h in reg.interiors])
    ylo, yhi = mesh.bounds[:, 1]
    ty = FV[1] + ylo - 26
    S.view_title(msp, "FRONT (FROM OUTBOARD)", "SCALE 1:1", FV[0], ty)
    S.view_title(msp, "RIGHT", "SCALE 1:1", RV[0], ty)
    S.view_title(msp, "SECTION A-A", "SCALE 1:1", SV[0], ty)
    S.view_title(msp, "ISOMETRIC", "SCALE 1:2", IV[0], 38.0)
    S.view_title(msp, "DETAIL D", "SCALE 4:1", DV[0], 38.0)

    # ---- centre lines, centre marks, cutting plane
    seat_r, rib_r = f["seat_r"], f["rib_r"]
    a0, a1 = build["axial"]
    for org in (RV, SV):
        S.line(msp, org + [zr(Zb(a1)) - 6, 0], org + [zr(Zb(a0)) + 6, 0], "CENTER")
    S.centre_mark(msp, FV, 8.0)
    for n in ("bolt1", "bolt2"):
        S.centre_mark(msp, FV + P[n][:2], 6.0)
    S.section_marks(msp, FV + [0, yhi + 8], FV + [0, ylo - 8], "A", (-1, 0))

    # ---- section A-A: seats (datum A), housing lengths from B, lower ball-joint holes (datum C), detail D
    tip = SV + [zr(Zb(a1)) + 9, seat_r]
    lab = "2X Ø68 P6 (-0.026/-0.045)"
    end = S.leader(msp, tip, SV + [zr(Zb(a1)) - 18, seat_r + 30], lab)
    x_l = end[0] - 1.0 - S.text_width(lab)
    yb = fcf_stack(msp, x_l, end[1] - 3.0, [("cyl", "0.005", ()), ("runout", "0.01", ("A",))])
    S.datum_feature(msp, "A", (x_l + 8.0, yb), (0, -1), length=4)
    S.surface_finish(msp, SV + [zr(Zb(a1)) + 3, -seat_r], "Ra 0.8")
    yd = 18.0
    S.dim_lin(msp, SV + [zr(Zb(a1)), seat_r], SV + [zr(0.0), rib_r], SV + [0, yd], txt=f"{Zb(a1):.0f}")
    S.dim_lin(msp, SV + [zr(0.0), rib_r], SV + [zr(Zb(a0)), seat_r], SV + [0, yd], txt=f"{-Zb(a0):.0f}")
    dc, dr = np.array([zr(1.0), 35.0]), 9.0
    S.circle(msp, SV + dc, dr, "THIN")
    S.text(msp, "D", *(SV + dc + [-dr - 6.0, dr + 7.0]), 4.0, align="MIDDLE_CENTER")
    for name, knee, lab in (("LBJ2", (52.0, -104.0), "C: Ø9.60 +0.05/-0"),
                            ("LBJ1", (56.0, -28.0), f"H1: {ins} SCREW-LOCKING INSERT IN Ø{boss['dia_mm']:g} BOSS")):
        q = wall(mesh, P[name] * [0, 1, 1], [0, 0, -1])        # hole wall on the cut plane, inboard side
        end = S.leader(msp, SV + [zr(q[2]), q[1]], SV + knee, lab)
        if name == "LBJ2":
            S.datum_feature(msp, "C", (end[0] + 1.0 + S.text_width(lab) / 2, end[1] - 2.2), (0, -1), length=3)

    # ---- detail D (rib, 4:1): rib width, rib bore, datum B and its orientation to A, the outboard face to B
    k = 4.0
    detail(msp, vis_s, regions, dc, dr, k, DV)
    to_d = lambda x, y: DV + (np.array([x, y]) - dc) * k   # noqa: E731
    S.dim_lin(msp, to_d(zr(2.0), rib_r), to_d(zr(0.0), rib_r), to_d(0.0, rib_r - 4.0), txt="2 ±0.05",
              text_at=to_d(0.0, rib_r - 4.0) + [16.0, 1.5])
    S.leader(msp, to_d(zr(1.6), rib_r), to_d(zr(1.6), rib_r) + [-30.0, -6.0], "Ø62 ±0.1")
    S.leader(msp, to_d(zr(2.0), rib_r + 1.8), to_d(zr(2.0), rib_r + 1.8) + [-38.0, 16.0])
    fcf_stack(msp, to_d(zr(2.0), rib_r + 1.8)[0] - 38.0 - 4.0, to_d(zr(2.0), rib_r + 1.8)[1] + 16.0 + 3.0,
              [("par", "0.01", ("B",))], right=True)
    S.datum_feature(msp, "B", to_d(zr(0.0), rib_r + 2.2), (1, 0), length=34.0)
    bx = to_d(zr(0.0), rib_r + 2.2)[0] + 34.0 + 6.0
    fcf_stack(msp, bx + 2.0, to_d(0.0, rib_r + 2.2)[1] + 3.0, [("perp", "0.01", ("A",))])

    # ---- right view: axial position of the caliper face from B (B shown hidden)
    S.line(msp, RV + [0, -rib_r], RV + [0, rib_r], "HIDDEN")
    mount = Zb(veh["hub"]["disc_axial_mm"] - veh["brake"]["caliper"]["mount_offset_mm"])
    S.dim_lin(msp, RV + [zr(0.0), ylo - 4], RV + [zr(mount), ylo - 4], RV + [0, ylo - 12], txt=f"{abs(mount):.1f} ±0.05",
              text_at=RV + [16.0, ylo - 10.5])
    end = S.leader(msp, RV + [zr(mount), 20.0], RV + [zr(mount) - 30.0, 56.0], "CALIPER FACE")
    fcf_stack(msp, end[0], end[1] + 3.0 + 12.0, [("flat", "0.05", ()), ("perp", "0.05", ("A",))], right=True)

    # ---- front view callouts (hole tags as in the hole table)
    def on_circle(c, r_dir, knee):
        u = np.asarray(knee, float) - c
        return c + u / np.linalg.norm(u) * r_dir

    r_cal = np.linalg.norm(wall(mesh, P["bolt1"], [0, 1, 0]) - P["bolt1"])
    kn = P["bolt1"][:2] + [16, 24]
    S.leader(msp, FV + on_circle(P["bolt1"][:2], r_cal, kn), FV + kn, "H4, H5: 2X Ø8.10 THRU")
    q = P["UBJ1"]
    S.leader(msp, FV + [q[0] + 3, q[1] + 2.5], FV + [q[0] + 30, q[1] + 18], f"H6, H7: 2X Ø{ubj_d:.2f} +0.05/-0 (UPPER BALL JOINT)")
    t = P["TR1"]
    S.leader(msp, FV + [t[0] - 2, t[1] + 2.5], FV + [t[0] - 10, t[1] + 46], "H2, H3: 2X Ø8.10 +0.1/-0 (TIE ROD)")

    # ---- hole table (basic, A|B|C)
    rows = []
    hole_rows = [("C", "LBJ2", "Ø9.60 +0.05/-0 (DATUM C)", "-"),
                 ("H1", "LBJ1", f"Ø{boss['tap_drill_mm']:g} DRILL, {boss['tap']} TAP", "Ø0.05 |A|B|C|"),   # insert named in the H1 leader
                 ("H2", "TR1", "Ø8.10 +0.1/-0", "Ø0.10 |A|B|C|"), ("H3", "TR2", "Ø8.10 +0.1/-0", "Ø0.10 |A|B|C|"),
                 ("H4", "bolt1", "Ø8.10 THRU", "Ø0.10 |A|B|C|"), ("H5", "bolt2", "Ø8.10 THRU", "Ø0.10 |A|B|C|"),
                 ("H6", "UBJ1", f"Ø{ubj_d:.2f} +0.05/-0", "Ø0.05 |A|B|C|"), ("H7", "UBJ2", f"Ø{ubj_d:.2f} +0.05/-0", "Ø0.05 |A|B|C|")]
    for tag, key, size, tol in hole_rows:
        p, a = P[key], A[key]
        a = a if a[np.argmax(np.abs(a))] > 0 else -a
        rows.append([tag, size, f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}", f"{a[0]:.4f} {a[1]:.4f} {a[2]:.4f}", tol])
    S.text(msp, "HOLE TABLE (BASIC, DATUM FRAME A|B|C; POINT = AXIS AT THE EAR OR PLATE MID-PLANE)", TX, 573.0, 2.6)
    yb_t = S.table(msp, TX, 569.0, ["", "SIZE", "X", "Y", "Z", "AXIS i j k", "POSITION"], rows,
                   [9, 56, 16, 16, 16, 40, 30], rh=6.0, h=2.5)

    # ---- notes
    m = build["masses_kg"]["upright"]
    cw = veh["hub"]["cup_width_mm"]
    notes = [
        "NOTES",
        "1. MATERIAL: 7075-T7351 ALUMINIUM PLATE (AMS 4078); ROLLING DIRECTION ALONG Y.",
        "2. DIMENSIONS IN MM. INTERPRET PER ASME Y14.5-2018. SURFACES WITHOUT A",
        "   DIMENSION: PROFILE 0.5 |A|B|C| TO THE 3D MODEL (upright_L.step, MODEL REV B).",
        "3. DATUMS: A = COMMON AXIS OF THE TWO Ø68 CUP SEATS; B = INBOARD FACE OF THE",
        "   CENTRAL RIB; C = LOWER BALL-JOINT HOLE. ORIGIN A/B, +Z OUTBOARD, -Y THROUGH C.",
        "4. COMPOUND-ANGLE HOLES: SEE HOLE TABLE; MACHINE 3+2 AXIS.",
        "5. FILLETS: R8 CALIPER-LUG AND LOWER-WEB ROOTS, R6 POST ROOT, R3 UPPER-CLEVIS",
        "   EAR ROOTS, R2 ALL OTHER EAR ROOTS (INCL. TWO NOT MODELLED IN CAD), R0.5",
        "   ELSEWHERE. BREAK SHARP EDGES 0.3.",
        "6. ROUGHNESS: CUP SEATS Ra 0.8; RIB, CALIPER AND CLEVIS FACES Ra 1.6; ELSE Ra 3.2.",
        "7. FINISH: SULFURIC ANODISE MIL-PRF-8625 TYPE II CLASS 1; MASK SEATS, RIB FACES",
        "   AND THREADS. FIT INSERTS AFTER ANODISE.",
        f"8. CUPS (SKF 32008 X, {cw:g} WIDE) PRESSED TO EACH RIB FACE WITH LOCTITE 641;",
        "   PRELOAD SET BY THE CONE SPACER, GROUND TO SUIT THE MEASURED RIB AT ASSEMBLY.",
        f"9. MASS {m:.2f} kg. LEFT HAND SHOWN; RIGHT HAND OPPOSITE (FS1-UPR-002).",
        "10. INDEPENDENT STUDENT DESIGN: NOT RELEASED FOR MANUFACTURE WITHOUT REVIEW.",
        f"11. CLEVIS GAPS BETWEEN EAR FACES: UPPER {gap_u:.1f} +0.2/-0, LOWER {gap_l:.1f} +0.2/-0 (BALL JOINT + SPACERS).",
    ] + (["12. HOLD: BOLT FIT, TOOL ACCESS AND THE H1 INSERT NOT YET CLEARED BY THE", "    FASTENER SWEEP (S-18)."]
         if hold else [])
    yn = S.lines_block(msp, notes, TX, yb_t - 9, h=2.8, gap=1.6)
    S.third_angle_symbol(msp, TX + 4, yn - 8.0)
    S.text(msp, "THIRD ANGLE PROJECTION", TX + 26, yn - 8.0, 2.6, align="MIDDLE_LEFT")
    S.title_block(msp, {"title": ["UPRIGHT, FRONT, LEFT", "FSAE-CLASS FRONT SUSPENSION"], "number": NUMBER, "rev": "C",
                        "fields": [("MATERIAL", "7075-T7351"), ("MASS", f"{m:.2f} kg"),
                                   ("SCALE", "1:1 UNLESS SHOWN"), ("SHEET", "A1  1 OF 1"), ("UNITS", "mm"),
                                   ("PROJECTION", "THIRD ANGLE"), ("DRAWN", "N. ROBSON"), ("DATE", "2026-11-08"),
                                   ("ROLE", "MECHANICAL DESIGNER (STUDENT)"), ("STATUS", "NOT RELEASED")],
                        "revisions": [("A", "2026-10-27", "FIRST ISSUE: UPRIGHT V3 (FEA-CHECKED)"),
                                      ("B", "2026-11-03", "V4: H1 INSERT BOSS; UPPER CLEVIS ONE PIECE (S-18)"),
                                      ("C", "2026-11-08", "H1 INSERT SCREW-LOCKING 3591-6CN375 (R-09)")]})
    OUT.mkdir(exist_ok=True)
    S.save(doc, str(OUT / NUMBER), title=f"{NUMBER} Upright, front, left", subject="FSAE-class front suspension")
    print(f"wrote {OUT / NUMBER}.dxf / .pdf / .png; {len(rows)} holes in the table, {len(regions)} section regions")


if __name__ == "__main__":
    main()
