"""Upright FEA: every load case, a mesh-convergence study, a hand-calc validation, stiffness and fatigue.

Geometry: cad/upright/upright_L_fea.step (upright_geom.py; one solid since v4: the upper clevis is part of the post).
Loads (data/upright_loads.json, already in the upright's frame, self-equilibrated to 1e-14):
  ball joints and tie rod   pin-bearing (cosine) pressure in the two ear holes of each clevis, half each,
                            over the 6 mm next to the joint (the lower clevis's thickened upper ear included)
  caliper bolts             pin-bearing pressure in each lug hole
  bearing rows              cosine pressure on each cup seat, tilted towards the central rib by
                            atan(pressure-centre offset / seat radius) as a tapered roller's contact force
                            is, which puts the row's line of action on its pressure centre (as the free
                            body assumed); the patch is turned until its resultant lies on the load
  bearing thrust            the row thrust left over, uniform on the face of the central rib the cups bear on
Every face a load case can load (the ear and lug holes, both cup seats, and both faces of the rib, since which
one the thrust bears on depends on the case) has its area checked against its drawing value on the base mesh
before the results are used (a wrong selection stops the run).
Support: a minimal 3-2-1 set of points on the housing. The applied loads must balance: the run stops if
their resultant force exceeds 0.1 % of the largest load or their resultant moment about the wheel centre
exceeds 1 % of (largest load x 100 mm). Discretising the curved load patches leaves a little; a wrong face
or sign leaves a residual as large as the moments themselves. The support reactions are reported.
Strength: von Mises at the quadrature points (raw P2 stress, no smoothing) vs the 7075-T7351 design
yield, per region (each arm's root, the housing, the clevises) and globally. Kept out of the peaks, and
reported on their own: points within 8 mm of the held vertices, within 1.5 mm of a pin or bolt hole
(pin contact; the holes get a nominal bearing check P / (d t) against 1.4 Fty instead), and within 2 mm
of a re-entrant corner the CAD kernel could not fillet (a sharp corner's FEA stress is singular and
grows without limit as the mesh is refined; those roots get a hand calc).
Validation (S-06c): the method (mesh, pin-bearing load, integration-point stress) against beam theory on a
slender pin-loaded bar (benchmark_lug.py, run first and recorded). The upright itself has no slender
section: its steering arm (20 mm long, 32 mm wide) is compared with beam theory for information only.
Stiffness: rotation of the bearing bore relative to the three joints (least-squares rigid fits) per kN.
Fatigue (S-07a): Goodman, amplitude = von Mises of half the stress-tensor range, mean = signed von
Mises of the mean tensor, over LC2 <-> LC2i and LC3 <-> static (taken as LC1 / 3).
Run: python upright_fea.py   Writes data/upright_fea.json, notes/upright_fea.md, plots/fea01_*.png
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DATA, NOTES, PLOTS, ROOT, figure, read_version, style_axes, SERIES, INK_2, unit  # noqa: E402
import fea  # noqa: E402
import benchmark_lug  # noqa: E402
import fingerprint  # noqa: E402
from fea import Model, load_mesh, mesh_step  # noqa: E402

STEP = ROOT / "cad" / "upright" / "upright_L_fea.step"
WORK = ROOT / ".tmp" / "fea"
SUPPORT_EXCLUDE_MM = 8.0
HOLE_EDGE_MM = 1.5
SHARP_EXCLUDE_MM = 2.0
MOMENT_LIMIT = 0.01              # residual applied moment / (largest load x MOMENT_ARM)
MOMENT_ARM = 100.0               # mm, the scale of the upright's load levers
AREA_TOL = 0.05                  # selected face area vs its drawing value
FOS = 1.5
BEARING_FACTOR = 1.4             # bearing yield Fbry ~ 1.4 Fty for 7075 lugs at e/D 1.5 (assumed; MMPDS gives e/D-dependent values)
LEVELS = [(6.0, 2.8), (4.0, 2.0), (4.0, 1.4)]      # (global, local) element sizes [mm]; a 4th level (~1.6 M DOFs)
BASE = 2                                           # does not fit in memory on a 16 GB laptop. BASE: all load cases
PRE = (6.0, 2.4)                                   # pre-pass mesh that finds the hot spots
HOT_FRACTION, HOT_RADIUS, HOT_MAX = 0.4, 10.0, 16  # refine round pre-pass points above 40 % of the worst peak
FATIGUE_CASES = ("LC1", "LC2", "LC2i", "LC3")
# Level 1 was (5.0, 2.0): gmsh's surface mesh self-intersects at a 2 mm ear-root fillet for any global size
# from 4.5 to 5.5 mm, whatever the local size, so level 1 uses 4.0 mm (its local size, which sets the
# peaks, is unchanged).
STAGE_VERSION = 2      # bump when a stage's computation changes: stage checkpoints are keyed on it
# v2: each cup loaded over its own 14.5 mm next to the rib, not the bearing's 19 mm overall width (audit 2026-10-27)


def setup():
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    upl = json.loads((DATA / "upright_loads.json").read_text())
    veh = json.loads((DATA / "vehicle.json").read_text())
    mat = json.loads((DATA / "materials.json").read_text())["7075_T7351_plate"]
    return build["faces"], build, upl, veh, mat


def regions(faces, build):
    """Named regions where peak stresses are tracked: (centre, radius) balls round each arm's root and
    clevis, sized from the design (half-width + fillet + margin), and the housing (every point closer
    than R_h + 1 mm to the spin axis: the ring, its bore and the first millimetre of the root fillets),
    plus refinement points."""
    s, wc = np.array(faces["spin"]), np.array(faces["wc"])
    R_h = build["R_h"]
    ud = json.loads((DATA / "upright_design.json").read_text())
    fil = ud["fillets"]["load_root_mm"]
    H = {h["name"]: h for h in faces["holes"]}

    def root_toward(p, axial):
        v = np.asarray(p) - wc
        v = v - s * v.dot(s)
        return wc + s * axial + unit(v) * R_h

    post_ax = np.mean(build["post_axial"])
    web_ax = np.mean(build["web_axial"])
    ubj = np.mean([H["UBJ1"]["centre"], H["UBJ2"]["centre"]], axis=0)
    lbj = np.mean([H["LBJ1"]["centre"], H["LBJ2"]["centre"]], axis=0)
    tro = np.mean([H["TR1"]["centre"], H["TR2"]["centre"]], axis=0)
    bolts = np.mean([H["bolt1"]["centre"], H["bolt2"]["centre"]], axis=0)
    cm = ud["caliper_mount"]
    plate_half = cm["root_half_spacing_mm"] + cm["lug_r_mm"] + fil
    reg = {"post root": (root_toward(ubj, post_ax), ud["post"]["half_width_mm"] + fil + 4),
           "lower web root": (root_toward(lbj, web_ax), ud["lower_web"]["half_width_mm"] + fil + 4),
           "caliper plate root": (root_toward(bolts, np.dot(bolts - wc, s)), math.hypot(plate_half, cm["lug_t_mm"] / 2) + 4),
           "steering arm root": (root_toward(tro, np.dot(tro - wc, s)), 18.0),
           "housing": ("shell", R_h + 1.0),
           "upper clevis": (ubj, 26.0), "lower clevis": (lbj, 28.0), "steering clevis": (tro, 20.0)}
    refine = [[np.array(c) for c, _ in reg.values() if not isinstance(c, str)] + [np.array(h["centre"]) for h in faces["holes"]]]
    return reg, refine


def region_mask(X, region, faces):
    """Points (3 x ...) inside a region: a ball, or the housing shell (radius from the spin axis below R)."""
    c, r = region
    shape = (3,) + (1,) * (X.ndim - 1)
    if isinstance(c, str):
        s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
        d = X - wc.reshape(shape)
        ax = np.einsum("i,i...->...", s, d)
        rad = np.linalg.norm(d - s.reshape(shape) * ax, axis=0)
        return rad < r
    return np.linalg.norm(X - np.asarray(c).reshape(shape), axis=0) < r


def build_model(level, faces, build, mat, reg, hot=None):
    """Mesh at LEVELS[level] (or PRE when level is "pre"): local size round the regions and holes (16 mm
    balls) and round the pre-pass hot spots (HOT_RADIUS balls)."""
    g, l = PRE if level == "pre" else LEVELS[level]
    WORK.mkdir(parents=True, exist_ok=True)
    pts = [np.array(c) for c, _ in reg.values() if not isinstance(c, str)] + [np.array(h["centre"]) for h in faces["holes"]]
    refine = [(pts, 16.0, l)] + ([(hot, HOT_RADIUS, l)] if hot else [])
    msh = mesh_step(STEP, g, WORK / f"upright_{level}.msh", refine=refine)
    m = Model(load_mesh(msh), mat["E_MPa"], mat["nu"])
    m.quality = dict(fea.QUALITY)
    return m


def hole_sel(h):
    c, a, r, hl = np.array(h["centre"]), unit(np.array(h["axis"])), h["radius"], h["half_len"]

    def where(x):
        d = x - c[:, None]
        ax = np.einsum("i,i...->...", a, d)
        rad = np.linalg.norm(d - a[:, None] * ax, axis=0)
        # facet centroids sit inside a curved surface by up to ~0.4 mm on coarse meshes; the ear faces lie
        # exactly at |ax| = hl, so a strict axial bound keeps them out while the radius tolerance is generous
        return (np.abs(rad - r) < 0.6) & (np.abs(ax) < hl - 0.01)
    return where


def _check_area(m, where, expected, what):
    got = m.area(where)
    if abs(got / expected - 1) > AREA_TOL:
        raise ValueError(f"{what}: selected face area {got:.1f} mm2, expected {expected:.1f} mm2")


def apply_case(m, case, faces, veh, check=False):
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    H = {h["name"]: h for h in faces["holes"]}
    for joint, names in (("UBJ", ("UBJ1", "UBJ2")), ("LBJ", ("LBJ1", "LBJ2")), ("TR", ("TR1", "TR2"))):
        F = np.array(case[joint]["force_N"]) / 2
        for n in names:
            if check:
                _check_area(m, hole_sel(H[n]), 2 * math.pi * H[n]["radius"] * 2 * H[n]["half_len"], f"hole {n}")
            m.bearing(hole_sel(H[n]), H[n]["axis"], H[n]["centre"], F)
    for n in ("bolt1", "bolt2"):
        if check:
            _check_area(m, hole_sel(H[n]), 2 * math.pi * H[n]["radius"] * 2 * H[n]["half_len"], f"hole {n}")
        m.bearing(hole_sel(H[n]), H[n]["axis"], H[n]["centre"], case[n]["force_N"])
    hub = veh["hub"]
    seat_r, rib_r = faces["seat_r"], faces["rib_r"]
    # each cup: cosine pressure, uniform over the cup's own width next to the rib (the overall bearing width
    # includes the cone standing proud of the cup), tilted towards the rib as a tapered roller's contact force
    # is; the tilt atan(distance from the cup's mid-width to the pressure centre / R) puts the row's line of
    # action on its pressure centre, the point the free body used. The tilt's axial resultant is balanced on
    # the rib with the thrust.
    seat_axial = np.zeros(3)
    rib0, rib1 = faces["rib"]
    for row, to_rib in (("row1", s), ("row2", -s)):
        k = 0 if row == "row1" else 1
        pc = hub["row_centres_axial_mm"][k] + (-1 if k == 0 else 1) * hub["pressure_centre_offset_mm"]
        lo, hi = (rib0 - hub["cup_width_mm"], rib0) if k == 0 else (rib1, rib1 + hub["cup_width_mm"])
        incline = math.degrees(math.atan(abs(pc - (lo + hi) / 2) / seat_r))

        def seat(x, lo=lo, hi=hi):
            d = x - wc[:, None]
            ax = np.einsum("i,i...->...", s, d)
            rad = np.linalg.norm(d - s[:, None] * ax, axis=0)
            return (np.abs(rad - seat_r) < 0.8) & (ax > lo + 0.01) & (ax < hi - 0.01)   # strict: not the rib faces
        if check:
            _check_area(m, seat, 2 * math.pi * seat_r * (hi - lo), f"cup seat {row}")   # the cup's own width
        seat_axial += m.bearing(seat, s, wc, case[row]["force_N"], incline_deg=incline, incline_dir=to_rib)
    Fa = np.array(case["axial"]["force_N"]) - seat_axial
    face_ax = faces["rib"][1] if Fa.dot(s) < 0 else faces["rib"][0]   # inboard push -> outboard rib face

    def rib(x, face_ax=face_ax):
        d = x - wc[:, None]
        ax = np.einsum("i,i...->...", s, d)
        rad = np.linalg.norm(d - s[:, None] * ax, axis=0)
        return (np.abs(ax - face_ax) < 0.05) & (rad > rib_r - 0.05) & (rad < seat_r + 0.05)
    if check:                          # both faces: which one the thrust bears on depends on the case
        for fa in faces["rib"]:
            _check_area(m, lambda x, fa=fa: rib(x, face_ax=fa), math.pi * (seat_r ** 2 - rib_r ** 2), f"rib face at {fa:g} mm")
    if np.linalg.norm(Fa) > 0:
        m.traction(rib, Fa)


def supports(faces, build):
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    up = unit(np.array([0.0, 0.0, 1.0]) - s * s[2])
    fwd = unit(np.cross(up, s))
    if fwd[0] < 0:
        fwd = -fwd
    R = build["R_h"]
    p1 = wc + R * unit(fwd + up)
    p2 = p1 + s * 12.0
    p3 = wc + R * unit(fwd - up)
    return [p1, p2, p3], [(0, 1, 2), (0, 2), (0,)]


def max_load(case):
    return max(np.linalg.norm(c["force_N"]) for c in case.values() if isinstance(c, dict) and "force_N" in c)


def run_case(m, case, faces, veh, sup_pts, sup_comp, method="amg", check=False):
    """Load, hold and solve one case. Stops if the applied loads do not add up to the case's own resultant
    force, or leave a moment about the wheel centre above MOMENT_LIMIT (a wrong face or sign)."""
    m.f[:] = 0.0
    apply_case(m, case, faces, veh, check=check)
    m.fixed = np.array([], dtype=np.int64)
    m.fix_points(sup_pts, sup_comp)
    applied = m.applied_resultant()
    target = sum(np.array(c["force_N"]) for c in case.values() if isinstance(c, dict) and "force_N" in c)
    Fmax = max_load(case)
    if np.linalg.norm(applied - target) > 1e-3 * Fmax:
        raise ValueError(f"applied loads add up to {applied.round(2)} N, the case to {target.round(2)} N")
    wc = np.array(faces["wc"])
    xyz = m.sb.doflocs.T - wc
    f = np.stack([m.f[m.comp[i]] for i in range(3)], axis=1)
    m.moment_ratio = float(np.linalg.norm(np.cross(xyz, f).sum(axis=0)) / (Fmax * MOMENT_ARM))
    if m.moment_ratio > MOMENT_LIMIT:
        raise ValueError(f"applied loads leave a moment of {m.moment_ratio:.1%} of (largest load x {MOMENT_ARM:g} mm): "
                         "they are not in balance")
    m.solve(method)
    react = m.support_reactions()
    sig, X = m.qp_stress()
    return sig, X, applied, react


def masks(X, faces, sup_xyz, sharp):
    """(near a held vertex, on a hole edge, near a sharp corner) boolean masks over the quadrature points."""
    near_sup = np.zeros(X.shape[1:], bool)
    for p in sup_xyz:
        near_sup |= np.linalg.norm(X - np.asarray(p)[:, None, None], axis=0) < SUPPORT_EXCLUDE_MM
    on_hole = np.zeros(X.shape[1:], bool)
    for h in faces["holes"]:
        c, a = np.array(h["centre"]), unit(np.array(h["axis"]))
        d = X - c[:, None, None]
        ax = np.einsum("i,i...->...", a, d)
        rad = np.linalg.norm(d - a[:, None, None] * ax, axis=0)
        on_hole |= (rad < h["radius"] + HOLE_EDGE_MM) & (np.abs(ax) <= h["half_len"] + 0.5)
    near_sharp = np.zeros(X.shape[1:], bool)
    if len(sharp):
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(sharp).query(X.reshape(3, -1).T)
        near_sharp = (dist < SHARP_EXCLUDE_MM).reshape(X.shape[1:])
    return near_sup, on_hole, near_sharp


def peaks(sig, X, reg, faces, sup_xyz, sharp):
    vm = Model.von_mises_of(sig)
    near_sup, on_hole, near_sharp = masks(X, faces, sup_xyz, sharp)
    body = ~near_sup & ~on_hole & ~near_sharp
    out = {}
    for name, region in reg.items():
        mask = region_mask(X, region, faces) & body
        if not mask.any():
            raise ValueError(f"region {name!r} contains no integration points: check its centre and radius")
        out[name] = float(vm[mask].max())
    i = np.unravel_index(np.argmax(np.where(body, vm, -1)), vm.shape)
    out["_global"] = float(vm[i])
    out["_global_at"] = X[:, i[0], i[1]].round(1).tolist()
    out["_hole_edges"] = float(vm[on_hole & ~near_sup].max()) if (on_hole & ~near_sup).any() else 0.0
    out["_sharp_corners"] = float(vm[near_sharp & ~near_sup].max()) if (near_sharp & ~near_sup).any() else 0.0
    return out, vm, body


def hole_bearing(case, faces):
    """Nominal bearing stress P / (d t) at each loaded hole [MPa] (the ball-joint and tie-rod loads split
    evenly between the two ears)."""
    out = {}
    for h in faces["holes"]:
        F = np.array(case[h["joint"]]["force_N"]) / (2 if h["joint"] in ("UBJ", "LBJ", "TR") else 1)
        a = unit(np.array(h["axis"]))
        Fn = np.linalg.norm(F - a * F.dot(a))
        out[h["name"]] = Fn / (2 * h["radius"] * 2 * h["half_len"])
    return out


def hot_spots(faces, build, upl, veh, mat, reg, sup_pts, sup_comp, sharp):
    """Solution-adaptive refinement: solve every case on the pre-pass mesh and return the hottest points
    (clustered 8 mm apart, above HOT_FRACTION of the worst peak), where the finer meshes are refined too.
    Cached in .tmp/fea against the geometry, the loads and the pre-pass settings."""
    import hashlib
    h = hashlib.sha256()
    for path in (STEP, ROOT / "cad" / "upright" / "upright_build.json", DATA / "upright_loads.json"):
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        h.update(raw[raw.find(b"DATA;"):] if path.suffix == ".step" else raw)
    h.update(repr((PRE, HOT_FRACTION, HOT_MAX, SUPPORT_EXCLUDE_MM, HOLE_EDGE_MM, SHARP_EXCLUDE_MM, STAGE_VERSION)).encode())
    h.update((DATA / "vehicle.json").read_bytes())
    key = h.hexdigest()[:16]
    cache = WORK / "hot_spots.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c["key"] == key:
            return [np.array(p) for p in c["hot"]], c["hot_v"]
    hot, hot_v = _hot_spots(faces, build, upl, veh, mat, reg, sup_pts, sup_comp, sharp)
    WORK.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "hot": [p.tolist() for p in hot], "hot_v": hot_v}))
    return hot, hot_v


def _hot_spots(faces, build, upl, veh, mat, reg, sup_pts, sup_comp, sharp):
    m = build_model("pre", faces, build, mat, reg)
    P_all, v_all = [], []
    for case in upl["cases"].values():
        sig, X, _, _ = run_case(m, case, faces, veh, sup_pts, sup_comp)
        _, vm, body = peaks(sig, X, reg, faces, m.support_xyz, sharp)
        v = np.where(body, vm, -1.0).ravel()
        top = np.argsort(v)[-3000:]
        P_all.append(X.reshape(3, -1).T[top])
        v_all.append(v[top])
    P, v = np.concatenate(P_all), np.concatenate(v_all)
    order = np.argsort(v)[::-1]
    picked = []
    for i in order:
        if v[i] < HOT_FRACTION * v[order[0]] or len(picked) >= HOT_MAX:
            break
        if all(np.linalg.norm(P[i] - P[j]) > 8.0 for j in picked):
            picked.append(i)
    return [P[i] for i in picked], [float(v[i]) for i in picked]


def rigid_rotation(points, disp):
    """Least-squares small rotation (rad, 3-vector) of a point set: u = t + theta x p."""
    p = points - points.mean(axis=0)
    A = np.zeros((3 * len(p), 6))
    for k, q in enumerate(p):
        A[3 * k:3 * k + 3, :3] = np.eye(3)
        A[3 * k:3 * k + 3, 3:] = -np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])
    sol, *_ = np.linalg.lstsq(A, disp.reshape(-1), rcond=None)
    return sol[3:]


def compliance(m, faces):
    """Rotation of the bearing bore relative to the three joints, as (camber about x, toe about z) [deg]."""
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    xyz = m.sb.doflocs.T
    u = np.stack([m.u[m.comp[i]] for i in range(3)], axis=1)
    d = xyz - wc
    ax = d @ s
    rad = np.linalg.norm(d - np.outer(ax, s), axis=1)
    bore = np.abs(rad - faces["seat_r"]) < 0.3
    joint = np.zeros(len(xyz), bool)
    for h in faces["holes"][:6]:
        c, a = np.array(h["centre"]), unit(np.array(h["axis"]))
        dd = xyz - c
        aa = dd @ a
        rr = np.linalg.norm(dd - np.outer(aa, a), axis=1)
        joint |= (np.abs(rr - h["radius"]) < 0.3) & (np.abs(aa) <= h["half_len"])
    rel = rigid_rotation(xyz[bore], u[bore]) - rigid_rotation(xyz[joint], u[joint])
    return math.degrees(rel[0]), math.degrees(rel[2])


def steering_section(faces, build):
    """Where to cut the steering-arm ears for the validation: halfway between the edge of the tie-rod bolt
    hole and the start of the root fillet (the plainest section the short arm has). Returns d_sec [mm]."""
    H = {h["name"]: h for h in faces["holes"]}
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    tro = np.mean([H["TR1"]["centre"], H["TR2"]["centre"]], axis=0)
    v = tro - wc
    L_root = np.linalg.norm(v - s * v.dot(s)) - build["R_h"]         # bolt to housing surface, radially
    return 0.5 * (H["TR1"]["radius"] + (L_root - json.loads((DATA / "upright_design.json").read_text())["fillets"]["min_mm"]))


def steering_validation(sig, X, faces, case, d_sec):
    """LC6: normal stress across the steering-arm ears at d_sec mm from the tie-rod bolt, FEA vs beam theory."""
    H = {h["name"]: h for h in faces["holes"]}
    ud = json.loads((DATA / "upright_design.json").read_text())
    s, wc = unit(np.array(faces["spin"])), np.array(faces["wc"])
    tro = np.mean([H["TR1"]["centre"], H["TR2"]["centre"]], axis=0)
    b = unit(np.array(H["TR1"]["axis"]))
    v = tro - wc
    eL = -unit(v - s * v.dot(s))                       # from the bolt towards the housing axis
    eL = unit(eL - b * eL.dot(b))
    eW = unit(np.cross(b, eL))
    t, h = ud["clevis"]["ear_t_mm"], 2 * ud["clevis"]["ear_r_mm"]
    F = np.array(case["TR"]["force_N"])                # the force ON the arm, at the bolt
    I = 2 * t * h ** 3 / 12
    A = 2 * t * h
    M = F.dot(eW) * d_sec                              # in-plane bending about b at the section [N mm]
    P = F.dot(eL)                                      # + towards the housing: compresses the arm
    d = X - tro[:, None, None]
    along, across = np.einsum("i,i...->...", eL, d), np.einsum("i,i...->...", eW, d)
    thick = np.abs(np.einsum("i,i...->...", b, d))
    in_ear = (np.abs(thick - np.linalg.norm(np.array(H["TR1"]["centre"]) - tro)) <= t / 2)
    slab = (np.abs(along - d_sec) < 1.2) & in_ear
    sLL = np.einsum("i,ij...,j->...", eL, sig, eL)
    res = []
    for side in (1.0, -1.0):
        band = slab & (side * across > h / 2 - 3.5) & (side * across < h / 2 - 0.5)
        w = float(across[band].mean())
        fea_v = float(sLL[band].mean())
        hand = -M * w / I - P / A
        res.append({"fibre_mm": w, "fea_MPa": fea_v, "hand_MPa": hand,
                    "error_pct": (fea_v / hand - 1) * 100 if hand else float("nan")})
    return res


def signed_vm(sg):
    """von Mises with the sign of the hydrostatic stress (tension +)."""
    return np.sign(sg[0, 0] + sg[1, 1] + sg[2, 2]) * Model.von_mises_of(sg)


def goodman(sig_a, sig_b, Se, Sut):
    """Per quadrature point, Goodman FoS for cycling between two stress states: amplitude = von Mises of
    half the tensor range (right for non-proportional loading), mean = signed von Mises of the mean tensor."""
    amp = Model.von_mises_of((sig_a - sig_b) / 2)
    mean = signed_vm((sig_a + sig_b) / 2)
    inv = amp / Se + np.maximum(mean, 0.0) / Sut
    return np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), np.inf)


def richardson(f, r):
    """Observed order and extrapolated value from three solutions f (coarse -> fine) at a constant
    refinement ratio r; GCI of the finest (Roache, safety factor 1.25). None unless the changes are monotone and
    shrink (an observed order above zero): otherwise the meshes are not in the asymptotic range and an extrapolation
    would mean nothing."""
    e21, e32 = f[1] - f[0], f[2] - f[1]
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0 or abs(e32) >= abs(e21):
        return None
    p = math.log(abs(e21 / e32)) / math.log(r)
    f_ext = f[2] + e32 / (r ** p - 1)
    return {"order": p, "extrapolated": f_ext, "gci_pct": 125 * abs(e32 / f[2]) / (r ** p - 1)}


def main():
    faces, build, upl, veh, mat = setup()
    reg, _ = regions(faces, build)
    sup_pts, sup_comp = supports(faces, build)
    sharp = np.array([p for e in build.get("sharp_edges", []) for p in e["points"]]).reshape(-1, 3)
    ver = read_version(DATA / "hardpoints.csv")
    Fty, Ftu = mat["Fty_MPa"], mat["Ftu_MPa"]
    t0 = time.time()
    inputs_sha = fingerprint.fea_inputs_hash()           # taken at the start: inputs edited mid-run make the result stale
    # each stage is checkpointed in .tmp/fea against the input fingerprint: a failure later in the run (or a
    # report-only fix) costs only that stage, never the hour already done
    data_sha = fingerprint.fea_inputs_hash(exclude=("scripts/fea/upright_fea.py",))

    def staged(name, fn, params=()):
        """Run fn once per (inputs other than this file, the stage's parameters, STAGE_VERSION): the stage is
        cached in .tmp/fea, so a report-only edit of this file does not re-run an hour of solving."""
        import hashlib
        key = hashlib.sha256(repr((data_sha, name, params, STAGE_VERSION)).encode()).hexdigest()[:16]
        path = WORK / f"stage_{name}_{key}.json"
        if path.exists():
            print(f"{name}: from checkpoint", flush=True)
            return json.loads(path.read_text())
        res = fn()
        WORK.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, default=float))
        return res
    # ---- method validation on a slender pin-loaded bar
    bench = staged("bench", benchmark_lug.run)
    print(f"method validation: {bench['max_err_pct']:.2f} % from beam theory ({time.time() - t0:.0f} s)", flush=True)
    # ---- pre-pass: where are the hot spots?
    hot, hot_v = hot_spots(faces, build, upl, veh, mat, reg, sup_pts, sup_comp, sharp)
    print(f"pre-pass: {len(hot)} hot spots, {hot_v[0]:.0f}-{hot_v[-1]:.0f} MPa ({time.time() - t0:.0f} s)", flush=True)
    d_sec = steering_section(faces, build)
    ka = 4.51 * mat["Ftu_typical_MPa"] ** -0.265
    Se = mat["fatigue_strength_5e8_MPa"] * ka * 0.9 * 0.814
    common = (np.round(np.array(hot), 3).tolist(), HOT_RADIUS, SUPPORT_EXCLUDE_MM, HOLE_EDGE_MM, SHARP_EXCLUDE_MM, MOMENT_LIMIT,
              MOMENT_ARM, AREA_TOL)
    # every face any case can load, area-checked on the base mesh: its own stage, so the check is part of every result
    # (the base stage checks on its first case only, when it is solved)
    staged("faces", lambda: face_check(faces, build, upl, veh, mat, reg, hot), (LEVELS[BASE], common))
    print(f"faces: every loadable face within {100 * AREA_TOL:.0f} % of its drawing area ({time.time() - t0:.0f} s)", flush=True)
    base = staged("base", lambda: base_stage(faces, build, upl, veh, mat, reg, hot, sup_pts, sup_comp, sharp, d_sec, Se, Ftu, t0),
                  (LEVELS[BASE], common, FATIGUE_CASES, round(Se, 6), d_sec))
    results, crit, dofs_base, mesh_base = base["results"], base["crit"], base["dofs_base"], base["mesh_base"]
    stiff, fat, fat_holes = base["stiff"], base["fat"], base["fat_holes"]
    # ---- convergence on the critical case (the base level is the run above)
    conv = []
    for lev in range(len(LEVELS)):
        if lev == BASE:
            c = {"peaks_MPa": results[crit]["peaks_MPa"], "dofs": dofs_base, "mesh": mesh_base}
        else:
            c = staged(f"conv{lev}", lambda lev=lev: conv_stage(lev, faces, build, upl, veh, mat, reg, hot, sup_pts, sup_comp,
                                                                 sharp, crit), (LEVELS[lev], common, crit))
        conv.append({"level": lev, "size_mm": LEVELS[lev], **c})
        print(f"convergence level {lev}: {c['dofs']} dofs, global {c['peaks_MPa']['_global']:.0f} MPa", flush=True)
    r_ref = (LEVELS[0][1] / LEVELS[-1][1]) ** (1 / (len(LEVELS) - 1))
    rich = richardson([c["peaks_MPa"]["_global"] for c in conv], r_ref)
    _results(results, crit, dofs_base, mesh_base, stiff, fat, fat_holes, conv, rich, r_ref, bench, d_sec, Se, hot, hot_v,
             build, reg, mat, ver, inputs_sha, Fty, t0)


def face_check(faces, build, upl, veh, mat, reg, hot):
    """The ear and lug holes, both cup seats and both rib faces, area-checked on the base mesh (no solve); raises on a
    wrong selection."""
    m = build_model(BASE, faces, build, mat, reg, hot)
    apply_case(m, next(iter(upl["cases"].values())), faces, veh, check=True)
    return {"checked": True, "dofs": int(m.K.shape[0])}


def conv_stage(lev, faces, build, upl, veh, mat, reg, hot, sup_pts, sup_comp, sharp, crit):
    mm = build_model(lev, faces, build, mat, reg, hot)
    sig, X, _, _ = run_case(mm, upl["cases"][crit], faces, veh, sup_pts, sup_comp)
    pk, _, _ = peaks(sig, X, reg, faces, mm.support_xyz, sharp)
    return {"peaks_MPa": pk, "dofs": mm.K.shape[0], "mesh": mm.quality}


def base_stage(faces, build, upl, veh, mat, reg, hot, sup_pts, sup_comp, sharp, d_sec, Se, Ftu, t0):
    """Every load case on the base mesh (stresses kept only for the fatigue pairs, to save memory), then
    stiffness and fatigue."""
    m = build_model(BASE, faces, build, mat, reg, hot)
    results, sigs = {}, {}
    for lcid, case in upl["cases"].items():
        sig, X, applied, react = run_case(m, case, faces, veh, sup_pts, sup_comp, check=(lcid == next(iter(upl["cases"]))))
        X_base, sup_base = X, m.support_xyz
        pk, vm, body = peaks(sig, X, reg, faces, m.support_xyz, sharp)
        if lcid in FATIGUE_CASES:
            sigs[lcid] = sig
        cam, toe = compliance(m, faces)
        brg = hole_bearing(case, faces)
        results[lcid] = {"peaks_MPa": pk, "applied_N": applied.round(3).tolist(), "max_reaction_N": float(np.abs(react).max()),
                         "moment_residual": m.moment_ratio,
                         "max_load_N": max_load(case), "hole_bearing_MPa": brg, "camber_deg": cam, "toe_deg": toe}
        if lcid == "LC6":
            results[lcid]["validation"] = steering_validation(sig, X, faces, case, d_sec)
        print(f"{lcid}: global {pk['_global']:.0f} MPa at {pk['_global_at']}, largest reaction {np.abs(react).max():.1f} N "
              f"({time.time() - t0:.0f} s)", flush=True)
    dofs_base, mesh_base = m.K.shape[0], m.quality
    crit = max(results, key=lambda k: results[k]["peaks_MPa"]["_global"])
    # ---- stiffness (LC2 lateral, LC3 braking) per kN of tyre force
    lcs = {r["id"]: r for r in _read_lcs()}
    stiff = {}
    for lcid, key in (("LC2", "Fy_N"), ("LC3", "Fx_N")):
        kN = abs(float(lcs[lcid][key])) / 1000
        stiff[lcid] = {"camber_deg_per_kN": results[lcid]["camber_deg"] / kN, "toe_deg_per_kN": results[lcid]["toe_deg"] / kN}
    # ---- fatigue: Goodman; endurance = fatigue strength at 5e8 cycles x Marin factors (Se, from main). ka is
    # Shigley's machined-surface fit (made for steels; commonly applied to aluminium) on the TYPICAL tensile
    # strength; the Goodman line itself uses the design (0.9 x typical) Ftu, which is conservative.
    X = X_base
    near_sup, on_hole, near_sharp = masks(X, faces, sup_base, sharp)
    body = ~near_sup & ~on_hole & ~near_sharp
    static = sigs["LC1"] / 3.0                          # LC1 (3 g bump at +30 mm) / 3 stands in for 1 g static
    fat, fat_holes = {}, {}
    for name, (sa, sb) in {"cornering LC2 <-> LC2i": (sigs["LC2"], sigs["LC2i"]), "braking LC3 <-> static": (sigs["LC3"], static)}.items():
        n = goodman(sa, sb, Se, Ftu)
        for store, mask in ((fat, body), (fat_holes, on_hole & ~near_sup)):
            nm = np.where(mask, n, np.inf)
            i = np.unravel_index(np.argmin(nm), nm.shape)
            store[name] = {"fos": float(nm[i]), "at": X[:, i[0], i[1]].round(1).tolist()}
    del sigs, static, X, X_base, body, m
    return {"results": results, "crit": crit, "dofs_base": dofs_base, "mesh_base": mesh_base, "stiff": stiff,
            "fat": fat, "fat_holes": fat_holes}


def _results(results, crit, dofs_base, mesh_base, stiff, fat, fat_holes, conv, rich, r_ref, bench, d_sec, Se, hot, hot_v,
             build, reg, mat, ver, inputs_sha, Fty, t0):
    # ---- results
    worst = {k: max(results[c]["peaks_MPa"][k] for c in results) for k in list(reg) + ["_global", "_hole_edges", "_sharp_corners"]}
    brg_max = max((v, c, h) for c in results for h, v in results[c]["hole_bearing_MPa"].items())
    out = {"_hardpoints": ver, "_inputs_sha": inputs_sha, "dofs_base": dofs_base, "mesh_base": mesh_base,
           "fillets": build["fillets"], "sharp_edges": [e["feature"] for e in build.get("sharp_edges", [])],
           "hot_spots": [{"at": np.round(p, 1).tolist(), "pre_MPa": v} for p, v in zip(hot, hot_v)], "critical_case": crit,
           "cases": results, "convergence": conv, "richardson": rich, "refinement_ratio": r_ref,
           "stiffness": stiff, "fatigue": {"Se_MPa": Se, **fat}, "fatigue_hole_edges": fat_holes, "worst_MPa": worst,
           "fos_yield": Fty / worst["_global"],
           "bearing_max_MPa": brg_max[0], "bearing_max_at": f"{brg_max[2]} in {brg_max[1]}",
           "fos_bearing": BEARING_FACTOR * Fty / brg_max[0],
           "fatigue_fos_min": min(v["fos"] for v in fat.values()),
           "method_validation": bench, "validation_err_pct": bench["max_err_pct"],
           "steering_arm_section_mm": d_sec,
           "steering_arm_err_pct": max(abs(v["error_pct"]) for v in results["LC6"]["validation"]),
           "convergence_change_pct": 100 * abs(conv[-1]["peaks_MPa"]["_global"] / conv[-2]["peaks_MPa"]["_global"] - 1),
           "max_reaction_ratio": max(r["max_reaction_N"] / r["max_load_N"] for r in results.values()),
           "max_moment_residual": max(r["moment_residual"] for r in results.values()),
           "seconds": time.time() - t0}
    write_report(out, reg, mat)
    plot_convergence(conv, reg)
    (DATA / "upright_fea.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")   # last: only a complete run is "current"
    print(json.dumps({k: out[k] for k in ("critical_case", "fos_yield", "fos_bearing", "fatigue_fos_min", "validation_err_pct",
                                          "steering_arm_err_pct",
                                          "convergence_change_pct", "richardson", "stiffness", "seconds")},
                     indent=1, default=float))


def _read_lcs():
    import csv
    with open(DATA / "loadcases.csv", newline="") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def write_report(out, reg, mat):
    ver = out["_hardpoints"]
    Fty = mat["Fty_MPa"]
    q = out["mesh_base"]
    sup_note = ""
    if q["straightened"]:
        sup_note = (f" {q['straightened']} elements with a scaled Jacobian below 0.05 after curving had their mid-side "
                    f"nodes put back on straight edges (worst afterwards {q['min_sj']:.2f}); the nearest is "
                    f"{min(np.linalg.norm(np.array(out['cases'][out['critical_case']]['peaks_MPa']['_global_at']) - np.array(p)) for p in q['straightened_at']):.1f} mm "
                    "from the governing peak.")
    fil = "; ".join(f"{f['feature']} {f['radius_mm']:g} mm" if f["radius_mm"] else f"{f['feature']} SHARP" for f in out["fillets"])
    md = [f"# Upright FEA (hardpoints {ver})", "",
          "Generated by `scripts/fea/upright_fea.py`. Do not edit by hand. gmsh + scikit-fem, quadratic tetrahedra; "
          f"7075-T7351, design yield {Fty:g} MPa (0.9 x typical {mat['Fty_typical_MPa']:g}, standing in for A-basis). Geometry: "
          "`cad/upright/upright_L_fea.step` (the one-piece upright, v4).", "",
          f"**Root fillets in the analysed geometry:** {fil}.", "",
          f"Base mesh: {out['dofs_base']:,} degrees of freedom, refined round {len(out['hot_spots'])} hot spots found by a "
          f"coarse pre-pass over every load case.{sup_note} Loads as they act (pin-bearing pressure in the ear and lug holes, "
          "cosine pressure on the cup seats tilted onto each row's pressure centre, thrust on the rib); every face a case "
          f"can load (each hole, both cup seats, both rib faces) area-checked on this mesh, within {100 * AREA_TOL:.0f} % of "
          "its drawing value; a minimal 3-2-1 support. Applied loads balance to a moment residual of at most "
          f"{100 * out['max_moment_residual']:.2f} % of (largest load x {MOMENT_ARM:g} mm) (the run stops above "
          f"{100 * MOMENT_LIMIT:.0f} %); the largest support reaction is {100 * out['max_reaction_ratio']:.2f} % of the largest load.", "",
          "## Peak von Mises stress per load case [MPa]", "",
          f"Raw stress at the integration points. Not in the peaks: within {SUPPORT_EXCLUDE_MM:g} mm of a held vertex; within "
          f"{HOLE_EDGE_MM:g} mm of a pin or bolt hole (pin contact; nominal bearing checked below); within {SHARP_EXCLUDE_MM:g} mm "
          "of a re-entrant corner left sharp (singular; hand calc).", "",
          "| LC | " + " | ".join(reg) + " | Global | At [mm] | Hole edges | Sharp corners | Largest reaction / largest load |",
          "|---|" + "---|" * (len(reg) + 5)]
    for lcid, r in out["cases"].items():
        pk = r["peaks_MPa"]
        md.append(f"| {lcid} | " + " | ".join(f"{pk[k]:.0f}" for k in reg) + f" | **{pk['_global']:.0f}** | "
                  f"{pk['_global_at']} | {pk['_hole_edges']:.0f} | {pk['_sharp_corners']:.0f} | "
                  f"{r['max_reaction_N'] / r['max_load_N']:.1e} |")
    md += ["", f"**Strength:** worst von Mises {out['worst_MPa']['_global']:.0f} MPa, yield FoS **{out['fos_yield']:.2f}** "
           f"(target {FOS}), excluding the roots left sharp (below). Nominal bearing stress P / (d t): worst {out['bearing_max_MPa']:.0f} MPa ({out['bearing_max_at']}), "
           f"FoS **{out['fos_bearing']:.2f}** against Fbry = {BEARING_FACTOR} x Fty (assumed). Hole-edge von Mises (pin contact, "
           f"not a yield check) reaches {out['worst_MPa']['_hole_edges']:.0f} MPa.", ""]
    if out["sharp_edges"]:
        md += [f"**Left sharp by the CAD kernel:** {', '.join(out['sharp_edges'])} (OpenCASCADE refused every radius at these "
               f"roots). Their singular FEA stress (up to {out['worst_MPa']['_sharp_corners']:.0f} MPa on this mesh) is not a "
               "result; the drawing calls for a 2 mm machining radius there (R2, note 5) and the roots are checked by hand calculation (HC-7, S-14).", ""]
    md += [f"## Mesh convergence ({out['critical_case']}, the governing case)", "",
           "| Element size global / local [mm] | DOFs | " + " | ".join(reg) + " | Global |", "|---|---|" + "---|" * (len(reg) + 1)]
    for c in out["convergence"]:
        md.append(f"| {c['size_mm'][0]:g} / {c['size_mm'][1]:g} | {c['dofs']:,} | " +
                  " | ".join(f"{c['peaks_MPa'][k]:.0f}" for k in reg) + f" | {c['peaks_MPa']['_global']:.0f} |")
    last, prev = out["convergence"][-1]["peaks_MPa"], out["convergence"][-2]["peaks_MPa"]
    md += ["", "Change between the two finest meshes: " + ", ".join(
        f"{k} {(last[k] / prev[k] - 1) * 100:+.1f} %" for k in reg) + f"; global {out['convergence_change_pct']:.1f} %. "
        "Regions whose peaks are far below the governing one may still move a few MPa; the governing peak and the arm "
        "roots are what the convergence target covers."]
    rich = out["richardson"]
    if rich:
        md.append(f"Richardson on the global peak (refinement ratio {out['refinement_ratio']:.2f}): observed order {rich['order']:.1f}, "
                  f"extrapolated {rich['extrapolated']:.0f} MPa, GCI {rich['gci_pct']:.1f} %.")
    else:
        g = [c["peaks_MPa"]["_global"] for c in out["convergence"]]
        why = "changes more on the finer step" if (g[1] - g[0]) * (g[2] - g[1]) > 0 else "is not monotone"
        md.append(f"No Richardson estimate: over the three meshes the global peak ({', '.join(f'{v:.0f}' for v in g)} MPa) "
                  f"{why}, so the meshes are not in the asymptotic range where it applies. The change between the two "
                  "finest meshes measures the mesh's effect; it is not an error bound.")
    md.append("")
    bench = out["method_validation"]
    md += ["## Validation of the method (S-06c)", "",
           "The upright has no section slender enough for beam theory, so the method is validated where beam theory holds: "
           "a 7075 bar 160 x 32 x 6 mm (the steering-arm ear's section, made long), clamped at one end and loaded through an "
           "8.1 mm pin hole by 1000 N across and 300 N along it, with the same mesher, pin-bearing load and integration-point "
           "stress as the upright (`benchmark_lug.py`). Normal stress at mid-span, 3 mm band at each edge:", "",
           "| Fibre [mm] | FEA [MPa] | Beam theory [MPa] | Error |", "|---|---|---|---|"]
    md += [f"| {v['fibre_mm']:+.1f} | {v['fea_MPa']:.2f} | {v['hand_MPa']:.2f} | {v['error_pct']:+.2f} % |" for v in bench["rows"]]
    md.append("")
    val = out["cases"].get("LC6", {}).get("validation")
    if val:
        md += [f"For information, the upright's steering-arm ears in LC6 against the same beam formula, at "
               f"{out['steering_arm_section_mm']:.1f} mm from the tie-rod bolt (halfway between the hole edge and the root): "
               "the arm is only about 20 mm long and 32 mm wide, so the whole of it is within the load and root disturbance "
               "and beam theory is not expected to hold:", "",
               "| Fibre [mm] | FEA [MPa] | Beam formula [MPa] | Difference |", "|---|---|---|---|"]
        md += [f"| {v['fibre_mm']:+.1f} | {v['fea_MPa']:.1f} | {v['hand_MPa']:.1f} | {v['error_pct']:+.1f} % |" for v in val]
        md.append("")
    md += ["## Stiffness: the upright's share (rotation of the bearing bore relative to the three joints)", "",
           "The upright alone, with the clevis tied on: bearing tilt, the spindle, the clevis joint and the links add to it "
           "(Week 3b compliance budget).", "",
           "| Case | Camber [deg/kN] | Toe [deg/kN] |", "|---|---|---|"]
    md += [f"| {k} | {v['camber_deg_per_kN']:.4f} | {v['toe_deg_per_kN']:.4f} |" for k, v in out["stiffness"].items()]
    md += ["", "## Fatigue (Goodman)", "",
           f"Se = {out['fatigue']['Se_MPa']:.0f} MPa: {mat['fatigue_strength_5e8_MPa']:g} MPa at 5e8 cycles x ka (machined, Shigley's "
           "fit) x kb 0.9 x ke 0.814 (99 % reliability). Amplitude = von Mises of half the stress-tensor range, mean = signed "
           f"von Mises of the mean; Goodman line to the design Ftu {mat['Ftu_MPa']:g} MPa; target FoS {FOS}. "
           "\"Static\" is LC1 / 3 (LC1 is a 3 g bump at +30 mm of travel).", ""]
    md += [f"- {k}: FoS **{v['fos']:.2f}** at {v['at']} mm" for k, v in out["fatigue"].items() if isinstance(v, dict)]
    md += [f"- at the hole edges (pin contact, for information): {k} FoS {v['fos']:.2f}" for k, v in out["fatigue_hole_edges"].items()]
    if min(v["fos"] for v in out["fatigue"].values() if isinstance(v, dict)) < FOS:
        md += ["", f"This is a screen: it asks for unlimited life with every cycle at the peak load, which 7075 cannot give, so it "
               f"falls short of {FOS:g} (deviation DV-3, S-07a). The fatigue check is Miner's rule over the duty spectrum in "
               "`notes/upright_fatigue.md` (S-07c)."]
    (NOTES / "upright_fea.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def plot_convergence(conv, reg):
    fig, axs = figure(1, 1, w=6.4, h=3.8)
    ax = axs[0][0]
    dofs = [c["dofs"] for c in conv]
    for i, k in enumerate(reg):
        ax.plot(dofs, [c["peaks_MPa"][k] for c in conv], marker="o", markersize=4, color=SERIES[i % len(SERIES)],
                linestyle="-" if i < len(SERIES) else ":", label=k)
    ax.plot(dofs, [c["peaks_MPa"]["_global"] for c in conv], marker="s", markersize=5, color=INK_2, linestyle="--",
            linewidth=2, label="global peak")
    ax.set_xscale("log")
    style_axes(ax, "Upright: peak von Mises vs mesh size (governing case)", "degrees of freedom", "peak von Mises [MPa]")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "fea01_upright_convergence.png", dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
