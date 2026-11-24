"""Week 1 physics tests for the front-suspension package (scripts/).

Run from the project root:  python -m pytest tests -q

Sign conventions are those of suspension.py (ISO 8855, LEFT front corner): x fwd, y left, z up,
camber - = top inboard, steer + = CCW from above, tension + in link forces.

Every expected value below comes from geometry or statics worked out independently of the code
under test (analytic 2D constructions, a parallelogram linkage, force/moment sums written from
scratch, a kingpin-axis moment balance, an energy balance), not from the code's own outputs.
"""
import copy
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep scripts/ free of __pycache__

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import forces  # noqa: E402
import kinematics  # noqa: E402
from common import DATA, EZ, NOTES, PLOTS, load_hardpoints, load_vehicle, unit  # noqa: E402
from suspension import LINKS, UPRIGHT_PTS, Corner, SolveError, front_view_ic, line_intersect_2d, make_corner, mirror, point_velocity  # noqa: E402

HP, HP_META = load_hardpoints()
VEH = load_vehicle()
R_TYRE = VEH["tyre"]["loaded_radius_mm"]
CORNER = make_corner(HP, HP_META, R_TYRE)
# v1.3+: the pushrod picks up on the lower-wishbone node and its force enters the upright through
# the lower ball joint, along its own line (exact at static, where the line passes through it)
ACTS = {"Pushrod": "LCA_O"} if CORNER.pushrod_on == "lca" else {}
MEMBERS = forces.read_csv_rows(DATA / "members.csv")
LOAD_CASES = forces.read_csv_rows(DATA / "loadcases.csv")
IDX = {m["member"]: j for j, m in enumerate(MEMBERS)}

_POSES = {}


def get_pose(z, rack=0.0):
    """Cached static/loaded pose of the real corner (poses are treated as read-only)."""
    key = (float(z), float(rack))
    if key not in _POSES:
        _POSES[key] = CORNER.pose(*key)
    return _POSES[key]


def lc_force(lc):
    return np.array([float(lc["Fx_N"]), float(lc["Fy_N"]), float(lc["Fz_N"])])


def isect_2d(p1, d1, p2, d2):
    """Independent line intersection (cross-product form), lines p + t d in the (y, z) plane."""
    p1, d1, p2, d2 = (np.asarray(a, float) for a in (p1, d1, p2, d2))
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    return p1 + t * d1


def synthetic_hp(overrides, tr_o):
    """Real hardpoints with the upright/wishbone/tie-rod points replaced. The rocker, pushrod and
    ARB group is inherited from the real set so Corner.pose() can still close its post-upright loops."""
    h = {k: v.copy() for k, v in HP.items()}
    for k, v in overrides.items():
        h[k] = np.array(v, float)
    h["TR_O"] = np.array(tr_o, float)
    return h


# ============================================================ 1. static pose
def test_static_pose_reproduces_design_values():
    p = get_pose(0.0)
    assert p["camber"] == pytest.approx(-1.25, abs=0.01)
    assert p["caster"] == pytest.approx(5.0, abs=0.01)
    assert p["kpi"] == pytest.approx(7.0, abs=0.01)
    assert p["scrub"] == pytest.approx(16.5, abs=0.01)
    assert p["trail"] == pytest.approx(15.0, abs=0.01)
    assert p["cp"][1] == pytest.approx(610.0, abs=0.01)
    assert p["cp"][2] == pytest.approx(0.0, abs=0.01)


def test_static_pose_has_zero_steer_and_cp_under_wheel_centre_in_x():
    p = get_pose(0.0)
    assert p["steer"] == pytest.approx(0.0, abs=1e-6)
    assert p["cp"][0] == pytest.approx(HP["WC"][0], abs=1e-6)


# ============================================================ 2. rigid links
Z_SET = [-30.0, -10.0, 0.0, 15.0, 30.0]
RACK_SET = [-20.0, 0.0, 20.0]


@pytest.mark.parametrize("rack", RACK_SET)
@pytest.mark.parametrize("z", Z_SET)
def test_link_lengths_constant(z, rack):
    P = get_pose(z, rack)["P"]
    errs = {}
    for a, b in LINKS:
        errs[f"{a}-{b}"] = abs(np.linalg.norm(P[a] - P[b]) - np.linalg.norm(HP[a] - HP[b]))
    errs["pushrod"] = abs(np.linalg.norm(P["PR_I"] - P["PR_O"]) - np.linalg.norm(HP["PR_I"] - HP["PR_O"]))
    errs["ARB drop link"] = abs(np.linalg.norm(P["ARB_L"] - P["ARB_R"]) - np.linalg.norm(HP["ARB_L"] - HP["ARB_R"]))
    worst = max(errs, key=errs.get)
    assert errs[worst] < 1e-6, f"{worst}: length drift {errs[worst]:.3e} mm"


@pytest.mark.parametrize("rack", RACK_SET)
@pytest.mark.parametrize("z", Z_SET)
def test_upright_and_rocker_stay_rigid_and_drive_coordinate_is_honoured(z, rack):
    P = get_pose(z, rack)["P"]
    for i, a in enumerate(UPRIGHT_PTS):
        for b in UPRIGHT_PTS[i + 1:]:
            assert abs(np.linalg.norm(P[a] - P[b]) - np.linalg.norm(HP[a] - HP[b])) < 1e-6, f"upright {a}-{b}"
    for a in ("PR_I", "DMP_R", "ARB_R"):
        assert abs(np.linalg.norm(P[a] - P["RKR_P"]) - np.linalg.norm(HP[a] - HP["RKR_P"])) < 1e-6, f"rocker {a}"
    assert P["WC"][2] == pytest.approx(HP["WC"][2] + z, abs=1e-8)


# ============================================================ 3. independent 2D front-view IC
def sla_hardpoints(uca_in, uca_o, lca_in, lca_o, half_u=150.0, half_l=180.0):
    """Wishbone pivot axes parallel to x (pivot pairs at +/- half-spacing, same y and z), both ball
    joints at x = 0. Ball joints then move in the x = 0 plane exactly like 2D arms of length
    sqrt(dy^2 + dz^2). The tie-rod inboard point is placed on the line from TR_O through the 2D
    instant centre, which makes the first-order bump steer zero (tie-rod velocity constraint)."""
    ic = isect_2d(uca_in, np.subtract(uca_o, uca_in), lca_in, np.subtract(lca_o, lca_in))
    tr_o = np.array([60.0, 600.0, 140.0])
    y_i = 214.5
    z_i = tr_o[2] + (y_i - tr_o[1]) * (ic[1] - tr_o[2]) / (ic[0] - tr_o[1])
    ov = {"UCA_F": (half_u, *uca_in), "UCA_R": (-half_u, *uca_in), "UCA_O": (0.0, *uca_o),
          "LCA_F": (half_l, *lca_in), "LCA_R": (-half_l, *lca_in), "LCA_O": (0.0, *lca_o),
          "TR_I": (60.0, y_i, z_i), "WC": (0.0, 605.0, 200.0), "WA": (0.0, 655.0, 200.0)}
    return synthetic_hp(ov, tr_o), ic


SLA_GEOMS = {  # (y, z) pairs
    "ic_inboard": dict(uca_in=(230.0, 200.0), uca_o=(560.0, 290.0), lca_in=(200.0, 90.0), lca_o=(590.0, 110.0)),
    "ic_outboard": dict(uca_in=(250.0, 320.0), uca_o=(560.0, 290.0), lca_in=(180.0, 90.0), lca_o=(590.0, 110.0)),
}
_SLA = {}


def sla_corner(name):
    if name not in _SLA:
        hp, ic = sla_hardpoints(**SLA_GEOMS[name])
        _SLA[name] = (Corner(hp, R_TYRE), ic)
    return _SLA[name]


@pytest.mark.parametrize("z", [-10.0, 0.0, 10.0])
@pytest.mark.parametrize("geom", list(SLA_GEOMS))
def test_front_view_ic_matches_2d_arm_intersection(geom, z):
    corner, ic0 = sla_corner(geom)
    pose = corner.pose(z)
    w, v_c, c, _, _ = corner.velocity(pose)
    ic = front_view_ic(w, v_c, c, pose["cp"])
    P = pose["P"]
    d_u, d_l = P["UCA_O"][1:] - P["UCA_F"][1:], P["LCA_O"][1:] - P["LCA_F"][1:]
    expected = isect_2d(P["UCA_F"][1:], d_u, P["LCA_F"][1:], d_l)
    assert np.linalg.norm(ic[1:] - expected) < 1.0, f"IC (y,z) {ic[1:]} vs 2D {expected}"
    # the library's own 2D intersection helper agrees with the independent one
    assert np.allclose(line_intersect_2d(P["UCA_F"][1:], d_u, P["LCA_F"][1:], d_l), expected, atol=1e-6)
    if z == 0.0:  # static: posed lines are the raw hardpoint lines
        assert np.linalg.norm(expected - ic0) < 1e-9


@pytest.mark.parametrize("geom", list(SLA_GEOMS))
def test_synthetic_tie_rod_tuning_gives_negligible_bump_steer(geom):
    """Guards test 3: the 'planar' premise only holds if bump steer really is negligible."""
    corner, _ = sla_corner(geom)
    w, *_ = corner.velocity(corner.pose(0.0))
    assert abs(math.degrees(w[2])) < 1e-6, "steer rate at z=0 [deg/mm]"
    for z in (-10.0, 10.0):
        assert abs(corner.pose(z)["steer"]) < 0.01, f"toe change at z={z}"


def test_line_intersect_2d_known_cases():
    assert np.allclose(line_intersect_2d(np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                                         np.array([0.0, 2.0]), np.array([1.0, -1.0])), [1.0, 1.0])
    # same intersection regardless of direction vector length or sign
    assert np.allclose(line_intersect_2d(np.array([0.0, 0.0]), np.array([-3.0, -3.0]),
                                         np.array([0.0, 2.0]), np.array([0.5, -0.5])), [1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError):  # exactly parallel lines have no intersection
        line_intersect_2d(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 0.0]))


# ============================================================ 4. equal-length parallel horizontal arms
def parallel_arm_hp():
    """Parallelogram: UCA and LCA are horizontal, equal (400 mm in front view), tie rod also
    horizontal and 400 mm. The upright then translates (no rotation) on a 400 mm-radius arc."""
    ov = {"UCA_F": (150, 200, 300), "UCA_R": (-150, 200, 300), "UCA_O": (0, 600, 300),
          "LCA_F": (150, 200, 100), "LCA_R": (-150, 200, 100), "LCA_O": (0, 600, 100),
          "TR_I": (60, 200, 150), "WC": (0, 600, 200), "WA": (0, 650, 200)}
    return synthetic_hp(ov, (60, 600, 150))


class CleanStartCorner(Corner):
    """Test-side workaround for a solver defect (see test_parallel_arms_assemble_via_pose...):
    zero out junk rotation components (|q| < 1e-7) in the warm start so fsolve's relative
    finite-difference step does not collapse. Physics is untouched."""

    def _solve_q(self, z, rack, q0):
        return super()._solve_q(z, rack, np.where(np.abs(q0) < 1e-7, 0.0, q0))


def _veh_for_enrich():
    v = dict(VEH)
    v["_front_brake_fraction"] = 0.6
    return v


def test_parallel_arms_assemble_via_pose_and_camber_unchanged():
    """User spec: camber change over +/-10 mm ~ 0. Uses the stock Corner. The exact parallelogram
    is a legitimate mechanism (analytic solution: pure translation on a circular arc) but
    Corner.solve_q marches in 2 mm steps and the rotation entries of the warm start come back as
    ~1e-21 instead of 0; fsolve's finite-difference step is proportional to |x| so the Jacobian
    columns vanish and the next step fails with SolveError."""
    corner = Corner(parallel_arm_hp(), R_TYRE)
    failed, camber = {}, {}
    for z in (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0):
        try:
            camber[z] = corner.pose(z)["camber"]
        except SolveError as e:
            failed[z] = str(e).split(":")[0]
    assert not failed, f"Corner.pose could not assemble the parallelogram at z = {sorted(failed)}: {list(failed.values())[0]}"
    assert max(abs(c) for c in camber.values()) < 1e-6


@pytest.mark.parametrize("z", [-10.0, -5.0, 5.0, 10.0])
def test_parallel_arms_physics_with_clean_warm_start(z):
    """Same mechanism through CleanStartCorner: camber and bump steer exactly zero, and the RC
    follows the analytic law rc = -cp_y * tan(asin(z / 400)) (arms tilt as the arc is swept)."""
    corner = CleanStartCorner(parallel_arm_hp(), R_TYRE)
    pose = corner.pose(z)
    assert abs(pose["camber"]) < 1e-6
    assert abs(pose["steer"]) < 1e-6
    # contact patch follows the circular arc: y = 600 - (400 - sqrt(400^2 - z^2)), z = z
    assert pose["cp"][1] == pytest.approx(600.0 - (400.0 - math.sqrt(400.0 ** 2 - z ** 2)), abs=1e-6)
    w, v_c, c, _, _ = corner.velocity(pose)
    v = point_velocity(w, v_c, c, pose["cp"])
    rc = pose["cp"][1] * v[1] / v[2]
    assert rc == pytest.approx(-pose["cp"][1] * math.tan(math.asin(z / 400.0)), abs=0.01)
    e = kinematics.enrich(corner, copy.copy(pose), _veh_for_enrich())
    assert e["rc_heave"] == pytest.approx(rc, abs=1e-3)


def test_parallel_arms_roll_centre_at_ground_level_statically():
    """User spec: RC at ground level within ~1 mm (static pose, stock Corner)."""
    corner = Corner(parallel_arm_hp(), R_TYRE)
    pose = corner.pose(0.0)
    assert pose["cp"][2] == pytest.approx(0.0, abs=1e-6)
    e = kinematics.enrich(corner, copy.copy(pose), _veh_for_enrich())
    assert abs(e["rc_heave"]) < 1.0, f"rc_heave = {e['rc_heave']}"
    w, v_c, c, _, _ = corner.velocity(pose)  # own RC from the contact-patch velocity direction
    v = point_velocity(w, v_c, c, pose["cp"])
    assert abs(pose["cp"][1] * v[1] / v[2]) < 1.0


# ============================================================ 5. link-force statics
def equilibrium_residuals(pose, f, F, at):
    """Sum of forces and sum of moments about WC and about the origin, written from scratch."""
    P = pose["P"]
    c = P["WC"]
    force = np.array(F, float)
    mom_c = np.cross(at - c, F)
    mom_o = np.cross(at, F)
    for fj, m in zip(f, MEMBERS):
        u = unit(P[m["in"]] - P[m["out"]])  # tension pulls the outboard end toward the inboard end
        r = P[ACTS.get(m["member"], m["out"])]
        force = force + fj * u
        mom_c = mom_c + np.cross(r - c, fj * u)
        mom_o = mom_o + np.cross(r, fj * u)
    return force, mom_c, mom_o


def lc_point(lc, pose):
    return forces.load_point(CORNER, pose, lc, VEH["tyre"]["tread_width_mm"])


LC_POSES = [(lc["id"], tag, z) for lc in LOAD_CASES for tag, z in (("static", 0.0), ("loaded", float(lc["z_pose_mm"])))]
LC_IDS = [f"{a}-{b}" for a, b, _ in LC_POSES]
LC_BY_ID = {lc["id"]: lc for lc in LOAD_CASES}


@pytest.mark.parametrize("lc_id,tag,z", LC_POSES, ids=LC_IDS)
def test_solve_links_equilibrium(lc_id, tag, z):
    lc = LC_BY_ID[lc_id]
    F = lc_force(lc)
    pose = get_pose(z)
    at = lc_point(lc, pose)
    f, cond, resid = forces.solve_links(pose, MEMBERS, F, ACTS, at)
    force, mom_c, mom_o = equilibrium_residuals(pose, f, F, at)
    assert np.abs(force).max() < 1e-6, f"force imbalance {force} N"
    assert np.abs(mom_c).max() < 1e-6, f"moment imbalance about WC {mom_c} N mm"
    assert np.abs(mom_o).max() < 1e-5, f"moment imbalance about origin {mom_o} N mm"
    assert resid < 1e-6
    assert cond < 1e4, f"cond(A) = {cond}"


@pytest.mark.parametrize("lc_id,tag,z", LC_POSES, ids=LC_IDS)
def test_pushrod_force_matches_virtual_work(lc_id, tag, z):
    """Virtual work is exact for the real mechanism. The 6x6 applies the pushrod at the lower ball
    joint, which is exact at static (the line passes through it) and off by the line's small
    drift elsewhere, so the loaded-pose tolerance is 1 % (about 0.2 % observed)."""
    lc = LC_BY_ID[lc_id]
    F = lc_force(lc)
    pose = get_pose(z)
    at = lc_point(lc, pose)
    f, _, _ = forces.solve_links(pose, MEMBERS, F, ACTS, at)
    fv = forces.virtual_work_pushrod(CORNER, pose, F, at)
    tol = 1e-4 if (tag == "static" or not ACTS) else 1e-2
    assert abs(f[IDX["Pushrod"]] - fv) / abs(f[IDX["Pushrod"]]) < tol, f"6x6 {f[IDX['Pushrod']]:.3f} vs VW {fv:.3f}"


@pytest.mark.parametrize("lc_id", [lc["id"] for lc in LOAD_CASES])
def test_damper_force_matches_energy_balance(lc_id):
    """rocker_and_damper vs virtual work on the damper: F . v_cp = f_d * d(len)/dz, d(len)/dz = -MR."""
    lc = LC_BY_ID[lc_id]
    F, z = lc_force(lc), float(lc["z_pose_mm"])
    pose = get_pose(z)
    at = lc_point(lc, pose)
    f, _, _ = forces.solve_links(pose, MEMBERS, F, ACTS, at)
    f_d, _ = forces.rocker_and_damper(pose, CORNER, f[IDX["Pushrod"]])
    w, v_c, c, mr, _ = CORNER.velocity(pose)
    f_d_vw = -np.dot(F, point_velocity(w, v_c, c, at)) / mr
    tol = 1e-2 if (ACTS and abs(z) > 1e-9) else 1e-4  # same idealisation as the pushrod test
    assert f_d == pytest.approx(f_d_vw, rel=tol), f"rocker balance {f_d:.2f} vs energy {f_d_vw:.2f}"


def test_solve_links_zero_load_and_superposition():
    pose = get_pose(0.0)
    f0, _, _ = forces.solve_links(pose, MEMBERS, np.zeros(3))
    assert np.abs(f0).max() < 1e-9
    F1, F2 = np.array([300.0, -200.0, 900.0]), np.array([-150.0, 400.0, 1200.0])
    f1, _, _ = forces.solve_links(pose, MEMBERS, F1)
    f2, _, _ = forces.solve_links(pose, MEMBERS, F2)
    f12, _, _ = forces.solve_links(pose, MEMBERS, F1 + 2 * F2)
    assert np.allclose(f12, f1 + 2 * f2, rtol=1e-9, atol=1e-8)


# ============================================================ 6. pure vertical load
VERT = np.array([0.0, 0.0, 1000.0])


@pytest.mark.parametrize("z", [-15.0, 0.0, 15.0])
def test_pure_vertical_load_puts_pushrod_in_compression(z):
    f, _, _ = forces.solve_links(get_pose(z), MEMBERS, VERT)
    assert f[IDX["Pushrod"]] < 0, f"pushrod force {f[IDX['Pushrod']]:.1f} N (tension +)"
    assert abs(f[IDX["Pushrod"]]) > VERT[2] * 0.5  # wheel load is amplified through the 1/MR path, not reduced


def test_pure_vertical_load_tie_rod_force_is_small_vs_pushrod():
    """'Small' read as < 10 % of the pushrod force, at wheel travel -15 / 0 / +15 mm."""
    ratios = {}
    for z in (-15.0, 0.0, 15.0):
        f, _, _ = forces.solve_links(get_pose(z), MEMBERS, VERT)
        ratios[z] = abs(f[IDX["Tie rod"]]) / abs(f[IDX["Pushrod"]])
    worst = max(ratios.values())
    assert worst < 0.10, "tie rod / pushrod force ratio by z: " + ", ".join(f"{z:+.0f} mm: {r * 100:.1f} %" for z, r in ratios.items())


@pytest.mark.parametrize("z", [-15.0, 0.0, 15.0])
def test_pure_vertical_load_tie_rod_force_matches_kingpin_moment_balance(z):
    """Independent derivation: the wishbone ball joints sit ON the kingpin axis, so only the tie
    rod, pushrod and ground force carry moment about that axis. Pushrod force taken from virtual
    work (not from the 6x6), so this checks the 6x6 tie-rod entry independently."""
    pose = get_pose(z)
    P, k, o = pose["P"], pose["kingpin"], pose["P"]["LCA_O"]
    u_t, u_p = unit(P["TR_I"] - P["TR_O"]), unit(P["PR_I"] - P["PR_O"])
    a_t = k @ np.cross(P["TR_O"] - o, u_t)
    a_p = k @ np.cross(P[ACTS.get("Pushrod", "PR_O")] - o, u_p)   # 0 when it acts through the ball joint
    m_F = k @ np.cross(pose["cp"] - o, VERT)
    f, _, _ = forces.solve_links(pose, MEMBERS, VERT, ACTS)
    f_tie = -(f[IDX["Pushrod"]] * a_p + m_F) / a_t
    assert f[IDX["Tie rod"]] == pytest.approx(f_tie, rel=1e-6)


# ============================================================ 7. mirror, steer symmetry, Ackermann
def test_mirror_flips_only_y_and_is_an_involution():
    p = np.array([12.5, 340.25, -7.0])
    assert np.allclose(mirror(p), [12.5, -340.25, -7.0])
    assert np.allclose(mirror(mirror(p)), p)
    assert np.allclose(mirror(np.zeros(3)), 0.0)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    assert np.allclose(mirror(arr), [[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])


def right_wheel_steer(pose):
    """Steer of a RIGHT wheel in the vehicle frame (wheel_geometry assumes the left wheel: its
    heading comes out flipped by 180 deg on a mirrored corner)."""
    n = unit(pose["P"]["WA"] - pose["P"]["WC"])  # outboard = -y
    h = unit(np.cross(EZ, n))                      # forward
    return math.degrees(math.atan2(h[1], h[0]))


@pytest.mark.parametrize("s", [2.0, 5.0, 12.0, 20.0, 40.0])
def test_steer_mirror_symmetry_left_vs_independent_right_corner(s):
    right = Corner({k: mirror(v) for k, v in HP.items()}, R_TYRE)  # separately solved mirrored corner
    left_pose = get_pose(0.0, +s)
    right_pose = right.pose(0.0, -s)
    assert left_pose["steer"] == pytest.approx(-right_wheel_steer(right_pose), abs=1e-6)
    assert left_pose["camber"] == pytest.approx(right_pose["camber"], abs=1e-6)
    for k in ("UCA_O", "LCA_O", "TR_O", "WC", "WA", "TR_I"):  # whole geometry is the mirror image
        assert np.allclose(right_pose["P"][k], mirror(left_pose["P"][k]), atol=1e-6), k


def test_rack_direction_convention_left_turn_is_positive_rack():
    assert get_pose(0.0, +10.0)["steer"] > 0 > get_pose(0.0, -10.0)["steer"]


@pytest.fixture(scope="module")
def evaluated():
    def snapshot():
        return {str(p): p.stat().st_mtime_ns for d in (DATA, NOTES, PLOTS) for p in d.rglob("*") if p.is_file()}
    before = snapshot()
    out = kinematics.evaluate(plots=False, write=False, quiet=True)
    return out, before, snapshot()


def test_evaluate_without_write_has_no_file_side_effects(evaluated):
    _, before, after = evaluated
    assert before == after


def test_evaluate_returns_expected_structure(evaluated):
    out = evaluated[0]
    for key in ("results", "heave", "roll", "ack"):
        assert key in out
    metrics = {r["metric"]: r for r in out["results"]}
    assert metrics["static_camber"]["value"] == pytest.approx(get_pose(0.0)["camber"], abs=1e-9)
    assert len(out["heave"]) == 61 and len(out["ack"]) > 10


def test_ackermann_identity_in_table(evaluated):
    """Ideal (100 %) Ackermann satisfies cot(outer) - cot(inner) = track / wheelbase."""
    out = evaluated[0]
    t_over_l = VEH["track_front_mm"] / VEH["wheelbase_mm"]
    for a in out["ack"]:
        cot = lambda deg: 1.0 / math.tan(math.radians(deg))
        assert cot(a["outer"]) - cot(a["inner_100pct"]) == pytest.approx(t_over_l, abs=1e-9)
        assert a["ackermann"] == pytest.approx((a["inner"] - a["outer"]) / (a["inner_100pct"] - a["outer"]) * 100.0, abs=1e-9)


def test_ackermann_small_angle_limit_is_finite_and_smooth():
    veh = copy.deepcopy(VEH)
    racks = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.05), 6)
    rows = kinematics.ackermann_table(kinematics.steer_sweep(CORNER, veh, racks), veh)
    assert rows, "no Ackermann rows"
    ack = np.array([r["ackermann"] for r in rows])
    outer = np.array([r["outer"] for r in rows])
    assert np.all(np.isfinite(ack)), "NaN/inf in Ackermann near zero steer"
    assert outer.min() > 0.05, "0/0 guard: rows with outer <= 0.05 deg must be skipped"
    assert outer.min() < 0.2, "fine grid should reach the near-zero-steer regime"
    small = ack[outer < 0.5]
    assert small.max() - small.min() < 0.5, f"Ackermann not settling to a limit: {small.min():.3f}..{small.max():.3f}"
    mid = ack[(outer > 4.0) & (outer < 5.0)] if np.any((outer > 4.0) & (outer < 5.0)) else ack[-1:]
    assert abs(small[0] - mid.mean()) < 5.0, "small-angle value should approach the value at moderate angles"
    assert np.all(np.abs(ack) < 150.0), "Ackermann blew up near zero steer"


def test_ackermann_within_40_to_60_between_5deg_and_steering_stop(evaluated):
    out = evaluated[0]
    metrics = {r["metric"]: r for r in out["results"]}
    assert "ackermann_min" in metrics and "ackermann_max" in metrics, "no 5 deg .. stop window rows"
    assert 40.0 <= metrics["ackermann_min"]["value"] <= 60.0, metrics["ackermann_min"]["value"]
    assert 40.0 <= metrics["ackermann_max"]["value"] <= 60.0, metrics["ackermann_max"]["value"]
    window = [a["ackermann"] for a in out["ack"] if 5.0 <= a["outer"] <= out["d_stop"]]
    assert len(window) >= 5
    assert min(window) >= 40.0 and max(window) <= 60.0, f"{min(window):.2f}..{max(window):.2f}"
    assert min(window) == pytest.approx(metrics["ackermann_min"]["value"], abs=1e-9)


def test_inner_wheel_steers_more_than_outer_in_a_turn(evaluated):
    assert all(a["inner"] > a["outer"] > 0 for a in evaluated[0]["ack"])


# ============================================================ 8. section properties
def test_section_15875_x_0889_matches_hand_values():
    A, I = forces.section(15.875, 0.889)
    assert A == pytest.approx(41.9, rel=2e-3)
    assert I == pytest.approx(1179.0, rel=1e-3)
    # exact closed forms: A = pi t (OD - t); I = pi/64 (OD^4 - ID^4) with ID = 14.097
    assert A == pytest.approx(math.pi * 0.889 * (15.875 - 0.889), rel=1e-12)
    assert I == pytest.approx(math.pi / 64 * (15.875 ** 4 - 14.097 ** 4), rel=1e-12)
    # thin-wall estimate I ~ pi r_m^3 t (independent route) is within 1 %
    assert I == pytest.approx(math.pi * ((15.875 - 0.889) / 2) ** 3 * 0.889, rel=1e-2)


@pytest.mark.parametrize("m", MEMBERS, ids=[m["member"] for m in MEMBERS])
def test_section_matches_closed_form_for_every_member_tube(m):
    od, t = float(m["tube_od_mm"]), float(m["tube_wall_mm"])
    A, I = forces.section(od, t)
    assert A == pytest.approx(math.pi * t * (od - t), rel=1e-12)
    assert I == pytest.approx(math.pi / 64 * (od ** 4 - (od - 2 * t) ** 4), rel=1e-12)
    assert I == pytest.approx(math.pi * ((od - t) / 2) ** 3 * t, rel=2e-2)  # thin-wall sanity


def test_section_limits_zero_wall_and_solid_bar():
    for bad in (0.0, -0.5, 10.5):  # no material, negative wall, wall past the centre
        with pytest.raises(ValueError):
            forces.section(20.0, bad)
    A1, I1 = forces.section(20.0, 10.0)  # wall = OD/2 -> solid rod
    assert A1 == pytest.approx(math.pi / 4 * 20.0 ** 2) and I1 == pytest.approx(math.pi / 64 * 20.0 ** 4)


def test_section_thicker_wall_increases_area_and_inertia():
    (A1, I1), (A2, I2) = forces.section(16.0, 1.0), forces.section(16.0, 1.5)
    assert A2 > A1 and I2 > I1


# ============================================================ vehicle_calcs (read-only checks)
def test_vehicle_calcs_import_has_no_side_effects_and_loadcases_are_consistent():
    before = (DATA / "loadcases.csv").stat().st_mtime_ns, (DATA / "vehicle_derived.json").stat().st_mtime_ns
    import vehicle_calcs
    assert ((DATA / "loadcases.csv").stat().st_mtime_ns, (DATA / "vehicle_derived.json").stat().st_mtime_ns) == before
    ks = vehicle_calcs.kin_summary()
    assert 0.5 < ks["motion_ratio"] < 1.5
    fz0 = VEH["mass_total_kg"] * VEH["front_weight_fraction"] * vehicle_calcs.G / 2
    by_id = {lc["id"]: lc for lc in VEH["load_cases"]}
    lc1 = LC_BY_ID["LC1"]
    assert float(lc1["Fz_N"]) == pytest.approx(3.0 * fz0, abs=0.2)  # 3 g bump, static weight transfer-free
    for row in LOAD_CASES:  # CSV rounded to 0.1 N
        spec = by_id[row["id"]]
        if spec["type"] == "g":  # friction utilisation: Fx = ax Fz, Fy towards the turn centre
            sgn = -1.0 if spec["wheel"] == "inside" else 1.0
            assert float(row["Fx_N"]) == pytest.approx(spec["ax_g"] * float(row["Fz_N"]), abs=0.2)
            assert float(row["Fy_N"]) == pytest.approx(-sgn * spec["ay_g"] * float(row["Fz_N"]), abs=0.2)
            assert float(row["brake_torque_Nm"]) == pytest.approx(abs(float(row["Fx_N"])) * VEH["tyre"]["loaded_radius_mm"] / 1000, abs=0.2)
        else:  # direct: multiples of the static corner load, no brake torque off the contact patch
            for axis in "xyz":
                assert float(row[f"F{axis}_N"]) == pytest.approx(spec[f"f{axis}_static"] * fz0, abs=0.2)
            assert float(row["brake_torque_Nm"]) == pytest.approx(0.0, abs=0.05)


# ============================================================ review follow-ups (2026-10-24)
def test_column_capacity_johnson_meets_euler_at_transition_and_caps_at_yield():
    A, I, E, sy = 41.85, 1179.1, 205000.0, 348.0
    r = math.sqrt(I / A)
    L_t = math.pi * r * math.sqrt(2 * E / sy)  # transition length: Euler stress = sy / 2
    P_lo, m_lo = forces.column_capacity(A, I, L_t * 0.999, E, sy)
    P_hi, m_hi = forces.column_capacity(A, I, L_t * 1.001, E, sy)
    assert (m_lo, m_hi) == ("Johnson", "Euler")
    assert P_lo == pytest.approx(P_hi, rel=5e-3)  # continuous across the transition
    assert forces.column_capacity(A, I, 1.0, E, sy)[0] == pytest.approx(A * sy, rel=1e-3)  # stubby -> squash load
    assert forces.column_capacity(A, I, 433.0, E, sy)[0] < math.pi ** 2 * E * I / 433.0 ** 2  # below Euler


def test_vehicle_calcs_refuses_stale_kinematics_summary(tmp_path, monkeypatch):
    import vehicle_calcs
    src = (DATA / "kinematics_summary.csv").read_text().splitlines()
    (tmp_path / "kinematics_summary.csv").write_text("\n".join(["# hardpoints,v0.0"] + src[1:]) + "\n")
    (tmp_path / "hardpoints.csv").write_text((DATA / "hardpoints.csv").read_text())
    monkeypatch.setattr(vehicle_calcs, "DATA", tmp_path)
    with pytest.raises(RuntimeError, match="re-run kinematics"):
        vehicle_calcs.kin_summary()


def test_pushrod_pickup_on_kingpin_axis_keeps_vertical_load_out_of_the_steering():
    hp, _ = load_hardpoints()
    c = Corner(hp, VEH["tyre"]["loaded_radius_mm"])
    p = c.pose(0.0)
    f, _, _ = forces.solve_links(p, MEMBERS, np.array([0.0, 0.0, 1000.0]))
    names = [m["member"] for m in MEMBERS]
    assert abs(f[names.index("Tie rod")]) < 0.1 * abs(f[names.index("Pushrod")])


def test_misalignment_check_catches_the_v12_upright_pushrod_and_passes_v13_plus():
    """The v1.2 pushrod picked up on the upright: steering swung its lower rod end 20+ deg.
    The misalignment check must flag that, and must pass the current (LCA-node) geometry."""
    import misalignment
    rack_stop = 25.0
    hp12, meta12 = load_hardpoints(DATA / "hardpoints_v1.2.csv")
    _, w12, _ = misalignment.worst_angles(hp12, meta12, VEH, rack_stop)
    assert w12["Pushrod lower rod end"][0] > 18.0          # beyond the AM-M8 clevis rating
    hp, meta = load_hardpoints()
    _, w, _ = misalignment.worst_angles(hp, meta, VEH, rack_stop)
    assert w["Pushrod lower rod end"][0] < 5.0
