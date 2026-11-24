"""Week 2 regression tests: the CAD-stage and Week 2 audit defects, so none can come back.

Run from the project root:  python -m pytest tests -q

Most checks are computed here from the hardpoints and data files (point-to-line distance, sampled
segment distances, the damper length from the solver poses, the torsion formula). Three inputs are
read from generated files, because recomputing them would duplicate whole scripts: the rack travel
at the steering stop and the ARB installation ratio (data/kinematics_summary.csv), and the
stiff-blade bar rate the balance target needs (data/vehicle_derived.json). Run scripts/run_all.py
first, so they match the data.
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import fcl
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import interference  # noqa: E402
import requirements_matrix  # noqa: E402
import rim_clearance  # noqa: E402
from common import DATA, cad_inputs_hash, load_hardpoints, load_materials, load_mesh, load_mesh_index, load_vehicle  # noqa: E402
from fingerprint import CODE_FILES, DATA_FILES  # noqa: E402
from suspension import make_corner  # noqa: E402

HP, HP_META = load_hardpoints()
VEH = load_vehicle()
MAT = json.loads((DATA / "materials.json").read_text())["4130N_tube"]
CORNER = make_corner(HP, HP_META, VEH["tyre"]["loaded_radius_mm"])
with open(DATA / "kinematics_summary.csv") as _f:
    KIN = {r.split(",")[0]: r.split(",")[1] for r in _f if not r.startswith("#") and "," in r}
RACK_STOP = float(KIN["rack_travel_at_stop"])


def seg_seg_distance(a0, a1, b0, b1, n=200):
    t = np.linspace(0.0, 1.0, n)[:, None]
    A, B = a0 + t * (a1 - a0), b0 + t * (b1 - b0)
    return float(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min())


# ---- v1.3: the pushrod's line of action must pass through the lower ball joint ----
def test_pushrod_line_passes_through_lower_ball_joint():
    d = HP["PR_I"] - HP["PR_O"]
    v = HP["LCA_O"] - HP["PR_O"]
    miss = np.linalg.norm(np.cross(v, d)) / np.linalg.norm(d)
    assert miss < 0.01, f"pushrod line misses the LBJ by {miss:.3f} mm: the two-force-member model no longer holds"


# ---- v1.5 (CAD sweep): rim lip vs wishbones and tie rod at full bump + lock at the stop ----
# (the gap falls about 2 mm per mm of rack beyond the stop, so the stop's tolerance is in the s6 stack-up)
def test_rim_lip_clears_the_links_over_heave_and_full_lock():
    half_w = VEH["wheel"]["rim_width_in"] * 25.4 / 2
    worst = rim_clearance.sweep(HP, HP_META, half_w, racks=np.linspace(-RACK_STOP, RACK_STOP, 9))
    tight = {k: round(v[0], 2) for k, v in worst.items() if v[0] < 5.0}
    assert not tight, f"rim clearance under 5 mm: {tight}"


# ---- v1.4: tie rod 4.6 mm from the lower wishbone leg ----
def test_tie_rod_clears_lower_wishbone_legs_over_heave_and_steer():
    r_tie = float(rim_clearance.TUBE_R["tie"])
    r_lca = float(rim_clearance.TUBE_R["lca_F"])
    worst = math.inf
    for z in (-30.0, -15.0, 0.0, 15.0, 30.0):
        for rack in np.linspace(-RACK_STOP, RACK_STOP, 7):
            P = CORNER.pose(z, float(rack))["P"]
            for leg in ("LCA_F", "LCA_R"):
                d = seg_seg_distance(P["TR_O"], P["TR_I"], P["LCA_O"], P[leg]) - r_tie - r_lca
                worst = min(worst, d)
    assert worst >= 5.0, f"tie rod comes within {worst:.1f} mm of a lower wishbone leg"


# ---- Week 2 audits: bump travel ends at the rubber inside the checked envelope; droop meets the rule ----
def _wheel_travel_at_damper_length(length, lo, hi):
    """Bisection for the wheel travel z in [lo, hi] at which the damper eye-to-eye length equals `length`."""
    f = lambda z: CORNER.pose(z)["damper_len"] - length          # noqa: E731  (falls as z rises)
    assert f(lo) > 0 > f(hi), "the damper length is not bracketed: the search range is wrong"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_bump_travel_ends_at_the_rubber_between_the_rule_and_the_checked_envelope():
    dmp = VEH["damper"]
    z_stop = _wheel_travel_at_damper_length(dmp["compressed_mm"] + dmp["bump_rubber_solid_mm_assumed"], 0.0, 45.0)
    assert 25.0 <= z_stop <= 30.0, f"bump travel ends at {z_stop:.1f} mm (rule >= 25, swept envelope <= 30)"


def test_droop_to_the_damper_top_out_meets_the_rule():
    droop = -_wheel_travel_at_damper_length(VEH["damper"]["extended_mm"], -45.0, 0.0)
    assert droop >= 25.0, f"droop to the damper top-out is {droop:.1f} mm, FSAE V.3.1.1 needs 25"


# ---- Week 2 audits, re-made for D26: the bars set the ARB's rate, and the stiff bar survives full warp ----
def test_arb_bars_give_their_settings_and_the_stiff_bar_survives_full_warp():
    bar = VEH["arb_bar"]
    mats = load_materials()
    up = bar["lever_hub_half_len_mm"] + bar["upset_runout_mm"]                   # upset end, on to each hub centre
    body = 2 * (abs(HP["ARB_P"][1]) - bar["lever_offset_mm"] - up)              # the body between the upsets
    arm = math.hypot(HP["ARB_L"][0] - HP["ARB_P"][0], HP["ARB_L"][2] - HP["ARB_P"][2])   # the ARB axis runs along y
    lb = arm - (bar["end_od_mm"] / 2 + 0.1 + bar["lever_hub_wall_mm"])          # the lever bends from its hub out
    i_lev = bar["lever_blade_t_mm"] * bar["lever_blade_w_mm"] ** 3 / 12
    k_lever = 3 * mats[bar["lever_material"]]["E_MPa"] * i_lev * arm ** 2 / lb ** 3 / 1000
    gj = lambda d: bar["shear_modulus_MPa"] * math.pi * d ** 4 / 32           # noqa: E731
    k_req = json.loads((DATA / "vehicle_derived.json").read_text())["arb_bar_stiffness_stiff_Nm_per_deg"] * 180 / math.pi
    for name, f in bar["settings"].items():
        d = bar["bars_mm"][name]
        k_bar = 1 / (body / gj(d) + 2 * up / gj(bar["end_od_mm"])) / 1000                # N m/rad
        k_eff = 1 / (1 / k_bar + 2 / k_lever)                                             # bar and both levers in series
        assert abs(k_eff / k_req / f - 1) <= bar["setting_tolerance_pct"] / 100, name
    assert k_lever / (gj(bar["bars_mm"]["stiff"]) / body / 1000) >= bar["lever_stiffness_ratio_min"]
    # full warp: one wheel at full bump, the other at the droop stop; the bar twists ratio x the equivalent roll
    droop = -_wheel_travel_at_damper_length(VEH["damper"]["extended_mm"], -45.0, 0.0)
    roll = math.atan((VEH["wheel_travel_design_mm"]["bump"] + droop) / VEH["track_front_mm"])
    torque = k_req * abs(float(KIN["arb_twist_per_roll"])) * roll                    # N m, stiff setting
    tau = 16 * torque * 1000 / (math.pi * bar["bars_mm"]["stiff"] ** 3)
    allow = mats[bar["material"]]["Fty_MPa"] * 0.577
    assert allow / tau >= 1.5, f"ARB bar shear FoS {allow / tau:.2f} in full warp ({torque:.0f} N m)"


# ---- Week 2 audit: the sweep's joint rules are per body, never per part ----
def _body(component, body):
    side, part, key = interference.part_of(component)
    return {"id": f"{part}/{body}", "side": side, "key": key}


def test_designed_joint_does_not_excuse_the_rest_of_the_part():
    bodies = [_body("L Upright", "upright"), _body("L Lower wishbone", "node"), _body("L Lower wishbone", "legs"),
              _body("L Lower wishbone", "pushrod bracket"), _body("L Pushrod", "lower rod end")]
    moving, _ = interference.classify(bodies)
    jointed = {(bodies[i]["id"], bodies[j]["id"]): jt for i, j, jt in moving}
    assert jointed[("Upright/upright", "Lower wishbone/node")] is True
    assert jointed[("Upright/upright", "Lower wishbone/legs")] is False
    assert jointed[("Upright/upright", "Lower wishbone/pushrod bracket")] is False
    assert jointed[("Lower wishbone/pushrod bracket", "Pushrod/lower rod end")] is True
    assert jointed[("Lower wishbone/legs", "Pushrod/lower rod end")] is False


def test_rotating_wheel_is_checked_against_the_caliper_in_its_own_group():
    bodies = [_body("L Wheel (Braid 10 in, envelope)", "wheel"), _body("L Caliper (envelope)", "caliper")]
    _, intra = interference.classify(bodies)
    assert len(intra) == 1 and intra[0][2] == 3.0


def _fcl(vertices, faces):
    b = fcl.BVHModel()
    b.beginModel(len(vertices), len(faces))
    b.addSubModel(np.asarray(vertices, float), np.asarray(faces, np.int32))
    b.endModel()
    return fcl.CollisionObject(b, fcl.Transform())


def test_a_body_fully_inside_another_counts_as_a_collision():
    big, small = trimesh.creation.box((50, 50, 50)), trimesh.creation.box((5, 5, 5))
    d, _ = interference._check(_fcl(big.vertices, big.faces), _fcl(small.vertices, small.faces),
                               (np.asarray(big.vertices), big.faces), (np.asarray(small.vertices), small.faces))
    assert d == 0.0


# ---- Week 2 audit: the committed meshes match today's inputs, and key static clearances hold ----
def test_committed_meshes_match_the_inputs_and_key_in_wheel_clearances_hold():
    index = load_mesh_index()                       # raises if the meshes were built from other inputs
    by_id = {(v["component"], v["body"]): k for k, v in index.items()}

    def gap(a, b):
        ma, mb = (load_mesh(by_id[k], index[by_id[k]]) for k in (a, b))
        if fcl.collide(_fcl(ma.vertices, ma.faces), _fcl(mb.vertices, mb.faces), fcl.CollisionRequest(), fcl.CollisionResult()):
            return 0.0
        return fcl.distance(_fcl(ma.vertices, ma.faces), _fcl(mb.vertices, mb.faces), fcl.DistanceRequest(),
                            fcl.DistanceResult())

    assert gap(("L Wheel (Braid 10 in, envelope)", "wheel"), ("L Caliper (envelope)", "caliper")) >= 3.0
    assert gap(("L Hub, bearings, nut", "hub"), ("L Caliper (envelope)", "caliper")) >= 3.0
    assert gap(("L Upright", "upright"), ("L Lower wishbone", "pushrod bracket")) >= 5.0


def test_cad_fingerprint_changes_with_a_hardpoint_and_with_the_cad_code(tmp_path):
    for rel in DATA_FILES + CODE_FILES + ("data/vehicle.json",):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    base = cad_inputs_hash(tmp_path)
    assert base == cad_inputs_hash(ROOT)
    hp = tmp_path / "data" / "hardpoints.csv"
    hp.write_text(re.sub(r"^(DMP_C,chassis,[-0-9.]+,)([-0-9.]+)", lambda m: m.group(1) + str(float(m.group(2)) + 0.1),
                         hp.read_text(), flags=re.M))
    assert cad_inputs_hash(tmp_path) != base, "moving a hardpoint did not change the fingerprint"
    shutil.copy(ROOT / "data" / "hardpoints.csv", hp)
    code = tmp_path / "scripts" / "fusion" / "fsae_parts.py"
    code.write_text(code.read_text(encoding="utf-8") + "\n# edit\n", encoding="utf-8")
    assert cad_inputs_hash(tmp_path) != base, "editing the CAD builder did not change the fingerprint"


# ---- Week 3: the upright free body balances, and the hub bearings are not overloaded ----
def test_upright_free_body_balances_and_hub_bearings_have_static_margin():
    import forces
    import upright_loads as UL
    members = forces.read_csv_rows(DATA / "members.csv")
    C0, X0, Y0 = VEH["hub"]["C0_kN"] * 1000, VEH["hub"]["X0"], VEH["hub"]["Y0"]
    for lc in forces.read_csv_rows(DATA / "loadcases.csv"):
        pose = CORNER.pose(float(lc["z_pose_mm"]))
        res = UL.split(CORNER, pose, members, forces.tyre_loads(CORNER, pose, lc, VEH), VEH,
                       (lc.get("apply_at") or "cp") == "cp")
        assert res["resid"] < 1e-6, f"{lc['id']}: upright out of balance by {res['resid']:.1e}"
        s0 = min(UL.row_s0((res["row1"], res["row2"]), float(np.dot(res["axial"], res["s"])), C0, X0, Y0)[2])
        assert s0 >= 1.5, f"{lc['id']}: hub bearing static safety {s0:.2f}"


# ---- the requirements matrix: its status logic drives the traceability claim ----
def test_requirements_matrix_status_logic():
    src = {"veh": (None, {"x": 5.0})}
    assert requirements_matrix.evaluate({"veh": "x", "lo": 1, "hi": 10}, src)[1] == "PASS"
    assert requirements_matrix.evaluate({"veh": "x", "lo": 6}, src)[1] == "FAIL"
    assert requirements_matrix.evaluate({"veh": "x", "lo": 6, "deviation": "DV-9"}, src)[1] == "DEVIATION DV-9"
    assert requirements_matrix.evaluate({"veh": "x", "lo": 6, "action": "fix it"}, src)[1] == "ACTION: fix it"
    assert requirements_matrix.evaluate({"veh": "x", "lo": 1, "deviation": "DV-9"}, src)[1] == "PASS"
    assert requirements_matrix.evaluate({"open": "Week 3"}, src)[1] == "OPEN"
