"""Week 3b regression tests: the duty spectrum and Miner's rule, the BOM, the hand-calc answer keys, the
requirements matrix's hand-calc source and the upright drawing.

Run from the project root:  python -m pytest tests -q

Formulas are checked against closed forms or limits; the generated files (duty_loads.json, bom_costed.csv,
the drawing's DXF) are read because they are the outputs under test.
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "fea"))

import bom  # noqa: E402
import handcalcs  # noqa: E402
import requirements_matrix  # noqa: E402
from common import DATA, load_materials  # noqa: E402
from upright_fatigue import damage, margin, sn_line  # noqa: E402

MATS = json.loads((DATA / "materials.json").read_text())
AL = MATS["7075_T7351_plate"]
VEH = json.loads((DATA / "vehicle.json").read_text())
DUTY = json.loads((DATA / "duty_loads.json").read_text())


# ---- S-N line and Miner's rule (upright_fatigue.py, reused by the hand calcs) ----
def test_sn_line_passes_through_its_two_construction_points():
    S1, b, Se = sn_line(AL)
    assert S1 == pytest.approx(AL["fatigue_f_1e3"] * AL["Ftu_MPa"])
    assert S1 * (5e8 / 1e3) ** b == pytest.approx(Se, rel=1e-12)
    assert b < 0


def test_miner_margin_matches_the_closed_form_for_one_fully_reversed_block():
    S1, b, _ = sn_line(AL)
    a, n = 80.0, 2.0e6
    k = margin(np.array([[a]]), np.zeros((1, 1)), np.array([n]), S1, b, AL["Ftu_MPa"])[0]
    assert k == pytest.approx(S1 / a * (n / 1e3) ** b, rel=1e-6)          # N(k a) = n
    assert damage(k, np.array([[a]]), np.zeros((1, 1)), np.array([n]), S1, b, AL["Ftu_MPa"])[0] == pytest.approx(1.0, rel=1e-5)


def test_miner_mean_stress_lowers_the_margin():
    S1, b, _ = sn_line(AL)
    amps, ns = np.array([[60.0]]), np.array([1e6])
    k0 = margin(amps, np.zeros((1, 1)), ns, S1, b, AL["Ftu_MPa"])[0]
    k1 = margin(amps, np.array([[100.0]]), ns, S1, b, AL["Ftu_MPa"])[0]
    assert k1 < k0


# ---- duty_loads.py: cycle bookkeeping and the free bodies ----
def test_duty_cycles_add_up_to_the_events_per_km_over_the_life():
    per_km = VEH["duty"]["cycle_model"]["events_per_km_assumed"]
    by_kind = {}
    for c in DUTY["cycles"]:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0.0) + c["n"]
    assert set(by_kind) == set(per_km)
    for kind, n in by_kind.items():
        assert n == pytest.approx(per_km[kind] * DUTY["life_km"], rel=1e-9), kind


def test_every_duty_state_is_a_balanced_free_body_of_the_upright():
    for name, st in DUTY["states"].items():
        F = sum(np.array(v["force_N"]) for k, v in st.items() if isinstance(v, dict) and "force_N" in v)
        assert np.abs(F).max() < 0.05, name                      # forces are stored to 0.01 N


# ---- bom.py ----
def test_bom_extended_costs_are_qty_times_unit_and_totals_exclude_quotes():
    with open(DATA / "bom_costed.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for r in rows:
        assert float(r["ext_usd"]) == pytest.approx(float(r["qty"]) * float(r["unit_usd"]), rel=2e-3, abs=0.02), r["id"]
    md = (ROOT / "notes" / "bom.md").read_text(encoding="utf-8")
    quotes = [r["id"] for r in rows if r["status"] == "quote"]
    assert quotes and all(q in md.split("Not in either total")[1].split("\n")[0] for q in quotes)


def test_bom_counts_one_rod_end_per_rod_end_joint_in_the_misalignment_table():
    assert bom.rod_end_count() == 10


def test_bom_tube_lengths_follow_the_member_lengths():
    members = bom.read_rows(DATA / "members.csv")
    checks = bom.read_rows(DATA / "member_checks.csv")
    feet = bom.tube_feet(members, checks)
    ends = {m["member"]: 1 if m["out"] in ("UCA_O", "LCA_O") else 2 for m in members}   # legs end in a welded node
    total_mm = sum(float(c["length_mm"]) for c in checks) \
        - sum(ends[m["member"]] * float(m["rod_end_shank_mm"]) for m in members) + bom.CUT_MM * len(members)
    assert sum(feet.values()) * 304.8 == pytest.approx(total_mm, rel=1e-9)


# ---- handcalcs.py ----
def test_kt_fit_falls_with_the_fillet_radius_and_rises_with_the_step():
    assert handcalcs.kt_shoulder(0.05, 1.2) < handcalcs.kt_shoulder(0.025, 1.2)
    assert handcalcs.kt_shoulder(0.05, 1.2) < handcalcs.kt_shoulder(0.05, 2.0)
    assert 1.0 < handcalcs.kt_shoulder(0.3, 2.0) < handcalcs.kt_shoulder(0.3, 2.0, flat=True)
    assert handcalcs.kt_shoulder(0.025, 1.15) == handcalcs.kt_shoulder(0.025, 1.2)        # rounds D/d up


def test_curved_beam_tends_to_the_straight_beam_for_a_large_radius():
    M, h, t = 1.0e4, 16.0, 4.0
    straight = 6 * M / (t * h ** 2)
    assert handcalcs.winkler_inner(M, h, t, 1e5) == pytest.approx(straight, rel=1e-3)
    assert handcalcs.winkler_inner(M, h, t, 3.0) > 1.5 * straight


def test_rocker_moment_balance_matches_the_force_solver():
    rk = handcalcs.rocker_lc1(MATS)
    assert rk["f_d"] == pytest.approx(rk["f_d_solver"], rel=1e-9)
    assert rk["resid_B"] < 1e-6 * abs(rk["f_d"])                    # the truss closes at the damper joint
    assert np.linalg.norm(rk["R"]) == pytest.approx(rk["R_solver"], rel=0.02)


def test_spindle_section_modulus_and_moment_closed_forms():
    sp = handcalcs.spindle(MATS, VEH)
    do, di = handcalcs.SPINDLE["do"], handcalcs.SPINDLE["di"]
    assert sp["Z"] == pytest.approx(math.pi * (do ** 4 - di ** 4) / (32 * do))
    st = next(s for s in sp["states"] if s["state"] == "corner 1.4 g, outside")    # 3D, as the static cases
    _, Fy, Fz = DUTY["states"]["corner 1.4 g, outside"]["tyre_N"]
    assert st["M_Nm"] * 1000 == pytest.approx(abs(-sp["a_sh"] * Fz + sp["R"] * Fy), rel=0.03)
    assert sp["used"] == pytest.approx(1.0, rel=1e-9)
    lc2 = next(s for s in sp["static"] if s[0] == "LC2")           # 3D, with camber and the pneumatic trail
    Fy, Fz = -2029.6, 1268.5                                       # LC2 in data/loadcases.csv
    assert lc2[1] * 1000 == pytest.approx(abs(-sp["a_sh"] * Fz + sp["R"] * Fy), rel=0.03)


def test_hand_calc_results_feed_the_matrix_and_pass():
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    src = {"hc": (hc["_hardpoints"], hc)}
    for key in ("rocker_fos_min", "ear_roots_fos_min", "spindle_fatigue_k"):
        txt, status = requirements_matrix.evaluate({"hc": key, "lo": 1.5}, src)
        assert status == "PASS", (key, txt)
    assert abs(hc["lc1_pushrod_hand_N"] / hc["lc1_pushrod_3d_N"] - 1) < 0.02


# ---- the upright drawing ----
def test_drawing_dxf_carries_the_number_the_datums_and_every_hole_tag():
    ezdxf = pytest.importorskip("ezdxf")
    dxf = ROOT / "drawings" / "FS1-UPR-001.dxf"
    if not dxf.exists():
        pytest.skip("drawing not generated")
    texts = {e.dxf.text for e in ezdxf.readfile(str(dxf)).modelspace().query("TEXT")}
    assert "FS1-UPR-001" in texts
    assert {"A", "B", "C"} <= texts
    assert {"C", "H1", "H2", "H3", "H4", "H5", "H6", "H7"} <= texts


# ---- the code review's fixes (2026-10-27) ----
class _StubRay:
    def intersects_location(self, origins, dirs):
        return np.asarray(origins) + np.asarray(dirs) * 10.0, None, None     # every ray hits the body 10 mm away


class _StubMesh:
    ray = _StubRay()


def _stub_ear():
    H = {"X": {"centre": [0.0, 0.0, 10.0], "axis": [0.0, 0.0, 1.0], "radius": 4.0},
         "Y": {"centre": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "radius": 4.0}}
    return handcalcs.ear_model(_StubMesh(), H, "X", "Y", 30.0, 6.0, UD)


UD = json.loads((DATA / "upright_design.json").read_text())


def test_ear_fatigue_cycle_keeps_the_sign_of_a_reversing_load():
    em = _stub_ear()
    A, B = em["stresses"]([500.0, 0.0, 0.0]), em["stresses"]([-500.0, 0.0, 0.0])
    amp, mean = em["cycle"](A, B)
    assert amp == pytest.approx(A["s_root"]) and mean == pytest.approx(0.0, abs=1e-9)   # full reversal, zero mean
    amp, mean = em["cycle"](A, A)
    assert amp == pytest.approx(0.0, abs=1e-9) and mean == pytest.approx(A["s_root"])


def test_ear_load_along_the_bolt_goes_only_to_the_ear_it_pushes():
    em = _stub_ear()                                                  # ear X is the upper one
    assert em["upper"]
    assert em["stresses"]([0.0, 0.0, 800.0])["P_ax"] == pytest.approx(800.0)
    assert em["stresses"]([0.0, 0.0, -800.0])["P_ax"] == 0.0


def test_ear_roots_refuse_a_missing_sharp_edge_list():
    with pytest.raises(RuntimeError):
        handcalcs.ear_roots({"faces": {"holes": []}}, {}, UD, MATS, None, 1.0)


def test_kt_takes_the_largest_fit_up_to_the_next_tabulated_step():
    A11, b11 = handcalcs.KT_FIT[1.1]
    assert handcalcs.kt_shoulder(0.025, 1.15) == pytest.approx(max(A11 * 0.025 ** b11, handcalcs.KT_FIT[1.2][0] * 0.025 ** handcalcs.KT_FIT[1.2][1]))


def test_matrix_marks_the_hand_calcs_stale_when_their_inputs_changed():
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    txt, status = requirements_matrix.evaluate({"hc": "rocker_fos_min", "lo": 1.5}, {"hc": (hc["_hardpoints"], hc), "hc_current": False})
    assert status.startswith("STALE")


# ---- S-18 fastener sweep and S-17 clevis joint (2026-11-03) ----
import fasteners  # noqa: E402
import trimesh  # noqa: E402

FAST = json.loads((DATA / "fasteners.json").read_text())


def test_fastener_stack_builds_outward_from_the_seat_face():
    face, out = np.zeros(3), np.array([0.0, 0.0, 1.0])
    bodies, (label, tool), tip = fasteners.end_bodies(face, out, "M8", "nut", FAST, role="turn")
    sz = FAST["sizes"]["M8"]
    zs = [(m.bounds[0][2], m.bounds[1][2]) for _, m in bodies]
    assert [b for b, _ in bodies] == ["washer (nut end)", "nut", "thread end"]
    assert zs[0][0] == pytest.approx(FAST["seat_gap_mm"], abs=1e-6)
    assert zs[1][1] - zs[1][0] == pytest.approx(sz["nut_h"], abs=1e-6)
    assert tip[2] == pytest.approx(FAST["seat_gap_mm"] + sz["washer_t"] + sz["nut_h"] + 2 * sz["pitch"])
    assert label == "socket" and tool.bounds[1][2] == pytest.approx(tip[2] + FAST["tool_length_mm"], abs=1e-6)


def test_held_end_gets_a_ring_spanner_round_the_hex():
    _, (label, tool), _ = fasteners.end_bodies(np.zeros(3), np.array([0.0, 0, 1.0]), "M8", "head", FAST, role="hold")
    r = (tool.bounds[1][0] - tool.bounds[0][0]) / 2
    assert label == "ring spanner"
    assert r == pytest.approx(FAST["ring_spanner_r_per_af"] * FAST["sizes"]["M8"]["head_af"], rel=0.01)


def test_insert_check_finds_a_plate_too_thin_for_the_insert():
    name = json.loads((DATA / "upright_design.json").read_text())["ball_joints"]["lower"]["insert"]
    part = FAST["inserts"][name]                                    # the insert the design names
    for thick, expect_out in ((30.0, False), (6.0, True)):
        block = trimesh.creation.box(extents=[60.0, 60.0, thick])
        top = np.array([0.0, 0.0, thick / 2])
        j = {"ends": [(top, np.array([0.0, 0, 1.0]), "insert", "hold")], "insert": {"ear": 0, "part": name}}
        bad, n = fasteners.insert_check(j, FAST, block.contains)
        assert (bad > 0) == expect_out, (thick, bad, n)
    assert part["length_mm"] > 6.0                                  # the 6 mm ear alone could never hold it


def test_step_classifier_on_the_upright():
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    H = {h["name"]: h for h in build["faces"]["holes"]}
    inside, vol = fasteners.step_solid(ROOT / "cad" / "upright" / "upright_L.step")
    c = np.array(H["LBJ1"]["centre"])
    assert vol == pytest.approx(build["masses_kg"]["upright"] / 2.81e-6, rel=0.01)   # 7075: 2.81 g/cm^3
    assert list(inside([c, c + [1e4, 0, 0]])) == [False, False]     # the hole and far away are not material


def test_right_side_fasteners_mirror_the_left():
    L, R = fasteners.joints("L"), fasteners.joints("R")
    assert [j["name"] for j in L] == [j["name"] for j in R]
    for jl, jr in zip(L, R):
        for (pl, ol, kl, rl), (pr, orr, kr, rr) in zip(jl["ends"], jr["ends"]):
            assert np.allclose(np.asarray(pr), np.asarray(pl) * [1, -1, 1])
            assert np.allclose(np.asarray(orr), np.asarray(ol) * [1, -1, 1]) and kl == kr and rl == rr


def test_fastener_body_names_are_unique_per_joint():
    # names key the sweep's results: two washers both called "washer" let one hide the other's contact
    for j in fasteners.joints("L"):
        names = []
        for fp, o, kind, role in j["ends"]:
            if kind != "insert":
                names += [n for n, _ in fasteners.end_bodies(np.asarray(fp), np.asarray(o), j["size"], kind, FAST, role)[0]]
        assert len(names) == len(set(names)), (j["name"], names)


def test_sweep_clear_fails_closed(tmp_path):
    p = tmp_path / "fastener_checks.csv"
    assert not fasteners.sweep_clear(p)[0]                          # missing
    stamp = "# hardpoints v1.7; cad 0000000000000000; inputs {}; poses 279; fails 0\n"
    head = stamp.format(fasteners.fingerprint.fastener_inputs_hash())
    cols = "check,item,against,min_mm,status,pose\n"
    ok_ins = "".join(f"{fasteners.INSERT_CHECK},{s} Lower ball joint,upright,144,OK,0 of 144\n" for s in "LR")
    p.write_text(head + cols, encoding="utf-8")
    assert not fasteners.sweep_clear(p)[0]                          # no rows
    p.write_text(stamp.format("0" * 16) + cols + ok_ins, encoding="utf-8")
    assert not fasteners.sweep_clear(p)[0]                          # made from other inputs
    p.write_text(head + cols + ok_ins.split("\n")[0] + "\n", encoding="utf-8")
    assert not fasteners.sweep_clear(p)[0]                          # one side's insert missing
    p.write_text(head + cols + ok_ins + "fastener body,L Caliper bolt 1,L Upright / upright,0.00,FAIL,x\n", encoding="utf-8")
    assert not fasteners.sweep_clear(p)[0]                          # a FAIL anywhere
    p.write_text(head + cols + ok_ins, encoding="utf-8")
    assert fasteners.sweep_clear(p)[0]


def test_fast_count_refuses_an_empty_sweep():
    with pytest.raises(RuntimeError):
        requirements_matrix.evaluate({"fast_count": "FAIL", "hi": 0}, {"fast": (None, [])})


def test_every_joint_turns_exactly_one_end():
    for j in fasteners.joints("L"):
        roles = [r for _, _, k, r in j["ends"] if k != "insert"]
        assert roles.count("turn") == 1, j["name"]


def test_upright_is_one_solid_with_the_insert_boss():
    build = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())
    H = {h["name"]: h for h in build["faces"]["holes"]}
    ud = json.loads((DATA / "upright_design.json").read_text())
    assert set(build["masses_kg"]) == {"upright"}                  # no bolted clevis, no shims (D20)
    assert H["LBJ1"]["boss_mm"] == ud["clevis"]["lbj_insert_boss"]["height_mm"]
    assert H["LBJ1"]["radius"] == pytest.approx(ud["clevis"]["lbj_insert_boss"]["tap_drill_mm"] / 2)
    assert 2 * H["LBJ1"]["half_len"] + H["LBJ1"]["boss_mm"] >= FAST["inserts"][H["LBJ1"]["insert"]]["length_mm"]


# ---- main spring on the real catalogue (D21, 2026-11-04) ----
def test_spring_is_a_rate_that_is_sold():
    # v1 picked 25 N/mm from an assumed metric list nobody sells for the TTX25; the pick must come from the sourced list
    v = json.loads((DATA / "vehicle.json").read_text(encoding="utf-8"))
    d = json.loads((DATA / "vehicle_derived.json").read_text(encoding="utf-8"))
    assert "spring_steps_N_per_mm_assumed" not in v
    assert v["spring_catalogue"]["source"].startswith("https://")
    assert any(d["spring_chosen_lbf_in"] == pytest.approx(s) for s in v["spring_steps_lbf_in"])


# ---- step B checks (PLAN s17, 2026-11-04) ----
import fingerprint  # noqa: E402
import week3b_checks as W3  # noqa: E402


def test_miner_factor_is_one_on_the_line():
    k, D = W3.miner_k([(63.0, 2e6)], 63.0, 3.0, 2e6)
    assert k == pytest.approx(1.0) and D == pytest.approx(1.0)
    assert W3.miner_k([(126.0, 2e6)], 63.0, 3.0, 2e6)[0] == pytest.approx(0.5)     # twice the range, half the factor
    assert W3.miner_k([], 63.0, 3.0, 2e6)[0] == math.inf


def test_disc_temperature_balances_its_power():
    assert W3.disc_equilibrium(1000.0, 0.03, 60.0, 0.0, 30.0) == pytest.approx(30.0 + 1000.0 / (60.0 * 0.03))
    T = W3.disc_equilibrium(1700.0, 0.0335, 60.0, 0.6, 30.0) + 273.15
    assert 60.0 * 0.0335 * (T - 303.15) + 0.6 * W3.SIGMA_SB * 0.0335 * (T ** 4 - 303.15 ** 4) == pytest.approx(1700.0)
    assert W3.disc_equilibrium(0.0, 0.03, 60.0, 0.6, 30.0) == 30.0


def test_brake_heat_mount_margins_and_matrix_action():
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    txt, status = requirements_matrix.evaluate(
        {"hc": "s16_mount_margin", "lo": 1.0, "action": "x"}, {"hc": (hc["_hardpoints"], hc), "hc_current": True})
    assert status == ("PASS" if hc["s16_mount_margin"] >= 1.0 else "ACTION: x")
    assert hc["s16_mount_margin"] > hc["s16_hat_margin"]                          # the floating disc resists more than v1.7's hat
    assert hc["s16_hat_margin"] < 1.0 <= hc["s16_mount_margin"]                   # why D23 changed it, and that it holds


def test_free_play_from_the_inputs():
    from common import load_vehicle
    C = json.loads((DATA / "checks_week3b.json").read_text(encoding="utf-8"))["free_play"]
    veh = load_vehicle()
    fp = W3.free_play(veh)
    play = C["rack_lash_mm_assumed"] + C["rod_ends_in_chain"] * C["rod_end_play_mm_assumed"]
    c = veh["steering"]["rack_c_factor_mm_per_rev_assumed"]
    assert fp["deg_total"] == pytest.approx(play * 360 / c + C["column_allocation_deg_assumed"])


def test_fea_benchmarks_pass_on_the_current_method():
    bench = json.loads((DATA / "fea_benchmarks.json").read_text())
    assert bench["_inputs_sha"] == fingerprint.bench_inputs_hash(), "re-run scripts/fea/benchmark_peaks.py"
    assert bench["ok"] and bench["hole_kt_err_abs_pct"] <= 3.0 and bench["ring_err_abs_pct"] <= 1.0
    assert bench["ring"]["max_support_reaction_N"] < 1e-3 * bench["ring"]["load_N"]   # 3-2-1 holds a balanced load


# ---- step C close-out: R-08 drop link's lever end, R-09 locking and the bolt schedule, R-10 (2026-11-08) ----
import misalignment  # noqa: E402
from common import load_hardpoints  # noqa: E402


def test_bolt_schedule_is_the_shortest_length_with_two_threads_past_each_nut():
    for side in ("L", "R"):
        for j in fasteners.joints(side):
            if not any(k == "nut" for _, _, k, _ in j["ends"]):
                continue
            sz = FAST["sizes"][j["size"]]
            s = fasteners.schedule(j, FAST)
            assert 2 * sz["pitch"] - 1e-9 <= s["past_mm"] < 2 * sz["pitch"] + sz["length_step_mm"], j["name"]
            assert s["length_mm"] / sz["length_step_mm"] == pytest.approx(round(s["length_mm"] / sz["length_step_mm"]))


def test_tapped_holes_engage_two_diameters_or_a_bought_parts_whole_thread():
    js = {j["name"]: j for j in fasteners.joints("L")}
    cal = js["Caliper bolt 1 (M8 socket head)"]
    s = fasteners.schedule(cal, FAST)
    ear = cal["tap_depth_mm"]
    assert ear < 2 * FAST["sizes"]["M8_SHCS"]["d"]                     # the caliper's ear is thinner than 2 d
    assert s["engage_need_mm"] == pytest.approx(ear) and s["engage_mm"] >= ear
    assert s["tip_past_mm"] == pytest.approx(s["engage_mm"] - ear) and s["tip_past_mm"] < FAST["sizes"]["M8_SHCS"]["pitch"]
    assert 25.0 - cal["grip_mm"] < 8.0           # the M8x25 the BOM listed until step C engaged under 1 d
    for name, j in js.items():
        if name.startswith("Rack bracket bolt"):
            assert fasteners.schedule(j, FAST)["engage_mm"] >= 2 * FAST["sizes"][j["size"]]["d"]
            assert j.get("interface")                                    # the block's thread depth is Kaz's


def test_a_blind_hole_too_shallow_for_the_bolt_fails_its_row():
    j = {"name": "x", "size": "M8", "ends": [((0, 0, 0), (0, 0, 1), "head", "turn")], "grip_mm": 4.0, "tap_depth_mm": 6.0}
    s = fasteners.schedule(j, FAST)
    assert s["engage_need_mm"] == pytest.approx(6.0) and s["engage_mm"] > 6.0      # 5 mm steps overshoot a 6 mm hole
    assert not j.get("tap_through")                                                  # main() marks such a bolt FAIL


def test_every_joint_is_positively_locked_and_an_unlocked_one_is_caught():
    for side in ("L", "R"):
        for j in fasteners.joints(side):
            assert fasteners.locking(j, FAST) in FAST["locking_allowed"], j["name"]
    tapped = {"name": "x", "size": "M8", "ends": [((0, 0, 0), (0, 0, 1), "head", "turn")], "grip_mm": 4.0}
    assert fasteners.locking(tapped, FAST) is None                 # a bolt into a part must say how it is locked
    ins = json.loads(json.dumps(FAST))
    ins["inserts"]["1191-6CN375"].pop("locking_version")
    lbj = next(j for j in fasteners.joints("L") if j.get("insert"))
    assert fasteners.locking(lbj, ins) is None                     # a free-running insert does not lock


def test_drop_link_lever_end_sits_beside_the_blade_captured():
    hp, _ = load_hardpoints()
    bar = json.loads((DATA / "vehicle.json").read_text())["arb_bar"]
    sp = json.loads((DATA / "upright_design.json").read_text())["clevis"]["spacer_mm"]
    assert bar["lever_offset_mm"] == pytest.approx(6.0 + sp + bar["lever_blade_t_mm"] / 2)
    j = next(j for j in fasteners.joints("L") if j["name"].startswith("Drop link lever end"))
    ax = (hp["ARB_A"] - hp["ARB_P"]) / np.linalg.norm(hp["ARB_A"] - hp["ARB_P"])
    (head, o1, k1, _), (nut, o2, k2, _) = j["ends"]
    assert (k1, k2) == ("head", "nut") and j.get("capture")
    assert np.dot(head - hp["ARB_L"], ax) == pytest.approx(6.0 + sp)                        # outer spacer, then the washer
    assert np.dot(nut - hp["ARB_L"], ax) == pytest.approx(-(bar["lever_offset_mm"] + bar["lever_blade_t_mm"] / 2))
    assert j["grip_mm"] == pytest.approx(12.0 + 2 * sp + bar["lever_blade_t_mm"])
    axis = next(d[4] for d in misalignment.joint_defs(hp) if d[0] == "Drop link lever end")
    assert abs(np.dot(axis, ax)) == pytest.approx(1.0)                                       # bolt along the bar's axis


def test_arb_lever_joint_closed_forms():
    from common import load_vehicle
    hp, _ = load_hardpoints()
    vder = json.loads((DATA / "vehicle_derived.json").read_text())
    mats = load_materials()
    ud = json.loads((DATA / "upright_design.json").read_text())
    r = W3.arb_lever_joint(vder, load_vehicle(), ud, mats, hp)
    assert r["arm"] == pytest.approx(np.linalg.norm(hp["ARB_L"] - hp["ARB_P"]))   # lever square to the link at static
    assert r["F"] * r["arm"] == pytest.approx(vder["arb_bar_torque_max_Nm"] * 1000)
    assert r["s_b"] == pytest.approx(r["F"] * r["a_b"] / (math.pi * r["ds"] ** 3 / 32))
    assert r["s_lever"] == pytest.approx(6 * r["F"] * r["L_root"] / (r["t"] * r["w"] ** 2))
    assert r["fos_lever"] == pytest.approx(r["lever_Fty"] / math.hypot(r["s_lever"], math.sqrt(3) * r["tau_lever"]))


def test_matrix_check_filter_takes_a_list_and_bom_lines_need_a_quantity():
    rows = [{"check": "a", "status": "OK"}, {"check": "b", "status": "FAIL"}, {"check": "c", "status": "FAIL"}]
    ev = requirements_matrix.evaluate
    assert ev({"fast_count": "FAIL", "check": ["a", "b"], "hi": 0}, {"fast": (None, rows)})[1] == "FAIL"
    assert ev({"fast_count": "FAIL", "check": "a", "hi": 0}, {"fast": (None, rows)})[1] == "PASS"
    bom_rows = [{"id": "J8", "qty": "10"}, {"id": "J9", "qty": "0"}]
    txt, status = ev({"bom_lines": ["J8", "J9"], "lo": 2}, {"bom": bom_rows})
    assert status == "FAIL" and "J9" in txt
