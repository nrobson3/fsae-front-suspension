"""Step D (PLAN s17): the DFMEA's coverage rule and the bar-swap ARB (D26).

Run scripts/run_all.py first: the coverage test reads the generated matrix and fastener sweep.
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dfmea  # noqa: E402
import handcalcs  # noqa: E402
from common import DATA, load_materials, load_vehicle, read_version  # noqa: E402

FM = json.loads((DATA / "dfmea.json").read_text(encoding="utf-8"))
REQS = {it["id"] for g in json.loads((DATA / "requirements.json").read_text())["groups"] for it in g["items"]}


def all_pass(**override):
    """A matrix where every requirement passes, except the overrides {id: status}."""
    return {q: (override.get(q.replace("-", "_"), "PASS"), "1.0") for q in REQS}


def test_dfmea_covers_every_requirement_not_passing_and_every_fastener_warning():
    ver, mat = dfmea.matrix()
    assert ver == read_version(DATA / "hardpoints.csv")
    veh = json.loads((DATA / "vehicle_derived.json").read_text())
    hc = json.loads((DATA / "handcalc_results.json").read_text())
    assert dfmea.check(FM, REQS, mat, dfmea.warn_items(), veh, hc) == []
    assert any(s.startswith("DEVIATION") for s, _ in mat.values())      # the rule is exercised, not vacuous
    assert dfmea.warn_items()                                          # and so is the warning rule


def test_dfmea_catches_a_deviation_nobody_closes():
    fm = copy.deepcopy(FM)
    for x in fm["rows"] + fm["tests"]:
        x["closes"].pop("S-06f", None)
    bad = dfmea.check(fm, REQS, all_pass(S_06f="DEVIATION DV-2"), [])
    assert len(bad) == 1 and "S-06f" in bad[0]


def test_dfmea_refuses_a_matrix_missing_a_requirement():
    mat = all_pass()
    mat.pop("S-23")
    assert any("S-23" in b and "not in the matrix" in b for b in dfmea.check(FM, REQS, mat, []))


def test_only_exact_pass_or_retired_counts_as_passing():
    bad = dfmea.check(FM, REQS, all_pass(S_01="PASS: provisional", S_06a="STALE (re-run scripts/fea/upright_fea.py)"), [])
    assert any("S-01" in b for b in bad) and any("S-06a" in b for b in bad)


def test_dfmea_catches_an_uncovered_warning_and_a_pattern_with_no_closing_entry():
    assert any("Wheel nut" in b for b in dfmea.check(FM, REQS, all_pass(), ["L Wheel nut (M12)"]))
    fm = copy.deepcopy(FM)
    row = next(r for r in fm["rows"] if r.get("covers_warn"))
    row["closes"].pop(f"WARN: {row['covers_warn'][0]}")
    assert any("does not say what closes" in b for b in dfmea.check(fm, REQS, all_pass(), []))


def test_dfmea_rejects_bad_ratings():
    fm = copy.deepcopy(FM)
    rows = {r["id"]: r for r in fm["rows"]}
    rows["F-01"]["after"]["D"] = rows["F-01"]["D"] + 1                   # an action that makes detection worse
    rows["F-02"]["actions"] = []                                         # severity 9 with no action, ratings lowered
    rows["F-03"]["O"] = 0
    rows["F-04"]["after"]["S"] = 2                                       # an action does not change severity
    bad = dfmea.check(fm, REQS, all_pass(), [])
    for frag in ("raises D", "with no action", "1-10", "severity (only", "lowered with no action"):
        assert any(frag in b for b in bad), frag


def test_placeholders_resolve_or_fail():
    mat = all_pass()
    assert dfmea.fill("FoS {req:S-06a} and {hc:x:.2f}", mat, {}, {"x": 1.234}) == ("FoS 1.0 and 1.23", [])
    assert dfmea.fill("{veh:missing}", mat, {}, {})[1] == ["{veh:missing}"]
    fm = copy.deepcopy(FM)
    fm["rows"][0]["cause"] = "margin {req:NOPE-1}"
    assert any("unresolved" in b for b in dfmea.check(fm, REQS, mat, [], {}, {}))


def test_rpn_after_actions_never_exceeds_before():
    for r in FM["rows"]:
        assert dfmea.rpn(r["after"]) <= dfmea.rpn(r), r["id"]
    assert max(dfmea.rpn(r) for r in FM["rows"]) > max(dfmea.rpn(r["after"]) for r in FM["rows"])


def test_arb_bar_static_and_fatigue_closed_forms():
    mats = load_materials()
    veh = load_vehicle()
    ab = handcalcs.arb_ends(mats, veh)
    T, Tr = ab["T_Nm"] * 1000, ab["Tr_Nm"] * 1000
    assert ab["tau_body"] == pytest.approx(16 * T / (math.pi * ab["D"] ** 3))
    assert ab["tau_end"] == pytest.approx(ab["kt"] * 16 * T / (math.pi * ab["Dm"] ** 3))
    assert ab["tau_body"] == pytest.approx(ab["tau_body_solver"], rel=1e-6)          # the vehicle calcs agree
    assert ab["D"] < ab["Dm"] < veh["arb_bar"]["end_od_mm"]                           # upset ends: spline above the body
    assert ab["fos_end"] * (ab["dm_need"] / ab["Dm"]) ** 3 == pytest.approx(1.5)      # dm_need gives exactly FoS 1.5
    assert ab["tau_a_sh"] == pytest.approx(ab["kt_sh"] * 16 * Tr / (math.pi * ab["D"] ** 3))
    sut = ab["Ftu"]
    ke = json.loads((DATA / "checks_week3b.json").read_text())["arb_bar_fatigue"]["reliability_ke"]
    assert ke == pytest.approx(0.814)                                                  # as the upright's fatigue
    assert ab["se_sh"] == pytest.approx(1.58 * sut ** -0.085 * 1.24 * ab["D"] ** -0.107 * 0.59 * ke * 0.5 * sut)
    assert 1.0 < ab["kt_sh"] < handcalcs.kt_torsion(1.0 / ab["D"], 2.0)               # a bigger fillet lowers Kt


def test_the_fitted_bar_is_one_of_the_three_and_the_lever_sits_beside_the_rod_end():
    bar = load_vehicle()["arb_bar"]
    assert bar["fitted"] in bar["bars_mm"] and set(bar["bars_mm"]) == set(bar["settings"])
    assert sorted(bar["bars_mm"].values()) == [bar["bars_mm"][k] for k in sorted(bar["settings"], key=bar["settings"].get)]
    sp = json.loads((DATA / "upright_design.json").read_text())["clevis"]["spacer_mm"]
    assert bar["lever_offset_mm"] == pytest.approx(6.0 + sp + bar["lever_blade_t_mm"] / 2)


def test_the_stored_fea_and_fatigue_match_their_inputs():
    # 2026-11-18: one new material entry in materials.json (which the FEA's fingerprint hashes whole) turned every FEA
    # item STALE without changing the FEA. Any change to the FEA's inputs must be deliberate: re-run the FEA then.
    import fingerprint
    fea = json.loads((DATA / "upright_fea.json").read_text())
    fat = json.loads((DATA / "upright_fatigue.json").read_text())
    assert fea["_inputs_sha"] == fingerprint.fea_inputs_hash(), "FEA inputs changed: re-run scripts/fea/upright_fea.py"
    assert fat["_inputs_sha"] == fingerprint.fatigue_inputs_hash(), "re-run scripts/fea/upright_fatigue.py"


# ---- step E: the write-up is generated, every result from an output (PLAN s17) ----
import writeup  # noqa: E402


def test_writeup_matches_its_template_and_the_outputs():
    import requirements_matrix
    assert requirements_matrix.is_current()[0] and dfmea.is_current()[0], "run run_all.py"
    out, missing = writeup.fill((ROOT / "notes" / "writeup_template.md").read_text(encoding="utf-8"), writeup.sources())
    assert missing == [] and writeup.stray_braces(out) == [] and writeup.missing_links(out)[1] == []
    body = (ROOT / "WRITEUP.md").read_text(encoding="utf-8").split("\n", 1)[1]
    assert body == out, "WRITEUP.md is stale or edited by hand: run scripts/writeup.py"


def test_render_looks_straight_down_without_failing():
    import render
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    img = render.render(V, F, (0.7, 0.7, 0.7), eye=(0.3, 0.3, 10.0), target=(0.3, 0.3, 0.0), width=64, height=48, ssaa=1)
    a = np.asarray(img, float)
    assert np.isfinite(a).all() and a.std() > 0                    # something was drawn, nothing NaN


def test_writeup_fill_refuses_unknown_placeholders():
    src = {"req": {"S-01": "3.23 (worst: LCA rear leg)"}, "veh": {"x": 0.5}}
    assert writeup.fill("{req:S-01} {veh:x:.0%}", src) == ("3.23 50%", [])
    assert writeup.fill("{req:S-99} {veh:nope} {zzz:a}", src)[1] == ["{req:S-99}", "{veh:nope}", "{zzz:a}"]


def test_setup_sensitivities_match_closed_forms():
    """The setup table comes from the kinematic solver; two closed forms, independent of it, check it."""
    from scipy.optimize import fsolve
    from common import load_hardpoints, unit
    hp, meta = load_hardpoints()
    _, members = writeup._csv(DATA / "members.csv")
    s = writeup.setup_sensitivities(hp, meta, 0.93, members, load_vehicle()["tyre"]["loaded_radius_mm"])
    p = next(float(r["rod_end_pitch_mm"]) for r in members if r["member"] == "UCA front leg")
    assert p == next(float(r["rod_end_pitch_mm"]) for r in members if r["member"] == "Tie rod")
    # toe: the tie rod's length change turns the upright about the kingpin axis k through the tie rod's own moment arm
    # (k x t) . u (u along the tie rod: it is not square to the steering arm), and toe is that turn's vertical part
    k = unit(hp["UCA_O"] - hp["LCA_O"])
    arm = abs(np.dot(np.cross(k, hp["TR_O"] - hp["LCA_O"]), unit(hp["TR_O"] - hp["TR_I"])))
    assert s["toe_per_turn"] == pytest.approx(math.degrees(2 * p / arm) * abs(k[2]), rel=0.01)
    # camber: with the lower ball joint held (the spring perch left alone holds the lower wishbone), the upper ball joint
    # moves to where the two longer legs meet, a ball-joint spacing from the lower one (three spheres); once the toe is
    # reset the wheel's camber change is the kingpin line's turn in the front view (code review, step E)
    F, R, O, L = hp["UCA_F"], hp["UCA_R"], hp["UCA_O"], hp["LCA_O"]
    lf, lr, lk = (float(np.linalg.norm(O - q)) for q in (F, R, L))
    O2 = fsolve(lambda q: [np.linalg.norm(q - F) - (lf + p), np.linalg.norm(q - R) - (lr + p), np.linalg.norm(q - L) - lk], O)
    front = lambda v: math.degrees(math.atan2(v[1], v[2]))       # noqa: E731
    assert s["camber_per_turn"] == pytest.approx(abs(front(O2 - L) - front(O - L)), rel=0.01)
    assert 0 < s["camber_toe_per_turn"] < s["toe_per_turn"] and 0 < s["camber_height_per_turn"] < p
    assert s["ride_per_mm"] == pytest.approx(1 / 0.93)


def test_writeup_fill_keeps_counts_whole_and_flags_stray_braces():
    out, missing = writeup.fill("{fast:checks} checks, {veh:x} N, {req: S-01}", {"fast": {"checks": 1010}, "veh": {"x": 1234.5}})
    assert out == "1010 checks, 1.23e+03 N, {req: S-01}" and missing == []     # the spaced one is not a placeholder...
    assert writeup.stray_braces(out) == [out]                                       # ...but its braces stop the run


def test_writeup_links_need_the_exact_file(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("x")
    text = "[a](notes/a.md#s6) [b](notes/A.md) [c](notes/gone.md#x) [d](https://x.org/y) [e](#top)\n[r]: notes/old.md\n"
    _, bad = writeup.missing_links(text, tmp_path)
    assert bad == ["notes/A.md", "notes/gone.md#x", "notes/old.md"]


def test_dfmea_fill_reads_exponents_formats_req_and_flags_other_placeholders():
    mat = {"S-06h": ("PASS", "8e-05 % (ring)"), "S-01": ("PASS", "3.234 (LCA rear leg)")}
    out, missing = dfmea.fill("{req:S-06h} {req:S-01:.1f} {fea:fos_yield}", mat, {}, {})
    assert out.startswith("8e-05 3.2 ") and len(missing) == 1 and missing[0].startswith("stray brace")


def test_writeup_trade_frontier_reads_the_studies(tmp_path):
    t = writeup.trade_frontier()
    assert t["mig_min"] <= t["mig_max"] and t["rec_min"] < t["rec_max"]
    assert t["per_10mm"] > 0                                       # wider upper pickups lower the migration
    c, w = tmp_path / "c.csv", tmp_path / "w.csv"
    c.write_text("# test\nrecovery_pct,rc_migration_pm25_mm,UCA_in_y\n40,30,250\n80,20,270\n")
    w.write_text("# test\nUCA_in_y,rc_migration_ground_pm25_mm\n200,40\n300,30\n")
    t = writeup.trade_frontier(c, w)
    assert (t["mig_min"], t["mig_max"], t["uca_y_max"]) == (20, 30, 270) and t["per_10mm"] == pytest.approx(1.0)
