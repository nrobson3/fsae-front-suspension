"""Week 3 regression tests: hub bearings, clevis bolts, the FEA's load application and its staleness guard.

Run from the project root:  python -m pytest tests -q

The formulas are checked against closed forms or against an independent recomputation from the data
files; bolt_checks.csv is read because it is the output under test. The FEA test meshes a small
holed block (a few seconds), not the upright.
"""
import csv
import io
import json
import math
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "fea"))

import bearings  # noqa: E402
import fingerprint  # noqa: E402
import requirements_matrix  # noqa: E402
from common import DATA, load_hardpoints, unit  # noqa: E402

HP, _ = load_hardpoints()
UD = json.loads((DATA / "upright_design.json").read_text())
UPL = json.loads((DATA / "upright_loads.json").read_text())


# ---- bearings.py: the press-fit and paired-bearing formulas ----
def test_lame_press_fit_matches_the_closed_form_for_a_solid_insert_of_the_same_material():
    E, nu, d, wall, delta = 71.7e3, 0.33, 68.0, 9.0, 0.03
    p, hoop = bearings.lame_hoop(delta, d, wall, d / 2, E, nu, Ec=E, nuc=nu)
    a, b = d / 2, d / 2 + wall
    assert p == pytest.approx(E * delta * (b ** 2 - a ** 2) / (2 * d * b ** 2), rel=1e-12)
    assert hoop == pytest.approx(p * (b ** 2 + a ** 2) / (b ** 2 - a ** 2), rel=1e-12)


def test_paired_tapered_rows_carry_the_external_thrust_and_at_least_their_induced_load():
    rng = np.random.default_rng(3)
    Y = json.loads((DATA / "vehicle.json").read_text())["hub"]["Y"]
    for _ in range(200):
        fr = rng.uniform(100, 12000, 2).tolist()
        ka = float(rng.uniform(-5000, 5000))
        fa = bearings.row_axial(fr, ka, Y)
        a, b = (1, 0) if ka < 0 else (0, 1)
        assert fa[a] - fa[b] == pytest.approx(abs(ka), abs=1e-9), "the rows' axial loads must balance the thrust"
        for i in range(2):
            assert fa[i] >= 0.5 * fr[i] / Y - 1e-9, "a row can never carry less than its own induced thrust"


# ---- bolts.py: the lug-pin model, and the load it uses is the worst over every case ----
def test_bolt_bending_follows_the_lug_pin_model_at_the_worst_load():
    import csv
    with open(DATA / "bolt_checks.csv", newline="") as f:
        rows = {r["joint"]: r for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    with open(DATA / "members.csv", newline="") as f:
        od = {r["member"]: float(r["tube_od_mm"]) for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    k = unit(HP["UCA_O"] - HP["LCA_O"])
    bj, ear, spacer = UD["ball_joints"], UD["clevis"]["ear_t_mm"], UD["clevis"]["spacer_mm"]
    for name, key, part, leg in (("Upper ball joint", "UBJ", bj["upper"], "UCA front leg"),
                                 ("Lower ball joint", "LBJ", bj["lower"], "LCA front leg")):
        r = rows[name]
        worst = max(np.linalg.norm(np.array(c[key]["force_N"]) - k * np.dot(c[key]["force_N"], k)) for c in UPL["cases"].values())
        assert float(r["load_N"]) == pytest.approx(worst, abs=0.6), f"{name}: not the worst load normal to the bolt"
        t_in = part["ball_w_in"] * 25.4
        gap = (od[leg] + 6.0 + 2 * spacer - t_in) / 2          # the wishbone node, spacers either side
        M = worst / 2 * (ear / 2 + gap + t_in / 4)
        assert float(r["moment_Nm"]) == pytest.approx(M / 1000, abs=0.01)
        sigma = M / (math.pi * part["bolt_d_mm"] ** 3 / 32)
        assert float(r["bending_MPa"]) == pytest.approx(sigma, abs=0.6)
        assert float(r["fos_bending"]) == pytest.approx(part["bolt_Fty_MPa"] / sigma, abs=0.01)


# ---- fea.py: pin and tapered-roller loads add up to exactly what was asked ----
@pytest.fixture(scope="module")
def holed_block(tmp_path_factory):
    import gmsh
    import fea
    g = fea._gmsh_start()
    box = gmsh.model.occ.addBox(-20, -20, -10, 40, 40, 20)
    hole = gmsh.model.occ.addCylinder(0, 0, -11, 0, 0, 22, 8.0)
    gmsh.model.occ.cut([(3, box)], [(3, hole)])
    gmsh.model.occ.synchronize()
    path = tmp_path_factory.mktemp("fea") / "block.msh"
    fea._gmsh_mesh(g, 4.0, path)
    return fea.Model(fea.load_mesh(path), 71.7e3, 0.33)


def _bore(x):
    return (np.abs(np.hypot(x[0], x[1]) - 8.0) < 0.6) & (np.abs(x[2]) < 10 - 0.01)


@pytest.mark.parametrize("force", [[1000.0, 0.0, 0.0], [300.0, -800.0, 0.0], [-500.0, 200.0, 400.0]])
def test_pin_bearing_load_resultant_equals_the_requested_force(holed_block, force):
    m = holed_block
    m.f[:] = 0.0
    m.bearing(_bore, [0, 0, 1], [0, 0, 0], force)
    assert np.allclose(m.applied_resultant(), force, atol=1e-6 * np.linalg.norm(force) + 1e-9)


def test_tapered_roller_traction_returns_its_axial_part_and_balances(holed_block):
    m = holed_block
    m.f[:] = 0.0
    force = np.array([0.0, 2000.0, 0.0])
    extra = m.bearing(_bore, [0, 0, 1], [0, 0, 0], force, incline_deg=11.0, incline_dir=[0, 0, 1])
    assert np.allclose(m.applied_resultant(), force + extra, atol=1e-3)
    # every roller's axial part is its radial part x tan(incline), and the radial parts of a +/-90 deg cosine
    # zone add up to 4/pi x the resultant: the same 1.25 Fr tan(alpha) as a tapered bearing's 0.5 Fr / Y
    assert extra[2] == pytest.approx(2000.0 * math.tan(math.radians(11.0)) * 4 / math.pi, rel=0.03)


def test_tapered_roller_traction_moves_the_line_of_action_by_R_tan_incline(holed_block):
    m = holed_block
    m.f[:] = 0.0
    m.bearing(_bore, [0, 0, 1], [0, 0, 0], [0.0, 2000.0, 0.0], incline_deg=11.0, incline_dir=[0, 0, 1])
    xyz = m.sb.doflocs
    f = np.stack([m.f[m.comp[i]] for i in range(3)])
    Mx = float(np.sum(xyz[1] * f[2] - xyz[2] * f[1]))
    # the tilt must put the row's line of action R tan(incline) along the axis (on its pressure centre)
    assert Mx == pytest.approx(2000.0 * 8.0 * math.tan(math.radians(11.0)), rel=0.03)


# ---- upright_fea.py: the post-processing functions ----
def test_goodman_amplitude_is_the_von_mises_of_the_tensor_range():
    import upright_fea as U
    S = np.zeros((3, 3, 1, 1))
    a, b = S.copy(), S.copy()
    a[0, 0] = 100.0                                    # proportional: uniaxial 100 <-> 0
    n = U.goodman(a, b, Se=100.0, Sut=400.0)
    assert n[0, 0] == pytest.approx(1 / (50 / 100 + 50 / 400))
    a, b = S.copy(), S.copy()
    a[0, 0], b[1, 1] = 100.0, 100.0                    # non-proportional: the scalar difference would give zero
    n = U.goodman(a, b, Se=100.0, Sut=400.0)
    amp = math.sqrt(50 ** 2 + 50 ** 2 + 50 ** 2)       # von Mises of diag(50, -50, 0)
    assert n[0, 0] == pytest.approx(1 / (amp / 100 + 50 / 400))


def test_rigid_rotation_recovers_a_small_rotation():
    import upright_fea as U
    rng = np.random.default_rng(1)
    P = rng.uniform(-50, 50, (40, 3))
    theta = np.array([1e-3, -2e-3, 5e-4])
    disp = np.array([0.1, 0.2, -0.3]) + np.cross(theta, P)
    assert np.allclose(U.rigid_rotation(P, disp), theta, atol=1e-12)


def test_region_masks_ball_and_housing_shell():
    import upright_fea as U
    faces = {"spin": [0.0, 1.0, 0.0], "wc": [0.0, 0.0, 0.0]}
    X = np.array([[0.0, 30.0, 50.0], [5.0, 50.0, 0.0], [0.0, 0.0, 0.0]])   # three points (columns)
    shell = U.region_mask(X, ("shell", 43.0), faces)
    assert shell.tolist() == [True, True, False]       # radius from the y axis: 0, 30, 50
    ball = U.region_mask(X, (np.array([0.0, 5.0, 0.0]), 1.0), faces)
    assert ball.tolist() == [True, False, False]


def test_richardson_recovers_a_known_order():
    import upright_fea as U
    r, p, f_exact = 1.4, 2.0, 200.0
    f = [f_exact - 30 / r ** (p * k) for k in range(3)]
    rich = U.richardson(f, r)
    assert rich["order"] == pytest.approx(p) and rich["extrapolated"] == pytest.approx(f_exact)
    assert U.richardson([100.0, 120.0, 110.0], r) is None
    assert U.richardson([166.0, 167.0, 170.0], 1.41) is None      # the change grows: not in the asymptotic range


# ---- the FEA result is tied to its inputs, and the matrix says so ----
def test_fea_fingerprint_changes_with_the_geometry_and_the_loads(tmp_path):
    for rel in fingerprint.FEA_FILES + ("data/vehicle.json",):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    base = fingerprint.fea_inputs_hash(tmp_path)
    assert base == fingerprint.fea_inputs_hash(ROOT)
    step = tmp_path / "cad" / "upright" / "upright_L_fea.step"
    step.write_bytes(step.read_bytes() + b"\n")
    assert fingerprint.fea_inputs_hash(tmp_path) != base, "a changed STEP did not change the FEA fingerprint"
    shutil.copy(ROOT / "cad" / "upright" / "upright_L_fea.step", step)
    loads = tmp_path / "data" / "upright_loads.json"
    loads.write_text(loads.read_text().replace("LC4", "LC4x", 1))
    assert fingerprint.fea_inputs_hash(tmp_path) != base, "changed loads did not change the FEA fingerprint"


def _floats(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _floats(v)
    elif isinstance(o, list):
        for v in o:
            yield from _floats(v)
    elif isinstance(o, float):
        yield o


def _floats_mapped(o, fn):
    """JSON data with fn applied to every float."""
    if isinstance(o, dict):
        return {k: _floats_mapped(v, fn) for k, v in o.items()}
    if isinstance(o, list):
        return [_floats_mapped(v, fn) for v in o]
    return fn(o) if isinstance(o, float) else o


def test_regenerated_inputs_are_fingerprinted_without_their_machine_digits(tmp_path):
    # 2026-11-23: run_all.py rewrites the interface loads, load cases, duty loads and bearing loads, and their last
    # digits depend on the machine (balance residuals near 1e-15, the sign of a zero rounded from noise, a bearing
    # load's 16th digit). On the Linux CI runner those must not mark the FEA or the fatigue check stale; a real
    # change still must.
    rels = fingerprint.FEA_FILES + ("data/vehicle.json", "data/duty_loads.json", "scripts/fea/upright_fatigue.py",
                                    "data/bearings.json")
    for rel in rels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    fea0, fat0 = fingerprint.fea_inputs_hash(tmp_path), fingerprint.fatigue_inputs_hash(tmp_path)
    assert (fea0, fat0) == (fingerprint.fea_inputs_hash(ROOT), fingerprint.fatigue_inputs_hash(ROOT))

    # a noise-sized number outside the residuals would carry the machine's digits through the 7-digit rounding
    for rel in ("data/upright_loads.json", "data/duty_loads.json", "data/bearings.json"):
        canon = json.loads(fingerprint.canonical(rel, (ROOT / rel).read_bytes()))
        tiny = [x for x in _floats(canon) if 0 < abs(x) < 1e-9]
        assert not tiny, f"{rel}: noise-sized numbers outside the residuals: {tiny[:3]}"

    def rewrite_json(rel, fn):
        p = tmp_path / rel
        p.write_text(json.dumps(fn(json.loads(p.read_text(encoding="utf-8"))), indent=2), encoding="utf-8")

    def rewrite_csv(fn):
        p = tmp_path / "data" / "loadcases.csv"
        rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8"))))
        out = io.StringIO()
        csv.writer(out, lineterminator="\r\n").writerows(fn(rows))
        p.write_text(out.getvalue(), encoding="utf-8", newline="")

    def one_ulp(x, toward=math.inf):     # what another machine writes: the other sign of zero, or one ulp off
        if not math.isfinite(x):
            return x
        return -x if x == 0 else math.nextafter(x, toward)

    def other_machine_cell(c):           # the CSV's decimals the same way (integers and text are exact anyway)
        if fingerprint._PLAIN_NUMBER.fullmatch(c) and any(ch in c for ch in ".eE"):
            return repr(one_ulp(float(c)))
        return c

    def other_machine(d):                # and other balance residuals
        for case in d["cases"].values():
            case["residual"] = case["residual"] * 3.7 + 1e-16
        return _floats_mapped(d, one_ulp)

    rewrite_json("data/upright_loads.json", other_machine)
    rewrite_json("data/duty_loads.json", lambda d: _floats_mapped(d, lambda x: one_ulp(x, 0)))
    rewrite_json("data/bearings.json", lambda d: _floats_mapped(d, lambda x: one_ulp(x, -math.inf)))
    rewrite_csv(lambda rows: [[other_machine_cell(c) for c in r] for r in rows])
    assert fingerprint.fea_inputs_hash(tmp_path) == fea0, "machine-level digits marked the FEA stale"
    assert fingerprint.fatigue_inputs_hash(tmp_path) == fat0, "machine-level digits marked the fatigue check stale"

    def bump_lc1_fz(rows):
        col = next(r for r in rows if r[:1] == ["id"]).index("Fz_N")
        return [r[:col] + [repr(float(r[col]) + 0.1)] + r[col + 1:] if r[:1] == ["LC1"] else r for r in rows]

    rewrite_csv(bump_lc1_fz)
    assert fingerprint.fea_inputs_hash(tmp_path) != fea0, "a changed load case did not change the FEA fingerprint"
    shutil.copy(ROOT / "data" / "loadcases.csv", tmp_path / "data" / "loadcases.csv")

    def bump_load(d):
        d["cases"]["LC1"]["UBJ"]["force_N"][0] += 0.01
        return d

    rewrite_json("data/upright_loads.json", bump_load)
    assert fingerprint.fea_inputs_hash(tmp_path) != fea0, "a changed interface load did not change the FEA fingerprint"
    assert fingerprint.fatigue_inputs_hash(tmp_path) != fat0, "the fatigue fingerprint did not follow the FEA's"
    shutil.copy(ROOT / "data" / "upright_loads.json", tmp_path / "data" / "upright_loads.json")

    def bump_bearing(d):
        d["rows"][0]["P"][0] *= 1 + 1e-5
        return d

    rewrite_json("data/bearings.json", bump_bearing)
    assert fingerprint.fea_inputs_hash(tmp_path) == fea0
    assert fingerprint.fatigue_inputs_hash(tmp_path) != fat0, "a changed bearing load did not change the fatigue one"


def test_canonical_form_drops_machine_digits_and_keeps_real_changes():
    # the same rules on small made-up files, so each case is exercised whatever today's data holds
    c = fingerprint.canonical
    loads, bear, lc = "data/upright_loads.json", "data/bearings.json", "data/loadcases.csv"
    a = {"cases": {"LC1": {"residual": 9.4e-16, "f": [-0.0, 459.17, 5.274779999353953, 3]}}}
    b = {"cases": {"LC1": {"residual": 3.4e-15, "f": [0.0, math.nextafter(459.17, 0), 5.274779999353954, 3]}}}
    same = json.dumps(b, indent=3).replace("\n", "\r\n").encode()
    assert c(loads, json.dumps(a).encode()) == c(loads, same)
    for changed in ([-0.0, 459.18, 5.274779999353953, 3], [-0.0, 459.17, 5.27479, 3], [-0.0, 459.17, 5.27478, 4],
                    [-0.0, 459.17, 5.274779999353953]):
        a2 = {"cases": {"LC1": {"residual": 9.4e-16, "f": changed}}}
        assert c(loads, json.dumps(a).encode()) != c(loads, json.dumps(a2).encode()), changed
    # residuals are dropped from the interface loads' cases only
    assert c(bear, json.dumps({"residual": 1.0}).encode()) != c(bear, json.dumps({"residual": 2.0}).encode())
    assert c(loads, b'{"residual": 1.0}') != c(loads, b'{"residual": 2.0}')

    csv_a = '# hardpoints v1.7\nid,name,F,n\nLC1,"bump, 3 g",-0.0,3\nLC2,x,2005.7,12345678901\n'
    csv_b = '# hardpoints v1.7\r\nid,name,F,n\r\nLC1,"bump, 3 g",0.0,3\r\nLC2,x,2005.7000000000003,12345678901\r\n'
    assert c(lc, csv_a.encode()) == c(lc, csv_b.encode()) == c(lc, csv_a.replace("\n", "\r").encode())
    for old, new in (("2005.7", "2005.8"), ("v1.7", "v1.8"), ("LC2,", "LC3,"), ('"bump, 3 g"', "bump"),
                     ("12345678901", "12345678902"), (",3\n", ",4\n")):
        assert c(lc, csv_a.encode()) != c(lc, csv_a.replace(old, new).encode()), (old, new)
    # only plain decimals are numbers: label-like cells stay text
    for x, y in (("007", "7"), ("inf", "Infinity"), (" 3", "3"), ("٣", "3"), ("1", "1.0")):
        assert c(lc, f"id\n{x}\n".encode()) != c(lc, f"id\n{y}\n".encode()), (x, y)
    # a CSV the parser rejects reads as a ValueError, which the requirements matrix turns into STALE (the field-size
    # limit is process-wide, so set here rather than assumed)
    old_limit = csv.field_size_limit(1000)
    try:
        with pytest.raises(ValueError):
            c(lc, b'a,"' + b"x" * 2000 + b'"\n')
    finally:
        csv.field_size_limit(old_limit)


def test_requirements_matrix_marks_an_fea_result_stale_when_its_inputs_changed():
    fea_out = {"fos_yield": 2.0, "critical_case": "LC2"}
    spec = {"fea": "fos_yield", "lo": 1.5, "note": "governing case {crit}"}
    val, status = requirements_matrix.evaluate(spec, {"fea": fea_out, "fea_current": True})
    assert status == "PASS" and "LC2" in val
    assert requirements_matrix.evaluate({"fea": "fos_yield", "lo": 2.5}, {"fea": fea_out, "fea_current": True})[1] == "FAIL"
    assert requirements_matrix.evaluate(spec, {"fea": fea_out, "fea_current": False})[1].startswith("STALE")
    assert requirements_matrix.evaluate(spec, {"fea": None, "fea_current": False})[1] == "OPEN"


def test_matrix_reads_the_fea_as_current_only_when_its_fingerprint_matches(monkeypatch):
    from common import read_version
    fea_out = json.loads((DATA / "upright_fea.json").read_text())
    version = read_version(DATA / "hardpoints.csv")
    monkeypatch.setattr(fingerprint, "fea_inputs_hash", lambda root=None, exclude=(): fea_out["_inputs_sha"])
    assert requirements_matrix.load_sources(version)["fea_current"] is True
    monkeypatch.setattr(fingerprint, "fea_inputs_hash", lambda root=None, exclude=(): "0" * 16)
    assert requirements_matrix.load_sources(version)["fea_current"] is False
