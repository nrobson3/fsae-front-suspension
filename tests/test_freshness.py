"""Freshness checks added after the step E pipeline audit (2026-11-21): each one refuses a stale or edited input.

1. The upright STEP is tied to the CAD: the mesh index records the STEP the meshes were exported with, and the
   STEP's own stamp says its inputs still reproduce it (stamps.verify_upright).
2. Each mesh file's hash is in the index; the sweeps and the mass budget refuse a swapped or edited mesh.
3. Every run_all output is stamped with the content of what it was made from (data/stamps.json), and
   requirements_matrix.load_sources refuses an output whose stamp no longer matches.
Most tests work on a copy of the project in tmp_path. The last ones read the real tree and need scripts/run_all.py
to have run (as the other test files do).
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
import fingerprint  # noqa: E402
import requirements_matrix  # noqa: E402
import stamps  # noqa: E402
from common import load_mesh, load_mesh_index  # noqa: E402

UPRIGHT_OUT = stamps.PRODUCERS["upright_geom"][2]


def _copy(rel, dst):
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst / rel)


@pytest.fixture()
def tree(tmp_path):
    """A copy of what load_mesh_index reads (the CAD fingerprint's files, the upright and its inputs, the meshes),
    with the upright stamped as it stands (the fixture does not regenerate it)."""
    for rel in fingerprint.DATA_FILES + fingerprint.CODE_FILES + ("data/vehicle.json", "data/materials.json",
                                                                  "scripts/common.py") + UPRIGHT_OUT:
        _copy(rel, tmp_path)
    shutil.copytree(ROOT / "cad" / "meshes", tmp_path / "cad" / "meshes")
    stamps._record("upright_geom", stamps.input_hashes("upright_geom", tmp_path), tmp_path)
    assert load_mesh_index(tmp_path), "the copy should pass before anything is changed"
    return tmp_path


def _edit_step(path):
    """Change one coordinate in a STEP's DATA section, as a regenerated upright with other geometry would."""
    raw = path.read_bytes()
    i = raw.index(b"CARTESIAN_POINT", raw.index(b"DATA;"))
    m = re.compile(rb"-?\d+\.\d+").search(raw, i)
    path.write_bytes(raw[:m.start()] + b"%.6f" % (float(m.group()) + 0.5) + raw[m.end():])


# ---- 1. the upright STEP and the CAD
def test_meshes_exported_with_another_upright_step_are_refused(tree):
    """A STEP regenerated without a CAD rebuild: even with the new STEP stamped as current, the meshes are refused."""
    _edit_step(tree / "cad" / "upright" / "upright_R.step")
    stamps._record("upright_geom", stamps.input_hashes("upright_geom", tree), tree)
    assert stamps.stale(["upright_geom"], root=tree) == []
    with pytest.raises(RuntimeError, match="exported with upright STEP"):
        load_mesh_index(tree)


def test_a_cad_rebuilt_on_a_step_older_than_its_inputs_is_refused(tree):
    """The other way round: the design data changed and the CAD was rebuilt and exported (the index carries the new
    CAD fingerprint) but the upright was not regenerated, so the STEP is older than its inputs."""
    p = tree / "data" / "upright_design.json"
    ud = json.loads(p.read_text(encoding="utf-8"))
    ud["housing"]["wall_mm"] += 0.5
    p.write_text(json.dumps(ud, indent=1), encoding="utf-8")
    ip = tree / "cad" / "meshes" / "index.json"
    index = json.loads(ip.read_text(encoding="utf-8"))
    index["_inputs_sha"] = fingerprint.cad_inputs_hash(str(tree))          # what the rebuild and export record
    ip.write_text(json.dumps(index, indent=1), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"upright STEP is not current.*upright_design\.json changed"):
        load_mesh_index(tree)


def test_an_upright_code_change_outside_the_cad_fingerprint_is_caught(tree):
    """common.py is not in the CAD fingerprint but the upright's script imports it: the STEP's stamp covers it."""
    p = tree / "scripts" / "common.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert fingerprint.cad_inputs_hash(str(tree)) == fingerprint.cad_inputs_hash()
    with pytest.raises(RuntimeError, match=r"scripts/common\.py changed"):
        load_mesh_index(tree)


def test_verify_upright_stamps_only_a_reproduced_step(tree):
    p = tree / "data" / "upright_design.json"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")       # the stamp is now stale
    seen = {}

    def reproduces(folder):          # a regeneration that gives the committed files back
        seen["inputs"] = sorted(q.relative_to(folder).as_posix() for q in folder.rglob("*") if q.is_file())
        for rel in UPRIGHT_OUT:
            _copy_from(tree, folder, rel)

    def differs(folder):             # one that gives other geometry
        reproduces(folder)
        _edit_step(folder / "cad" / "upright" / "upright_L_fea.step")

    with pytest.raises(RuntimeError, match=r"not what scripts/upright/upright_geom\.py gives.*upright_L_fea\.step"):
        stamps.verify_upright(differs, root=tree)
    assert stamps.stale(["upright_geom"], root=tree), "a refused check must not renew the stamp"
    # the scratch folder held only the declared inputs (and the stamping machinery, when the tree has it), so an
    # undeclared read fails the regeneration
    machinery = {f"scripts/{m}" for m in stamps.NOT_CODE}
    assert [f for f in seen["inputs"] if f not in machinery] == sorted(stamps.input_hashes("upright_geom", tree))
    assert stamps.verify_upright(reproduces, root=tree)                # renewed: says what had changed
    assert stamps.stale(["upright_geom"], root=tree) == []
    assert stamps.verify_upright(differs, root=tree) == []            # current: not regenerated again


def _copy_from(src_root, dst_root, rel):
    (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root / rel, dst_root / rel)


def _step_surface(step):
    """The STEP solid's surface as triangles whose nodes lie on it (16 per turn on curves, 3 mm at most)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        idx = {t: i for i, t in enumerate(tags)}
        tris = [np.array([idx[n] for n in nodes]).reshape(-1, 3)
                for t, _, nodes in zip(*gmsh.model.mesh.getElements(2)) if t == 2]
    finally:
        gmsh.finalize()
    return trimesh.Trimesh(xyz.reshape(-1, 3), np.vstack(tris), process=True)


def test_the_committed_upright_meshes_are_the_step_solids():
    """Independent of any recorded hash: each upright mesh lies on its STEP's surface and covers it, within the
    tessellation (0.12 and 0.14 mm here; 0.08 and 0.14 on a finer surface mesh on 2026-11-21, when the index's STEP
    hash was added without a re-export); the same mesh shifted by 0.5 mm reads 0.55 mm."""
    index = load_mesh_index()
    for side in ("L", "R"):
        fn = next(k for k, v in index.items() if v["component"] == f"{side} Upright")
        stl = load_mesh(fn, index[fn], process=True)
        surf = _step_surface(ROOT / "cad" / "upright" / f"upright_{side}.step")
        assert trimesh.proximity.closest_point(surf, stl.vertices)[1].max() < 0.25, side
        assert trimesh.proximity.closest_point(stl, surf.vertices)[1].max() < 0.25, side


# ---- 2. each mesh file's hash
def test_a_swapped_edited_or_missing_mesh_is_refused(tree):
    md = tree / "cad" / "meshes"
    left, right = md / "L_Upright__upright.stl", md / "R_Upright__upright.stl"
    a, b = left.read_bytes(), right.read_bytes()
    left.write_bytes(b)
    right.write_bytes(a)
    with pytest.raises(RuntimeError, match=r"L_Upright__upright\.stl is not the mesh the index recorded"):
        load_mesh_index(tree)
    left.write_bytes(a)
    right.write_bytes(b)
    load_mesh_index(tree)
    edited = bytearray(a)
    edited[84 + 12] ^= 0x01                       # one bit of the first triangle's first vertex
    left.write_bytes(bytes(edited))
    with pytest.raises(RuntimeError, match="is not the mesh the index recorded"):
        load_mesh_index(tree)
    left.unlink()
    with pytest.raises(RuntimeError, match=r"hash missing"):
        load_mesh_index(tree)


def test_load_mesh_checks_the_bytes_it_parses(tree):
    index = load_mesh_index(tree)
    fn = "L_Upright__upright.stl"
    assert len(load_mesh(fn, index[fn], root=tree, process=False).faces) == index[fn]["triangles"]
    with pytest.raises(RuntimeError, match="is not the mesh the index recorded"):
        load_mesh(fn, {**index[fn], "sha": "0" * 16}, root=tree)
    with pytest.raises(RuntimeError, match="is not the mesh the index recorded"):
        load_mesh(fn, {k: v for k, v in index[fn].items() if k != "sha"}, root=tree)


def test_the_sweeps_and_the_mass_budget_refuse_a_swapped_mesh(tree, monkeypatch):
    import fasteners
    import interference
    import mass_budget
    md = tree / "cad" / "meshes"
    a, b = (md / "L_Upright__upright.stl").read_bytes(), (md / "R_Upright__upright.stl").read_bytes()
    (md / "L_Upright__upright.stl").write_bytes(b)
    (md / "R_Upright__upright.stl").write_bytes(a)
    for rel in ("data/fasteners.json", "data/poses.json"):
        _copy(rel, tree)
    poses = json.loads((tree / "data" / "poses.json").read_text())
    poses["poses"] = poses["poses"][:1]              # if the check failed, the sweep would run one pose in tmp_path
    (tree / "data" / "poses.json").write_text(json.dumps(poses))
    monkeypatch.setattr(common, "ROOT", tree)
    monkeypatch.setattr(common, "MESH_DIR", md)
    monkeypatch.setattr(common, "DATA", tree / "data")
    for mod in (interference, fasteners, mass_budget):     # nothing can be written outside tmp_path
        monkeypatch.setattr(mod, "DATA", tree / "data")
        monkeypatch.setattr(mod, "NOTES", tree)
    for run in (interference.load_bodies, interference.main, fasteners.main, mass_budget.main):
        with pytest.raises(RuntimeError, match=r"L_Upright__upright\.stl is not the mesh the index recorded"):
            run()


def test_the_sweeps_read_meshes_only_through_load_mesh():
    for name in ("interference", "fasteners", "mass_budget"):
        src = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "load_mesh(" in src and "load_mesh_index(" in src, name
        assert "MESH_DIR" not in src and "trimesh.load" not in src, f"{name} reads a mesh file without its hash check"


def test_the_index_records_every_mesh_and_the_upright_step():
    index = json.loads((ROOT / "cad" / "meshes" / "index.json").read_text(encoding="utf-8"))
    files = {p.name for p in (ROOT / "cad" / "meshes").glob("*.stl")}
    entries = {k for k in index if not k.startswith("_")}
    assert entries == files
    assert all(re.fullmatch(r"[0-9a-f]{16}", index[k]["sha"]) for k in entries)
    assert index["_upright_step_sha"] == fingerprint.upright_step_sha()


# ---- 3. every output stamped with what it was made from
@pytest.fixture()
def stamped(tmp_path):
    """A copy of the data, the analysis scripts and the upright, with every producer stamped as the files stand."""
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts", ignore=shutil.ignore_patterns("__pycache__", "fusion",
                                                                                          "fea", "drawings"))
    for rel in UPRIGHT_OUT + ("cad/meshes/index.json",):
        _copy(rel, tmp_path)
    for name in stamps.PRODUCERS:
        stamps._record(name, stamps.input_hashes(name, tmp_path), tmp_path)
    assert stamps.stale(root=tmp_path) == []
    return tmp_path


def test_a_stamp_refuses_an_edited_input(stamped):
    p = stamped / "data" / "members.csv"
    p.write_text(p.read_text() + "\n")
    why = stamps.stale(["forces"], root=stamped, follow=False)
    assert len(why) == 1 and "data/members.csv changed since it ran" in why[0]
    hit = {w.split(":")[0].split(" ")[0] for w in stamps.stale(root=stamped)}
    assert {"forces", "upright_loads", "bearings", "bolts", "duty_loads", "bom", "interference", "fasteners",
            "mass_budget", "upright_geom"} <= hit                  # everything that reads members.csv


def test_a_stamp_refuses_an_edited_output_and_follows_it_downstream(stamped):
    p = stamped / "data" / "loadcases.csv"                       # vehicle_calcs wrote it; forces read it
    p.write_text(p.read_text().replace("LC1", "LC1 ", 1))
    why = stamps.stale(["forces"], root=stamped)
    assert any("data/loadcases.csv is not what vehicle_calcs wrote" in w for w in why)
    assert any(w.startswith("forces: data/loadcases.csv changed") for w in why)


def test_a_stamp_refuses_a_code_change_in_an_imported_module(stamped):
    p = stamped / "scripts" / "suspension.py"                    # forces imports it; kinematics, poses... too
    p.write_text(p.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert any("scripts/suspension.py changed" in w for w in stamps.stale(["forces"], root=stamped, follow=False))


def test_a_missing_stamp_or_output_is_refused(stamped):
    st = json.loads((stamped / "data" / "stamps.json").read_text(encoding="utf-8"))
    del st["poses"]
    (stamped / "data" / "stamps.json").write_text(json.dumps(st), encoding="utf-8")
    assert stamps.stale(["poses"], root=stamped) == [
        "poses has no stamp (run scripts/run_all.py)"]
    (stamped / "data" / "bolt_checks.csv").unlink()
    assert any("data/bolt_checks.csv is missing" in w for w in stamps.stale(["bolts"], root=stamped))


def test_code_inputs_follow_the_imports():
    code = stamps.code_files("scripts/upright_loads.py")
    assert {"scripts/upright_loads.py", "scripts/forces.py", "scripts/suspension.py", "scripts/common.py"} <= set(code)
    assert not any(c.endswith(("fingerprint.py", "stamps.py")) for c in code)
    assert stamps.code_files("scripts/upright/upright_geom.py") == ["scripts/common.py", "scripts/upright/upright_geom.py"]
    assert "scripts/interference.py" in stamps.code_files("scripts/fasteners.py")


def test_a_producer_is_stamped_only_after_a_full_run_that_read_only_what_it_declares(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "dummy.py").write_text("x = 1\n")
    (tmp_path / "data").mkdir()
    a, b, out = (tmp_path / "data" / n for n in ("a.csv", "b.csv", "out.csv"))
    a.write_text("1\n")
    b.write_text("2\n")
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setitem(stamps.PRODUCERS, "dummy", ("scripts/dummy.py", ("data/a.csv",), ("data/out.csv",)))

    @stamps.producer("dummy", when=lambda *_, trial=False, **__: not trial)
    def run(read_b=False, write=True, edit_a=False, trial=False):
        with open(a) as f:
            v = f.read()
        if read_b:
            v += b.read_text()
        if write:
            out.write_text(v)
        if edit_a:
            a.write_text("edited during the run\n")

    def current():
        return stamps.stale(["dummy"], root=tmp_path) == []

    run()
    assert current()
    a.write_text("3\n")
    assert not current()
    with pytest.raises(RuntimeError, match=r"data/b\.csv"):
        run(read_b=True)                              # an undeclared read: refused, not stamped
    assert not current()
    run(write=False)                                  # wrote nothing: the old stamp stays
    assert not current()
    run(trial=True)                                   # a trial run (when is false): not stamped
    assert not current()
    run(edit_a=True)                                  # an input edited during the run: the stamp holds the old one
    assert not current()
    run()
    assert current()
    out.write_text("edited by hand\n")
    assert not current()


def test_code_inputs_include_modules_imported_from_packages_and_relatively(tmp_path):
    s = tmp_path / "scripts"
    (s / "pkg").mkdir(parents=True)
    (s / "pkg" / "__init__.py").write_text("")
    (s / "pkg" / "mod.py").write_text("from . import sib\n")
    (s / "pkg" / "sib.py").write_text("")
    (s / "lib.py").write_text("import fingerprint\n")
    (s / "main.py").write_text("from pkg import mod\nimport lib\nfrom lib import name_not_a_module\n")
    assert stamps.code_files("scripts/main.py", tmp_path) == [
        "scripts/lib.py", "scripts/main.py", "scripts/pkg/__init__.py", "scripts/pkg/mod.py", "scripts/pkg/sib.py"]


def _dummy(tmp_path, monkeypatch, inputs=("data/a.csv",), outputs=("data/out.csv",)):
    for rel in ("scripts/dummy.py",) + tuple(inputs):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n")
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setitem(stamps.PRODUCERS, "dummy", ("scripts/dummy.py", tuple(inputs), tuple(outputs)))


def test_a_move_into_place_counts_as_a_write_and_cache_or_unindexed_mesh_reads_are_refused(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch, inputs=("data/a.csv", "cad/meshes/index.json"))
    (tmp_path / "cad" / "meshes" / "index.json").write_text(json.dumps({"_hardpoints": "v", "m.stl": {"sha": "x"}}))
    for rel in ("cad/meshes/m.stl", "cad/meshes/stray.stl", ".tmp/cache.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("1")
    out = tmp_path / "data" / "out.csv"

    @stamps.producer("dummy")
    def run(extra=None):
        (tmp_path / "data" / "a.csv").read_text()
        (tmp_path / "cad" / "meshes" / "m.stl").read_bytes()          # listed in the declared index: allowed
        if extra:
            (tmp_path / extra).read_bytes()
        tmp = tmp_path / "data" / "out.tmp"
        tmp.write_text("v\n")
        os.replace(tmp, out)                                         # an atomic write: the output is written

    run()
    assert stamps.stale(["dummy"], root=tmp_path) == []
    for extra in ("cad/meshes/stray.stl", ".tmp/cache.json"):
        with pytest.raises(RuntimeError, match=re.escape(extra)):
            run(extra)


def test_a_missing_declared_input_is_never_current(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch, inputs=("data/a.csv", "data/gone.csv"))
    (tmp_path / "data" / "gone.csv").unlink()

    @stamps.producer("dummy")
    def run():
        (tmp_path / "data" / "out.csv").write_text("v\n")

    with pytest.raises(RuntimeError, match=r"data/gone\.csv.*missing"):
        run()
    st = {"dummy": {"inputs": stamps.input_hashes("dummy", tmp_path), "outputs": {"data/out.csv": "x"}}}
    (tmp_path / "data" / "stamps.json").write_text(json.dumps(st))     # a stamp made while it was missing
    assert any("missing" in w for w in stamps.stale(["dummy"], root=tmp_path))


def test_nested_producer_runs_are_both_traced(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch)
    monkeypatch.setitem(stamps.PRODUCERS, "inner", ("scripts/dummy.py", ("data/a.csv",), ("data/inner.csv",)))

    @stamps.producer("inner")
    def inner():
        (tmp_path / "data" / "inner.csv").write_text("i\n")

    @stamps.producer("dummy")
    def outer():
        inner()                                    # starts with nothing recorded: its record equals the outer's
        (tmp_path / "data" / "a.csv").read_text()
        (tmp_path / "data" / "out.csv").write_text("o\n")

    outer()
    assert stamps.stale(["dummy", "inner"], root=tmp_path) == []
    assert stamps._RUNS == []


def test_an_output_read_before_it_is_written_is_an_undeclared_input(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch)
    out = tmp_path / "data" / "out.csv"
    out.write_text("old\n")

    @stamps.producer("dummy")
    def run(carry_over):
        (tmp_path / "data" / "a.csv").read_text()
        old = out.read_text() if carry_over else ""
        out.write_text(old + "new\n")
        out.read_text()                            # reading back what it wrote is fine

    run(False)
    assert stamps.stale(["dummy"], root=tmp_path) == []
    with pytest.raises(RuntimeError, match=r"data/out\.csv"):
        run(True)

    @stamps.producer("dummy")
    def append():                                  # an append keeps the old content: it is read
        (tmp_path / "data" / "a.csv").read_text()
        with open(out, "a") as f:
            f.write("more\n")

    with pytest.raises(RuntimeError, match=r"data/out\.csv"):
        append()


def test_code_read_as_data_or_imported_unseen_is_refused(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch)
    s = tmp_path / "scripts"
    (s / "other.py").write_text("y = 2\n")
    (s / "__pycache__").mkdir()
    for n in ("dummy", "other"):
        (s / "__pycache__" / f"{n}.cpython-313.pyc").write_bytes(b"\0")

    @stamps.producer("dummy")
    def run(extra=None):
        (tmp_path / "data" / "a.csv").read_text()
        (s / "dummy.py").read_text()                                   # its own code: hashed by the stamp
        (s / "__pycache__" / "dummy.cpython-313.pyc").read_bytes()
        if extra:
            (s / extra).read_bytes()
        (tmp_path / "data" / "out.csv").write_text("v\n")

    run()
    assert stamps.stale(["dummy"], root=tmp_path) == []
    for extra in ("other.py", "__pycache__/other.cpython-313.pyc"):  # not imported by it, so not in its stamp
        with pytest.raises(RuntimeError, match="other"):
            run(extra)


def test_a_run_whose_trace_failed_is_not_stamped(tmp_path, monkeypatch):
    _dummy(tmp_path, monkeypatch)

    @stamps.producer("dummy")
    def run():
        (tmp_path / "data" / "a.csv").read_text()
        (tmp_path / "data" / "out.csv").write_text("v\n")

    real = stamps._note

    def broken(*_):
        raise OSError("boom")

    monkeypatch.setattr(stamps, "_note", broken)
    with pytest.raises(RuntimeError, match="trace of the files it opened failed"):
        run()
    assert stamps.stale(["dummy"], root=tmp_path) == ["dummy has no stamp (run scripts/run_all.py)"]
    monkeypatch.setattr(stamps, "_note", real)
    run()
    assert stamps.stale(["dummy"], root=tmp_path) == []


def test_an_unreadable_or_damaged_stamp_input_is_reported_not_raised(stamped):
    step = stamped / "cad" / "upright" / "upright_L.step"
    step.write_bytes(step.read_bytes().replace(b"\nDATA;", b"\nDATA"))      # no DATA section any more
    assert any("unreadable" in w for w in stamps.stale(["bom"], root=stamped, follow=False))
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        stamps._record("bom", stamps.input_hashes("bom", stamped), stamped)
    p = stamped / "scripts" / "bolts.py"
    p.write_text(p.read_text(encoding="utf-8") + "\ndef (:\n", encoding="utf-8")
    assert any("bolts: its stamp cannot be checked (SyntaxError" in w for w in stamps.stale(["bolts"], root=stamped))
    (stamped / "data" / "stamps.json").write_text("{ not json")
    assert stamps.stale(root=stamped)[0].startswith("data/stamps.json is not valid JSON")


def test_a_stamp_file_held_by_another_program_reads_as_not_current(stamped, monkeypatch):
    def held(root):
        raise PermissionError("in use")

    monkeypatch.setattr(stamps, "_load", held)
    assert stamps.stale(root=stamped) == ["data/stamps.json cannot be read (in use): try again"]


IMPORTS = """
import importlib.util, json, os, sys
root = os.path.normcase(os.path.abspath(sys.argv[1])) + os.sep
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
def project_modules():
    out = set()
    for key, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        f = os.path.normcase(os.path.abspath(f)) if isinstance(f, str) else ""
        if f.startswith(root):
            out.add((key, f[len(root):].replace(os.sep, "/")))
    return out
result = {}
for name, rel in json.loads(sys.argv[2]).items():
    for key, _ in project_modules():
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location("producer_" + name, os.path.join(sys.argv[1], rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    result[name] = sorted({f for _, f in project_modules()})
print(json.dumps(result))
"""


def test_the_code_a_producer_imports_is_all_in_its_stamp(tmp_path):
    """Import each producer in a fresh module state and list the project files Python actually loaded: each must be
    in code_files (which reads the import statements), so a module reached another way would be caught here."""
    import subprocess
    (tmp_path / "imports.py").write_text(IMPORTS)
    scripts = {n: p[0] for n, p in stamps.PRODUCERS.items()}
    res = subprocess.run([sys.executable, "-B", str(tmp_path / "imports.py"), str(ROOT), json.dumps(scripts)],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    loaded = json.loads(res.stdout.strip().splitlines()[-1])

    def norm(paths):                               # the subprocess reports paths in the platform's case
        return {os.path.normcase(p).replace(os.sep, "/") for p in paths}

    machinery = norm(f"scripts/{m}" for m in stamps.NOT_CODE)
    assert set(loaded) == set(scripts) and all(loaded.values())
    for name, files in loaded.items():
        extra = norm(files) - norm(stamps.code_files(scripts[name])) - machinery
        assert not extra, f"{name} loads {sorted(extra)}, which its stamp does not hash"


def test_a_step_without_a_data_section_reads_as_changed(tmp_path):
    good = b"ISO-10303-21;\nHEADER;\nFILE_NAME('METADATA;');\nENDSEC;\nDATA;\n#1=X();\nENDSEC;\n"
    assert fingerprint.step_data(good).startswith(b"DATA;\n#1")      # not cut at the header's METADATA;
    p = tmp_path / "x.step"
    p.write_bytes(good.replace(b"\nDATA;", b"\nDATA"))
    assert stamps.content_sha(p) == stamps.UNREADABLE
    with pytest.raises(ValueError):
        fingerprint.step_data(p.read_bytes())


def test_every_matrix_source_and_every_stamped_entry_point_is_declared():
    produced = {o for _, _, outs in stamps.PRODUCERS.values() for o in outs}
    src = (ROOT / "scripts" / "requirements_matrix.py").read_text(encoding="utf-8")
    read = set(re.findall(r'DATA / "([a-z_]+\.(?:csv|json))"', src))
    own_fingerprint = {"upright_fea.json", "upright_fatigue.json", "fea_benchmarks.json", "handcalc_results.json"}
    generated = read - own_fingerprint - {"hardpoints.csv", "requirements.json"}           # the last two are inputs
    assert len(generated) >= 12 and {f"data/{n}" for n in generated} <= produced, generated - {
        o.split("/")[-1] for o in produced}
    decorated = set()
    for p in (ROOT / "scripts").glob("*.py"):
        decorated |= set(re.findall(r'@stamps\.producer\("([a-z_]+)"', p.read_text(encoding="utf-8")))
    assert decorated == set(stamps.PRODUCERS) - {"upright_geom"}       # the upright is stamped by verify_upright


# ---- the real tree (after scripts/run_all.py)
def test_the_committed_outputs_match_their_stamps():
    assert stamps.stale() == [], "re-run scripts/run_all.py"


def test_load_sources_refuses_an_output_whose_stamp_no_longer_matches(monkeypatch):
    version = common.read_version(common.DATA / "hardpoints.csv")
    requirements_matrix.load_sources(version)          # current first, so the refusal below is for this reason
    real = stamps.content_sha
    monkeypatch.setattr(stamps, "content_sha",
                        lambda p: "0" * 16 if Path(p).name == "member_checks.csv" else real(p))
    with pytest.raises(RuntimeError, match=r"member_checks\.csv is not what forces wrote"):
        requirements_matrix.load_sources(version)
