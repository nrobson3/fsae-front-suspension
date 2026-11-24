"""Fingerprint of every input the CAD build reads (standard library only, so Fusion can import it).

The CAD meshes record the fingerprint they were built from (cad/meshes/index.json); the sweep and
the mass budget refuse meshes whose fingerprint no longer matches, so geometry built from old data
or old builder code can never pass silently. Used by common.load_mesh_index (analysis side) and by
scripts/fusion/job_build.py + job_export_meshes.py (Fusion side): one implementation, no copies.
The index also records the upright STEP the meshes were exported with (upright_step_sha) and each mesh file's
hash (file_sha). Neither is part of cad_inputs_hash, so adding them left the committed meshes valid.
"""
import csv
import hashlib
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = ("data/hardpoints.csv", "data/members.csv", "data/cockpit_envelope.json", "data/upright_design.json",
              "data/chassis_tabs.json")
CODE_FILES = ("scripts/fusion/fsae_parts.py", "scripts/fusion/job_build.py", "scripts/fusion/fsae_cad.py",
              "scripts/upright/upright_geom.py", "scripts/chassis_tabs.py")
VEHICLE_KEYS = ("tyre", "wheel", "damper", "arb_bar", "brake", "hub")   # the vehicle.json entries the CAD build reads


def cad_inputs_hash(root=ROOT):
    """16-hex-digit SHA-256 over the CAD's data files, builder code and the vehicle.json entries it uses."""
    h = hashlib.sha256()
    for rel in DATA_FILES + CODE_FILES:
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    with open(os.path.join(root, "data", "vehicle.json"), encoding="utf-8") as f:
        veh = json.load(f)
    h.update(json.dumps({k: veh[k] for k in VEHICLE_KEYS}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def sha16(data):
    """16 hex digits of the SHA-256 of some bytes (the length every fingerprint here uses)."""
    return hashlib.sha256(data).hexdigest()[:16]


def file_sha(path):
    """A file's bytes as they are (the meshes are binary STL): the mesh index records one per mesh, so a swapped or
    edited mesh is refused (common.load_mesh_index, common.load_mesh)."""
    with open(path, "rb") as f:
        return sha16(f.read())


def step_data(raw):
    """A STEP file's DATA section, line endings ignored (its header carries a timestamp). The section starts at a
    line that begins with DATA; (text such as METADATA; in a header string is not it)."""
    raw = raw.replace(b"\r\n", b"\n")
    i = raw.find(b"\nDATA;")
    if i < 0:
        raise ValueError("not a STEP file: no DATA section")
    return raw[i + 1:]


UPRIGHT_STEPS = ("cad/upright/upright_L.step", "cad/upright/upright_R.step")   # what job_build.stage_corner imports


def upright_step_sha(root=ROOT):
    """The upright STEP files the CAD build imports (their DATA sections). job_export_meshes.py records it in the
    mesh index, and common.load_mesh_index refuses meshes exported with another STEP."""
    h = hashlib.sha256()
    for rel in UPRIGHT_STEPS:
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            h.update(step_data(f.read()))
    return h.hexdigest()[:16]


# Inputs that run_all.py rewrites and the FEA or the fatigue check (run on their own) reads. Their last digits depend
# on the machine: least-squares balance residuals near 1e-15, the sign of a zero rounded from noise, the 16th digit of
# a bearing load (BLAS kernels, the maths library). A Linux CI runner, or a numpy upgrade here, rewrites them with
# nothing changed, so the fingerprints hash a canonical form: every non-integer number to 7 significant digits, -0.0
# as 0.0, and the interface loads' balance residuals left out (a check, not an input: upright_loads.py stops when one
# exceeds 1e-3). Values that agree to 7 digits hash the same; a change of one part in 10^6 or more always shows. Two
# machines can still round a full-precision value on either side of a 7-digit boundary (about one chance in 10^6 per
# value): the FEA then reads STALE, the safe side, and a re-run settles it.
GENERATED_INPUTS = ("data/upright_loads.json", "data/loadcases.csv", "data/duty_loads.json", "data/bearings.json")
SIG_DIGITS = 7
_PLAIN_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?")   # a number as Python writes one


def _drop_residuals(d):
    """upright_loads.json: each load case's balance residual (rounding noise, which nothing reads)."""
    cases = d.get("cases") if isinstance(d, dict) else None
    for case in (cases.values() if isinstance(cases, dict) else ()):
        if isinstance(case, dict):
            case.pop("residual", None)
    return d


DROP = {"data/upright_loads.json": _drop_residuals}


def _canon_number(x):
    return float(f"{x:.{SIG_DIGITS}g}") + 0.0        # (adding 0.0 turns -0.0 into 0.0)


def _canon_json(o):
    if isinstance(o, dict):
        return {k: _canon_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_canon_json(v) for v in o]
    if isinstance(o, float):
        return _canon_number(o)
    return o


def _canon_cell(cell):
    """A CSV cell. A plain decimal is a number (an integer stays exact, as in JSON); anything else, such as "007",
    "inf", " 3 " or a name, stays text."""
    m = _PLAIN_NUMBER.fullmatch(cell)
    if not m:
        return cell
    if m.group(1) is None and m.group(2) is None:
        return str(int(cell))
    return repr(_canon_number(float(cell)))


def canonical(rel, raw):
    """A generated input's content without its machine-dependent digits (GENERATED_INPUTS): JSON re-written with
    sorted keys, CSV cell by cell. Malformed content raises ValueError, which the requirements matrix reads as STALE."""
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if rel.endswith(".json"):
        data = DROP.get(rel, lambda d: d)(json.loads(text))
        return json.dumps(_canon_json(data), sort_keys=True, separators=(",", ":")).encode()
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    try:
        for row in csv.reader(io.StringIO(text)):
            writer.writerow([_canon_cell(c) for c in row])
    except csv.Error as e:
        raise ValueError(f"{rel}: {e}") from e
    return out.getvalue().encode()


FEA_FILES =("cad/upright/upright_L_fea.step", "cad/upright/upright_build.json", "data/upright_loads.json",
             "data/loadcases.csv", "data/materials.json", "data/upright_design.json",
             "scripts/fea/fea.py", "scripts/fea/upright_fea.py", "scripts/fea/benchmark_lug.py")


def fea_inputs_hash(root=ROOT, exclude=()):
    """16-hex-digit SHA-256 over what the upright FEA reads and runs: the meshed STEP (its DATA section:
    the header carries a timestamp), the build record (hole centres, sharp corners), the interface loads,
    the load cases, the material, the design data and the FEA code (the files run_all.py rewrites in their
    canonical form: GENERATED_INPUTS). upright_fea.json records it, and the requirements matrix marks the FEA's
    items STALE when it no longer matches."""
    h = hashlib.sha256()
    for rel in FEA_FILES:
        if rel in exclude:
            continue
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            raw = f.read().replace(b"\r\n", b"\n")
        if rel.endswith(".step"):
            raw = step_data(raw)      # (raises on a STEP with no DATA section; the same bytes as before otherwise)
        elif rel in GENERATED_INPUTS:
            raw = canonical(rel, raw)
        h.update(raw)
    with open(os.path.join(root, "data", "vehicle.json"), encoding="utf-8") as f:
        h.update(json.dumps(json.load(f)["hub"], sort_keys=True).encode())
    return h.hexdigest()[:16]


BENCH_FILES = ("scripts/fea/fea.py", "scripts/fea/benchmark_peaks.py")


def bench_inputs_hash(root=ROOT):
    """The FEA library and the benchmark script: fea_benchmarks.json records it, so a change to the FEA
    method marks the benchmarks STALE."""
    h = hashlib.sha256()
    for rel in BENCH_FILES:
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


FASTENER_FILES = ("data/fasteners.json", "data/poses.json", "scripts/fasteners.py", "scripts/interference.py")


def fastener_inputs_hash(root=ROOT):
    """The CAD's fingerprint plus the hardware and tool sizes, the pose grid and the sweep code:
    fastener_checks.csv records it, so changing a washer or a tool length makes the old result stale."""
    h = hashlib.sha256(cad_inputs_hash(root).encode())
    for rel in FASTENER_FILES:
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


HANDCALC_FILES = ("data/hardpoints.csv", "data/members.csv", "data/member_checks.csv", "data/loadcases.csv",
                  "data/upright_loads.json", "data/duty_loads.json", "data/materials.json", "data/materials_arb.json", "data/vehicle.json",
                  "data/vehicle_derived.json", "data/upright_design.json", "data/upright_fea.json", "data/fasteners.json",
                  "data/checks_week3b.json", "data/bearings.json", "scripts/week3b_checks.py",
                  "data/chassis_tabs.json", "scripts/chassis_tabs.py", "data/cockpit_envelope.json",
                  "cad/upright/upright_L.step", "cad/upright/upright_build.json",
                  "scripts/handcalcs.py", "scripts/handcalc_lc1.py", "scripts/forces.py", "scripts/suspension.py",
                  "scripts/fea/upright_fatigue.py", "scripts/fusion/fsae_parts.py")


def handcalc_inputs_hash(root=ROOT):
    """16-hex-digit SHA-256 over what the hand-calc answer keys read, including the CAD builder whose sizes
    handcalcs.py copies (rocker plates, bracket, spindle): handcalc_results.json records it."""
    h = hashlib.sha256()
    for rel in HANDCALC_FILES:
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            raw = f.read().replace(b"\r\n", b"\n")
        if rel.endswith(".step"):
            raw = step_data(raw)
        h.update(raw)
    return h.hexdigest()[:16]


def fatigue_inputs_hash(root=ROOT):
    """The FEA's fingerprint plus the duty loads, the fatigue script and the bearing loads (the generated ones in
    their canonical form): upright_fatigue.json records it."""
    h = hashlib.sha256(fea_inputs_hash(root).encode())
    for rel in ("data/duty_loads.json", "scripts/fea/upright_fatigue.py", "data/bearings.json"):
        h.update(rel.encode())
        with open(os.path.join(root, rel), "rb") as f:
            raw = f.read().replace(b"\r\n", b"\n")
        h.update(canonical(rel, raw) if rel in GENERATED_INPUTS else raw)
    return h.hexdigest()[:16]
