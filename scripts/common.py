"""Shared helpers: project paths, hardpoint and vehicle I/O, small vector maths, plot style."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
NOTES = ROOT / "notes"

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])
MIRROR = np.array([1.0, -1.0, 1.0])  # left corner -> right corner (y -> -y)


def unit(v):
    return v / np.linalg.norm(v)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_matrix(rv):
    """Rodrigues: rotation matrix for rotation vector rv (axis * angle, rad)."""
    th = np.linalg.norm(rv)
    if th < 1e-15:
        return np.eye(3) + skew(rv)
    k = skew(rv / th)
    return np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)


def rotate_about(p, origin, axis_unit, angle):
    return origin + rotvec_matrix(axis_unit * angle) @ (p - origin)


def read_version(path):
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("# version:"):
                return line.split(":", 1)[1].strip().split()[0]
    return "unversioned"


def load_hardpoints(path=None):
    """Return ({id: xyz ndarray}, {id: csv row}) from a hardpoint CSV ('#' lines are comments)."""
    path = Path(path) if path else DATA / "hardpoints.csv"
    with open(path, newline="") as f:
        rows = [line for line in f if not line.lstrip().startswith("#")]
    pts, meta = {}, {}
    for r in csv.DictReader(rows):
        pts[r["id"]] = np.array([float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])])
        meta[r["id"]] = r
    return pts, meta


def load_vehicle(path=None):
    with open(Path(path) if path else DATA / "vehicle.json") as f:
        return json.load(f)


# ---- CAD meshes: committed in cad/meshes so the sweep and mass budget rerun without Fusion ----
MESH_DIR = ROOT / "cad" / "meshes"


def load_materials():
    """materials.json plus materials_arb.json (kept apart: the FEA fingerprint hashes materials.json whole)."""
    out = {}
    for name in ("materials.json", "materials_arb.json"):
        d = json.loads((DATA / name).read_text(encoding="utf-8"))
        d.pop("_about", None)
        if set(d) & set(out):
            raise RuntimeError(f"material defined twice: {sorted(set(d) & set(out))}")
        out.update(d)
    return out


def cad_inputs_hash(root=ROOT):
    """Fingerprint of every input the CAD build reads (see fingerprint.py, shared with the Fusion jobs)."""
    from fingerprint import cad_inputs_hash as _h
    return _h(str(root))


REEXPORT = "rebuild the CAD (scripts/fusion/job_build.py) and re-export the meshes (job_export_meshes.py)"


def load_mesh_index(root=None):
    """{stl file: {component, body, triangles, sha}} for the exported CAD meshes. Refuses meshes built from other
    inputs (the CAD fingerprint), meshes exported with another upright STEP than cad/upright holds, an upright STEP
    that its own inputs no longer reproduce (its stamp: stamps.verify_upright), and any mesh file whose bytes are not
    the ones the index recorded (a swapped or edited mesh). root: another project tree (the tests)."""
    import fingerprint
    import stamps
    root, mesh_dir, data = (ROOT, MESH_DIR, DATA) if root is None else (Path(root), Path(root) / "cad" / "meshes",
                                                                        Path(root) / "data")
    path = mesh_dir / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"no CAD meshes: {REEXPORT}")
    index = json.loads(path.read_text(encoding="utf-8"))
    ver, sha = index.pop("_hardpoints", None), index.pop("_inputs_sha", None)
    step = index.pop("_upright_step_sha", None)
    hp_ver, hp_sha = read_version(data / "hardpoints.csv"), cad_inputs_hash(root)
    if (ver, sha) != (hp_ver, hp_sha):
        raise RuntimeError(f"CAD meshes are from hardpoints {ver} / inputs {sha}, the data is {hp_ver} / {hp_sha}: {REEXPORT}")
    now = fingerprint.upright_step_sha(str(root))
    if step != now:
        raise RuntimeError(f"the CAD meshes were exported with upright STEP {step}, cad/upright now holds {now}: "
                           f"{REEXPORT}")
    why = stamps.stale(["upright_geom"], root=root)
    if why:
        raise RuntimeError("the upright STEP is not current: " + "; ".join(why))
    for fn, info in index.items():
        have = fingerprint.file_sha(mesh_dir / fn) if (mesh_dir / fn).exists() else "missing"
        if have != info.get("sha"):
            raise RuntimeError(f"cad/meshes/{fn} is not the mesh the index recorded (hash {have}, the index says "
                               f"{info.get('sha')}): a mesh was swapped, edited or lost; {REEXPORT}")
    return index


def load_mesh(fn, info, root=None, **kw):
    """One exported CAD body as a trimesh, parsed from the very bytes whose hash is checked against its index entry
    (info, from load_mesh_index): the sweeps and the mass budget read every mesh through this."""
    import io

    import trimesh
    from fingerprint import sha16
    mesh_dir = MESH_DIR if root is None else Path(root) / "cad" / "meshes"
    raw = (mesh_dir / fn).read_bytes()
    if sha16(raw) != info.get("sha"):
        raise RuntimeError(f"cad/meshes/{fn} is not the mesh the index recorded (hash {sha16(raw)}, the index says "
                           f"{info.get('sha')}): {REEXPORT}")
    return trimesh.load_mesh(io.BytesIO(raw), file_type="stl", **kw)


# ---- plot style (dataviz reference palette, light mode, static PNG for the report) ----
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BAND = "#0ca30c"      # status-good, used only as a 10 % wash for target bands
FAIL = "#d03b3b"      # status-critical, used only with a text label


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def band(ax, lo, hi, label="target"):
    """Shade a target band (green 10 % wash) with hairline edges and a text label."""
    ax.axhspan(lo, hi, color=BAND, alpha=0.10, linewidth=0, zorder=0)
    for v in (lo, hi):
        ax.axhline(v, color=BAND, linewidth=0.8, alpha=0.6, zorder=0)
    ax.text(0.995, hi, f" {label} ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7, color=INK_2)


def figure(nrows=1, ncols=1, w=6.4, h=3.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2.0,
                         "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False,
                             facecolor=SURFACE)
    return fig, axes
