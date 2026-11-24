"""Fusion job: write every body of our tagged document as a binary STL in mm (static pose).

ARGS: {"project", "tag", "skip": [component-name substrings]}; output in <project>/cad/meshes
Meshes come from body.meshManager.createMeshCalculator (cleaner than the STL exporter).
Returns {file: component} so the Python side can map bodies to rigid groups.
index.json records the CAD fingerprint the document was built from, the upright STEP on disk at export
(fingerprint.upright_step_sha: the build does not record which STEP it imported, so export straight after the
build) and each mesh file's hash; common.load_mesh_index refuses meshes that no longer match any of them.
"""
import importlib
import json
import os
import struct
import sys

import adsk.core
import adsk.fusion

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import fsae_cad as C  # noqa: E402
import fingerprint  # noqa: E402

importlib.reload(fingerprint)          # Fusion keeps modules between runs


def write_stl(path, coords_cm, idx):
    n_tri = len(idx) // 3
    with open(path, "wb") as f:
        f.write(b"fsae mesh, mm".ljust(80, b" "))
        f.write(struct.pack("<I", n_tri))
        for t in range(n_tri):
            pts = []
            for j in range(3):
                i = idx[3 * t + j]
                pts.append((coords_cm[3 * i] * 10.0, coords_cm[3 * i + 1] * 10.0, coords_cm[3 * i + 2] * 10.0))
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for p in pts:
                f.write(struct.pack("<3f", *p))
            f.write(b"\x00\x00")


def main(args):
    """ARGS: {"project": <project folder>, "tag": <doc tag>}. Writes cad/meshes/*.stl + index.json."""
    doc, des = C.use_doc(args.get("tag", "fsae-front-suspension"))
    root = des.rootComponent
    stamp = des.attributes.itemByName("fsae", "hardpoints")
    built_from = des.attributes.itemByName("fsae", "inputs_sha")
    index = {"_hardpoints": stamp.value if stamp else None, "_inputs_sha": built_from.value if built_from else None,
             "_upright_step_sha": fingerprint.upright_step_sha(args["project"])}
    now = fingerprint.cad_inputs_hash(args["project"])
    if index["_inputs_sha"] != now:
        raise RuntimeError(f"the open CAD was built from inputs {index['_inputs_sha']}, the data is now {now}: rebuild it")
    out_dir = os.path.join(args["project"], "cad", "meshes")
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.endswith(".stl") or fn == "index.json":
            os.remove(os.path.join(out_dir, fn))
    for i in range(root.occurrences.count):
        occ = root.occurrences.item(i)
        comp = occ.component
        if any(s in comp.name for s in args.get("skip", [])):
            continue
        for j in range(comp.bRepBodies.count):
            body = comp.bRepBodies.item(j)
            mc = body.meshManager.createMeshCalculator()
            mc.setQuality(adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh)
            mesh = mc.calculate()
            safe = "".join(ch if ch.isalnum() else "_" for ch in f"{comp.name}__{body.name}")[:120]
            path = os.path.join(out_dir, safe + ".stl")
            write_stl(path, mesh.nodeCoordinatesAsDouble, mesh.nodeIndices)
            index[os.path.basename(path)] = {"component": comp.name, "body": body.name, "triangles": mesh.triangleCount,
                                             "sha": fingerprint.file_sha(path)}
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    return index
