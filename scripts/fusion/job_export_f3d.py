"""Fusion job: export our tagged design as a Fusion archive (.f3d) that Navy can open.

ARGS: {"tag", "path"}. Works on an unsaved document, so it does not use a cloud document slot.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fsae_cad as C  # noqa: E402


def main(args):
    doc, des = C.use_doc(args.get("tag", "fsae-front-suspension"))
    em = des.exportManager
    opts = em.createFusionArchiveExportOptions(args["path"])
    ok = em.execute(opts)
    return {"ok": bool(ok), "path": args["path"], "bytes": os.path.getsize(args["path"]) if os.path.exists(args["path"]) else 0}
