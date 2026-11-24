"""Fusion job: save viewport images of our tagged document.

ARGS: {"tag", "out_dir", "width", "height", "views": [{"name", "eye": [x,y,z] mm, "target": [x,y,z] mm,
       "up": [x,y,z], "ortho": bool}], "hide": [component-name substrings to hide]}
Ground-plane effects are switched off for the capture and restored afterwards.
"""
import os
import sys

import adsk.core
import adsk.fusion

sys.path.insert(0, os.path.dirname(__file__))
import fsae_cad as C  # noqa: E402


def main(args):
    a = C.app()
    prev = a.activeDocument
    doc, des = C.use_doc(args.get("tag", "fsae-front-suspension"))
    adsk.doEvents()                                  # let the activation reach the UI before taking its viewport
    vp = a.activeViewport
    gp = a.preferences.graphicsPreferences
    saved = None
    try:
        saved = (gp.graphicsPreset, gp.canvasEffects.isGroundPlaneEnabled,
                 gp.canvasEffects.isGroundShadowEnabled, gp.canvasEffects.isGroundReflectionEnabled)
        gp.graphicsPreset = adsk.fusion.GraphicsPresets.CustomGraphicsPreset if hasattr(adsk.fusion, "GraphicsPresets") else gp.graphicsPreset
        gp.canvasEffects.isGroundPlaneEnabled = False
        gp.canvasEffects.isGroundShadowEnabled = False
        gp.canvasEffects.isGroundReflectionEnabled = False
    except Exception:
        saved = None
    root = des.rootComponent
    hidden = []
    for i in range(root.occurrences.count):
        occ = root.occurrences.item(i)
        if any(s in occ.component.name for s in args.get("hide", [])):
            if occ.isLightBulbOn:
                occ.isLightBulbOn = False
                hidden.append(occ)
    out = []
    os.makedirs(args["out_dir"], exist_ok=True)
    # optional pose: {component name: 16 floats, row-major, cm}; applied with no compute before capture
    posed = []
    for i in range(root.occurrences.count):
        occ = root.occurrences.item(i)
        m = args.get("pose", {}).get(occ.component.name)
        if m:
            mat = adsk.core.Matrix3D.create()
            mat.setWithArray(m)
            occ.transform2 = mat
            posed.append(occ)
    try:
        for v in args["views"]:
            cam = vp.camera
            cam.isSmoothTransition = False
            cam.cameraType = adsk.core.CameraTypes.OrthographicCameraType if v.get("ortho") else adsk.core.CameraTypes.PerspectiveCameraType
            cam.eye = C.P(v["eye"])
            cam.target = C.P(v["target"])
            cam.upVector = C.V(v.get("up", [0, 0, 1]))
            vp.camera = cam
            if v.get("fit"):
                vp.fit()
            adsk.doEvents()
            path = os.path.join(args["out_dir"], v["name"] + ".png")
            tmp = path + ".new.png"                 # the old image stays until a new one really exists
            if os.path.exists(tmp):
                os.remove(tmp)
            ok = vp.saveAsImageFile(tmp, args.get("width", 1600), args.get("height", 1000))
            if not ok or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
                # seen 2026-11-03: True returned and no file written while another session's document held the window
                raise RuntimeError(f"viewport image not written: {path} (Fusion window minimised or busy?)")
            os.replace(tmp, path)
            out.append(path)
    finally:                                        # restore the shared Fusion even when a capture fails
        for occ in posed:
            occ.transform2 = adsk.core.Matrix3D.create()
        for occ in hidden:
            occ.isLightBulbOn = True
        if saved:
            try:
                gp.graphicsPreset, gp.canvasEffects.isGroundPlaneEnabled, gp.canvasEffects.isGroundShadowEnabled, \
                    gp.canvasEffects.isGroundReflectionEnabled = saved
            except Exception:
                pass
        if prev is not None and prev != doc:        # Fusion is shared: hand the other session its document back
            prev.activate()
    return out
