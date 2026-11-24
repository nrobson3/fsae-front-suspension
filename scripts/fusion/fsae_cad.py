"""Geometry helpers that run INSIDE Fusion (imported by the job scripts).

All public functions take millimetres and convert to Fusion's internal centimetres. Bodies
are built with TemporaryBRepManager primitives in WORLD coordinates and committed into their
own component through a base feature (fast, robust, and the build script stays the source
of truth). Fusion is shared with other sessions, so the design is found by attribute.
"""
import math

import adsk.core
import adsk.fusion

TAG_GROUP, TAG_NAME = "fsae", "doc"
CM = 0.1  # mm -> cm


def app():
    return adsk.core.Application.get()


def find_doc(tag):
    a = app()
    for i in range(a.documents.count):
        d = a.documents.item(i)
        try:
            des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
        except Exception:
            continue
        if des:
            at = des.attributes.itemByName(TAG_GROUP, TAG_NAME)
            if at and at.value == tag:
                return d, des
    return None, None


def fresh_doc(tag):
    """Close our previous UNSAVED document with this tag (never anyone else's), open a new one."""
    d, _ = find_doc(tag)
    if d is not None and not d.isSaved:
        d.close(False)
    d = app().documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    des = adsk.fusion.Design.cast(d.products.itemByProductType("DesignProductType"))
    des.designType = adsk.fusion.DesignTypes.ParametricDesignType
    des.attributes.add(TAG_GROUP, TAG_NAME, tag)
    des.fusionUnitsManager.distanceDisplayUnits = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
    return d, des


def use_doc(tag):
    d, des = find_doc(tag)
    if d is None:
        raise RuntimeError(f"no open document tagged {tag!r}; run the build job first")
    d.activate()
    return d, des


def tm():
    return adsk.fusion.TemporaryBRepManager.get()


def P(p):
    return adsk.core.Point3D.create(p[0] * CM, p[1] * CM, p[2] * CM)


def V(v):
    return adsk.core.Vector3D.create(v[0], v[1], v[2])


# ---------------- small vector maths (plain tuples, mm) ----------------
def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def perp(a, hint=(0.0, 0.0, 1.0)):
    """A unit vector normal to a, as close as possible to hint."""
    a = unit(a)
    h = sub(hint, mul(a, dot(hint, a)))
    if norm(h) < 1e-6:
        h = sub((1.0, 0.0, 0.0), mul(a, a[0]))
    return unit(h)


def lerp(a, b, t):
    return add(a, mul(sub(b, a), t))


# ---------------- primitives (mm in, temp bodies out) ----------------
def cyl(p1, p2, r, r2=None):
    return tm().createCylinderOrCone(P(p1), r * CM, P(p2), (r if r2 is None else r2) * CM)


def tube(p1, p2, ro, ri):
    b = cyl(p1, p2, ro)
    if ri > 0:
        ext = mul(unit(sub(p2, p1)), 0.5)  # inner slightly longer: no zero-thickness end skins
        tm().booleanOperation(b, cyl(sub(p1, ext), add(p2, ext), ri), adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return b


def box(center, ldir, wdir, L, W, H):
    obb = adsk.core.OrientedBoundingBox3D.create(P(center), V(unit(ldir)), V(unit(wdir)), L * CM, W * CM, H * CM)
    return tm().createBox(obb)


def sphere(c, r):
    return tm().createSphere(P(c), r * CM)


def ring(center, axis, r_out, r_in, width):
    """Annulus of `width` along `axis`, centred at `center`."""
    a = mul(unit(axis), width / 2)
    return tube(sub(center, a), add(center, a), r_out, r_in)


def union(bodies):
    bodies = [b for b in bodies if b is not None]
    base = bodies[0]
    for b in bodies[1:]:
        tm().booleanOperation(base, b, adsk.fusion.BooleanTypes.UnionBooleanType)
    return base


def cut(base, tool):
    tm().booleanOperation(base, tool, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return base


def strut(p1, p2, w, h, up=(0.0, 0.0, 1.0)):
    """Rectangular bar from p1 to p2 (w across, h in the `up` direction)."""
    d = sub(p2, p1)
    L = norm(d)
    wdir = cross(perp(d, up), unit(d))
    return box(lerp(p1, p2, 0.5), d, wdir, L, w, h)


def plate(p1, p2, r, t, normal):
    """Flat bar with round ends ('dog bone' link plate) between p1 and p2, thickness t along normal."""
    n = mul(unit(normal), t / 2)
    body = cyl(sub(p1, n), add(p1, n), r)
    body = union([body, cyl(sub(p2, n), add(p2, n), r)])
    d = sub(p2, p1)
    wdir = unit(cross(unit(normal), unit(d)))
    body = union([body, box(lerp(p1, p2, 0.5), d, wdir, norm(d), 2 * r, t)])
    return body


# ---------------- committing and inspecting ----------------
def new_component(root, name):
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ


def step_bodies(root, path):
    """Temporary copies of the solids in a STEP file. TemporaryBRepManager.createFromFile does not
    read STEP, so the file is imported into a scratch component, copied, and the scratch deleted."""
    import os
    scratch = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    opts = app().importManager.createSTEPImportOptions(os.path.normpath(path))
    if not app().importManager.importToTarget(opts, scratch.component):
        raise RuntimeError(f"STEP import failed: {path}")
    found = []

    def walk(comp):
        found.extend(comp.bRepBodies.item(i) for i in range(comp.bRepBodies.count))
        for i in range(comp.occurrences.count):
            walk(comp.occurrences.item(i).component)

    walk(scratch.component)
    temps = [tm().copy(b) for b in found]
    scratch.deleteMe()
    return temps


def commit(comp, bodies, names=None):
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    for b in bodies:
        comp.bRepBodies.add(b, bf)
    bf.finishEdit()
    out = [comp.bRepBodies.item(i) for i in range(comp.bRepBodies.count)]
    if names:
        for b, n in zip(out[-len(names):], names):
            b.name = n
    return out


_APP_CACHE = {}


def appearance(design, name_options):
    key = tuple(name_options)
    if key in _APP_CACHE:
        return _APP_CACHE[key]
    lib = app().materialLibraries.itemByName("Fusion Appearance Library")
    found = None
    for nm in name_options:
        try:
            a = lib.appearances.itemByName(nm)
        except Exception:
            a = None
        if a:
            found = design.appearances.itemByName(nm) or design.appearances.addByCopy(a, nm)
            break
    _APP_CACHE[key] = found
    return found


def paint(design, comp, options, opacity=None):
    ap = appearance(design, options)
    for i in range(comp.bRepBodies.count):
        b = comp.bRepBodies.item(i)
        if ap:
            b.appearance = ap
    if opacity is not None:
        comp.opacity = opacity if hasattr(comp, "opacity") else None


def bbox_mm(comp):
    bb = None
    for i in range(comp.bRepBodies.count):
        b = comp.bRepBodies.item(i).boundingBox
        if bb is None:
            bb = b.copy()
        else:
            bb.combine(b)  # combine mutates in place and returns a bool
    if bb is None:
        return None
    return [round(bb.minPoint.x / CM, 2), round(bb.minPoint.y / CM, 2), round(bb.minPoint.z / CM, 2),
            round(bb.maxPoint.x / CM, 2), round(bb.maxPoint.y / CM, 2), round(bb.maxPoint.z / CM, 2)]


def body_report(comp):
    out = []
    for i in range(comp.bRepBodies.count):
        b = comp.bRepBodies.item(i)
        vol = b.getPhysicalProperties(adsk.fusion.CalculationAccuracy.HighCalculationAccuracy).volume / CM ** 3  # mm3
        out.append({"body": b.name, "lumps": b.lumps.count, "volume_mm3": round(vol, 1), "solid": b.isSolid})
    return out
