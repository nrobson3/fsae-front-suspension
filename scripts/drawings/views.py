"""Orthographic views and sections of a solid, from a fine surface mesh (for code-generated drawings).

Lines drawn: sharp edges (where the faces meet at more than FEATURE_DEG; tangent fillet boundaries are
left out, as drawings usually do) and silhouettes (the outline of curved faces as seen from the view
direction). Each segment is visible or hidden by casting a ray from its midpoint towards the viewer.
Sections: the mesh cut by a plane, the far side kept, the cut face returned as polygons for hatching.
"""
import numpy as np
import trimesh

FEATURE_DEG = 20.0


def mesh_solid(step_path, pick="largest", size=1.0, curvature=16):
    """Surface mesh (trimesh) of one solid in a STEP file (by default the largest)."""
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    for opt in ("Geometry.OCCFixDegenerated", "Geometry.OCCFixSmallEdges", "Geometry.OCCFixSmallFaces"):
        gmsh.option.setNumber(opt, 1)
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    keep = max(vols, key=lambda v: gmsh.model.occ.getMass(*v)) if pick == "largest" else vols[pick]
    for v in vols:
        if v != keep:
            gmsh.model.occ.remove([v], recursive=True)
    gmsh.model.occ.synchronize()
    errors = []
    for algo, k, curv in ((6, 1.0, curvature), (5, 1.0, curvature), (6, 1.0, 12), (1, 1.0, 12), (6, 1.3, 12)):
        gmsh.option.setNumber("Mesh.Algorithm", algo)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size * k)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * k / 8)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
        try:
            gmsh.model.mesh.generate(2)
            break
        except Exception as ex:
            errors.append(f"({algo}, {k}, {curv}): {str(ex)[:50]}")
            gmsh.model.mesh.clear()
    else:
        raise RuntimeError("surface mesh failed: " + "; ".join(errors))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i for i, t in enumerate(tags)}
    xyz = xyz.reshape(-1, 3)
    tris = []
    for d, t in gmsh.model.getEntities(2):
        types, _, nodes = gmsh.model.mesh.getElements(2, t)
        for ty, nn in zip(types, nodes):
            if ty == 2:
                tris.append(np.array([idx[int(n)] for n in nn]).reshape(-1, 3))
    gmsh.finalize()
    m = trimesh.Trimesh(vertices=xyz, faces=np.vstack(tris), process=True)
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    return m


def to_frame(mesh, origin, X, Y, Z):
    """Mesh in a local frame (rows X, Y, Z), origin at `origin`."""
    R = np.array([X, Y, Z])
    m = mesh.copy()
    m.vertices = (np.asarray(mesh.vertices) - origin) @ R.T
    return m


def view_lines(mesh, d, up, feature_deg=FEATURE_DEG, extra_edges=None):
    """2D segments (visible, hidden) of `mesh` seen along direction d (from the viewer into the scene)
    with `up` up: returns two arrays (n, 2, 2) in view coordinates (right, up)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    up = np.asarray(up, float) - d * np.dot(up, d)
    up /= np.linalg.norm(up)
    right = np.cross(d, up)
    fa = mesh.face_adjacency
    ang = mesh.face_adjacency_angles
    fn = mesh.face_normals
    sharp = ang > np.radians(feature_deg)
    s1, s2 = fn[fa[:, 0]] @ d, fn[fa[:, 1]] @ d
    silhouette = (s1 * s2 < 0) & ~sharp
    edges = mesh.face_adjacency_edges[sharp | silhouette]
    # boundary edges (open meshes, e.g. a cut) are always drawn
    bnd = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    edges = np.vstack([edges, mesh.edges_sorted[bnd]]) if len(bnd) else edges
    P = mesh.vertices[edges]                       # (n, 2, 3)
    if extra_edges is not None and len(extra_edges):
        P = np.vstack([P, extra_edges])
    mid = P.mean(axis=1)
    eps = 0.02 * max(mesh.extents)
    origins = mid - d * eps * 0.02 - d * 0.0
    origins = mid - d * 1e-3                       # nudge towards the viewer
    hits = mesh.ray.intersects_any(origins, np.tile(-d, (len(origins), 1)))
    # a segment lying ON a face can hit that face: retry from a point a little further out
    hits2 = mesh.ray.intersects_any(mid - d * 0.05, np.tile(-d, (len(origins), 1)))
    hidden = hits & hits2
    proj = np.stack([P @ right, P @ up], axis=-1)
    return proj[~hidden], proj[hidden]


def section(mesh, origin, normal, d_view, up):
    """Cut `mesh` by the plane (origin, normal) keeping the side the viewer looks at (opposite the normal),
    and return (visible lines, hidden lines, cut polygons in view coordinates) for the section view."""
    half = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=-np.asarray(normal, float), plane_origin=origin, cap=True)
    vis, hid = view_lines(half, d_view, up)
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    d = np.asarray(d_view, float) / np.linalg.norm(d_view)
    upv = np.asarray(up, float) - d * np.dot(up, d)
    upv /= np.linalg.norm(upv)
    right = np.cross(d, upv)
    polys = []
    if sec is not None:
        for e in sec.discrete:
            polys.append(np.stack([e @ right, e @ upv], axis=-1))
    return vis, hid, polys


def chain(segs, tol=1e-3):
    """Join 2D segments that share end points into polylines (fewer DXF entities)."""
    from collections import defaultdict
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))       # noqa: E731
    adj = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        adj[key(a)].append((i, 0))
        adj[key(b)].append((i, 1))
    used = np.zeros(len(segs), bool)
    out = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        line = [segs[i][0], segs[i][1]]
        for end in (1, 0):
            while True:
                p = line[-1] if end == 1 else line[0]
                nxt = None
                for j, k in adj[key(p)]:
                    if not used[j]:
                        nxt = (j, k)
                        break
                if nxt is None:
                    break
                j, k = nxt
                used[j] = True
                q = segs[j][1 - k]
                if end == 1:
                    line.append(q)
                else:
                    line.insert(0, q)
        out.append(np.array(line))
    return out
