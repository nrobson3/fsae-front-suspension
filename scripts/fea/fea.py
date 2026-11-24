"""Linear-elastic 3D FEA: gmsh meshes, scikit-fem solves (quadratic tetrahedra).

Why these tools (decision D4, revised 2026-10-26): both install with `pip`, run on any OS and can
be re-run by anyone from the repository, which a downloaded CalculiX binary cannot promise. The
chain is checked against beam theory (benchmark_cantilever.py) before it is trusted on a part.

Units: mm, N, MPa throughout.

    m = Model(load_mesh(mesh_box(200, 20, 20, 5.0, path)), E=200e3, nu=0.3)
    m.fix(lambda x: x[0] < 1e-6)
    m.traction(lambda x: x[0] > 200 - 1e-6, [0, 0, -1000])
    m.solve()
    vm = m.von_mises()                      # nodal von Mises (L2 projection)
    m.probe(vm, [[100, 0, 10]])             # value at a point
"""
import math
import re

import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, FacetBasis, Functional, LinearForm, asm, condense, solve
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress, sym_grad


def _robust_invF(self, x, tind=None, newton_max_iters=100, newton_tol=1e-10):
    """scikit-fem's inverse isoparametric map, started from the element centroid (its default start, 0.5 in
    every coordinate, lies outside the tetrahedron), and accepting a point that maps back to within 1e-6 mm
    when the step-size test is too strict for a strongly curved element."""
    X = np.zeros(x.shape) + 0.25
    for _ in range(newton_max_iters):
        dX = np.einsum("ijkl,jkl->ikl", self.invDF(X, tind), x - self.F(X, tind))
        X = np.clip(X + dX, 0.0, 1.0)
        if (np.linalg.norm(dX, 1, (0, 2)) < newton_tol).all():
            return X
    miss = np.abs(x - self.F(X, tind)).max()
    if miss < 1e-6:
        return X
    raise ValueError(f"inverse isoparametric map missed by {miss:.2e} mm")


def _patch_skfem():
    from skfem.mapping import MappingIsoparametric
    MappingIsoparametric.invF = _robust_invF


_patch_skfem()


# ---------------- meshing (gmsh) ----------------
def _gmsh_start():
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("part")
    return gmsh


def _gmsh_mesh(gmsh, size, out_msh, refine=None):
    """Second-order tetrahedra, max edge `size`. refine: [(points, radius, size)] local balls.
    When the 3D mesher reports a surface-mesh self-intersection at a point, a finer ball is added there
    and the mesh is retried (a local repair, recorded in QUALITY["repairs"])."""
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size / 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)       # 12 elements per full circle on holes
    fields = []

    def ball(p, radius, s):
        f = gmsh.model.mesh.field.add("Ball")
        for k, v in zip(("XCenter", "YCenter", "ZCenter"), p):
            gmsh.model.mesh.field.setNumber(f, k, float(v))
        gmsh.model.mesh.field.setNumber(f, "Radius", radius)
        gmsh.model.mesh.field.setNumber(f, "VIn", s)
        gmsh.model.mesh.field.setNumber(f, "VOut", size)
        fields.append(f)

    for pts, radius, s in refine or []:
        for p in pts:
            ball(p, radius, s)
    local = min([s for _, _, s in refine or []] + [size])
    fmin = gmsh.model.mesh.field.add("Min")
    if not fields:
        ball((0.0, 0.0, 0.0), 0.0, size)
    gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    repairs = []
    for attempt in range(8):
        try:
            mesher = _generate(gmsh)
            break
        except RuntimeError as ex:
            pts = [tuple(float(v) for v in m) for m in re.findall(r"point \(([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)\)", str(ex))]
            if not pts:
                raise
            for q in pts[:1]:
                ball(q, 3.0 * local, local / (2.0 + attempt))
                repairs.append([round(v, 2) for v in q])
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.clear()
    else:
        raise RuntimeError(f"mesh still failing after {len(repairs)} local repairs at {repairs}")
    _straighten_folded(gmsh)
    QUALITY["mesher"] = mesher
    QUALITY["repairs"] = repairs
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(out_msh))
    gmsh.finalize()
    return out_msh


MESHERS = [(6, 1), (5, 1), (1, 1), (6, 10)]   # (2D, 3D) gmsh algorithms to try, gmsh's default first


def _generate(gmsh):
    """Generate the volume mesh, trying the next surface / volume algorithm when one fails (the 3D mesher
    can stop with "a segment and a facet intersect" on a surface mesh it cannot recover at some sizes).
    Returns the (2D, 3D) algorithms that worked; the full error texts (with any reported points) otherwise."""
    errors = []
    for a2, a3 in MESHERS:
        gmsh.option.setNumber("Mesh.Algorithm", a2)
        gmsh.option.setNumber("Mesh.Algorithm3D", a3)
        try:
            gmsh.model.mesh.generate(3)
            if 11 not in list(gmsh.model.mesh.getElements(3)[0]):
                raise RuntimeError("no quadratic tetrahedra produced")
            return [a2, a3]
        except Exception as ex:
            errors.append(f"({a2}, {a3}): {ex}")
            gmsh.model.mesh.clear()
    raise RuntimeError("every mesher failed: " + "; ".join(errors))


TET10_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]     # gmsh's mid-side node order
QUALITY = {}                                       # last mesh: {"min_sj", "straightened", "straightened_at"}


def _straighten_folded(gmsh, threshold=0.05, passes=4):
    """Curving the mid-side nodes onto a tight fillet corner can fold a quadratic tet inside out (scaled
    Jacobian <= 0), which breaks the element mapping. Put those elements' mid-side nodes back on their
    straight edges (a local fix: the part's surface there is faceted over one element) and repeat until
    none is left below `threshold` (which also catches valid but badly curved elements). gmsh's own
    high-order optimiser aborts the process on this part, so it is not used. The minimum quality, the
    count and the centroids of the straightened elements are kept in QUALITY."""
    fixed, where = set(), []
    for it in range(passes + 1):
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        k = list(types).index(11)
        tg, nn = np.asarray(tags[k]), np.asarray(nodes[k]).reshape(-1, 10)
        q = np.array(gmsh.model.mesh.getElementQualities(tg, "minSJ"))
        bad = np.flatnonzero(q < threshold)
        QUALITY.clear()
        QUALITY.update({"min_sj": float(q.min()), "straightened": len(fixed), "straightened_at": where})
        if len(bad) == 0:
            return
        if it == passes:
            raise ValueError(f"{len(bad)} quadratic tets still below scaled Jacobian {threshold} after "
                             f"{passes} straightening passes (min {q.min():.2f})")
        for i in bad:
            row = nn[i]
            P = {int(n): np.array(gmsh.model.mesh.getNode(int(n))[0]) for n in row[:4]}
            for (a, b), mid in zip(TET10_EDGES, row[4:]):
                gmsh.model.mesh.setNode(int(mid), ((P[int(row[a])] + P[int(row[b])]) / 2).tolist(), [])
            if int(tg[i]) not in fixed:
                fixed.add(int(tg[i]))
                where.append(np.mean(list(P.values()), axis=0).round(2).tolist())


def mesh_box(L, W, H, size, out_msh):
    """Box [0, L] x [-W/2, W/2] x [-H/2, H/2] (mm)."""
    gmsh = _gmsh_start()
    gmsh.model.occ.addBox(0, -W / 2, -H / 2, L, W, H)
    gmsh.model.occ.synchronize()
    return _gmsh_mesh(gmsh, size, out_msh)


def mesh_step(step_path, size, out_msh, refine=None):
    """Mesh the solid in a STEP file; gmsh reads STEP in mm. (OCC's shape healing at import is left off: on
    this part it turned a narrow fillet strip into a surface mesh the 3D mesher could not recover.)"""
    gmsh = _gmsh_start()
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()
    return _gmsh_mesh(gmsh, size, out_msh, refine)


def load_mesh(path):
    """Quadratic tetrahedral mesh (scikit-fem MeshTet2) from a gmsh .msh file."""
    import meshio
    from skfem.io.meshio import from_meshio
    m = meshio.read(str(path))
    m.cells = [c for c in m.cells if c.type == "tetra10"]       # volume elements only
    return from_meshio(m, ignore_orientation=True, ignore_interior_facets=True)


# ---------------- model ----------------
CHUNK_ELEMENTS = 20000      # above this, assemble and evaluate stress in element chunks (memory, see Model)


class Model:
    def __init__(self, mesh, E, nu):
        """A scikit-fem basis stores every shape function's value and gradient at every integration point
        of every element: about 8 GB for a quarter-million quadratic tets. Large meshes therefore keep only
        a one-point global basis (for the dof numbering, supports and loads) and assemble the stiffness and
        evaluate the stress in chunks of CHUNK_ELEMENTS, with the usual integration order; the result is
        the same as a one-shot assembly."""
        self.mesh = mesh
        self.lam, self.mu = lame_parameters(E, nu)
        self.chunked = mesh.t.shape[1] > CHUNK_ELEMENTS
        self.ib = Basis(mesh, ElementVector(ElementTetP2()), **({"intorder": 1} if self.chunked else {}))
        self.sb = self.ib.with_element(ElementTetP2())
        self.comp = self.ib.split_indices()               # dof indices of u_x, u_y, u_z (scalar-basis order)
        form = linear_elasticity(self.lam, self.mu)
        if self.chunked:
            K = None
            for b in self._chunk_bases():
                Kc = asm(form, b)
                K = Kc if K is None else K + Kc
                del Kc, b
            self.K = K.tocsr()
        else:
            self.K = asm(form, self.ib)
        self.f = np.zeros(self.K.shape[0])
        self.fixed = np.array([], dtype=np.int64)
        self.u = None

    def _chunk_bases(self):
        nel = self.mesh.t.shape[1]
        for i in range(0, nel, CHUNK_ELEMENTS):
            yield Basis(self.mesh, ElementVector(ElementTetP2()), elements=np.arange(i, min(i + CHUNK_ELEMENTS, nel)))

    def facets(self, where):
        f = self.mesh.facets_satisfying(where, boundaries_only=True)
        if len(f) == 0:
            raise ValueError("no boundary facets selected")
        return f

    # ---- supports ----
    def fix(self, where, components=(0, 1, 2)):
        """Hold the boundary faces where `where(x)` is true (x: 3 x n points) in the given directions."""
        dofs = self.ib.get_dofs(self.facets(where))
        sel = np.concatenate([dofs.all([f"u^{c + 1}"]) for c in components])
        self.fixed = np.union1d(self.fixed, sel)
        return sel

    # ---- loads ----
    def area(self, where):
        fb = FacetBasis(self.mesh, ElementTetP2(), facets=self.facets(where))
        return float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fb))

    def traction(self, where, force):
        """Uniform traction on the faces where `where(x)`, adding up to `force` (N vector)."""
        t = np.asarray(force, float) / self.area(where)
        fb = FacetBasis(self.mesh, self.ib.elem, facets=self.facets(where))

        @LinearForm
        def tr(v, w):
            return t[0] * v[0] + t[1] * v[1] + t[2] * v[2]

        self.f += asm(tr, fb)

    def bearing(self, where, axis, centre, force, half_angle_deg=90.0, axial_weight=None, incline_deg=0.0,
                incline_dir=None):
        """Pin or bearing load on a cylindrical face (the bore of a lug or a bearing housing).

        Radial pressure on the side the force pushes towards, with a cosine distribution over
        +/- half_angle about the load direction, scaled so the radial resultant equals the radial
        part of `force` (N); its component along `axis` is applied as a uniform traction on the same face.
        axial_weight(a) optionally weights the pressure along the axis (a = axial coordinate from `centre`).
        incline_deg / incline_dir tilt the traction towards incline_dir, as a tapered roller's contact
        force does on its cup: that adds an axial resultant (returned, so the caller can balance it)
        and moves the radial force's line of action by R tan(incline) along the axis."""
        extra_axial = np.zeros(3)
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        F = np.asarray(force, float)
        Fax = a * F.dot(a)
        Frad = F - Fax
        c = np.asarray(centre, float)
        fb_s = FacetBasis(self.mesh, ElementTetP2(), facets=self.facets(where))
        fb_v = FacetBasis(self.mesh, self.ib.elem, facets=self.facets(where))
        if np.linalg.norm(Frad) > 0:
            target = Frad / np.linalg.norm(Frad)
            d = target.copy()
            beta = np.radians(half_angle_deg)

            def shape(x):
                r = x - c[:, None, None]
                ax_coord = np.einsum("i,i...->...", a, r)
                r = r - a[:, None, None] * ax_coord
                rn = r / np.linalg.norm(r, axis=0)
                ang = np.arccos(np.clip(np.einsum("i,i...->...", d, rn), -1.0, 1.0))
                w = np.where(ang < beta, np.cos(ang * np.pi / (2 * beta)), 0.0)
                if axial_weight is not None:
                    w = w * axial_weight(ax_coord)
                return w, rn

            # a face that is not a full, evenly meshed cylinder (cut by a feature, or coarse facets) gives the
            # cosine patch a sideways resultant; turn the patch until its resultant lies along the load
            def resultant():
                r = np.array([Functional(lambda w, i=i: shape(w.x)[0] * shape(w.x)[1][i]).assemble(fb_s)
                              for i in range(3)])
                return r - a * r.dot(a)

            for _ in range(8):
                res = resultant()
                if not np.all(np.isfinite(res)) or np.linalg.norm(res) == 0:
                    raise ValueError("bearing patch has no area in the loaded half: check the face selection")
                err = res / np.linalg.norm(res) - target
                if np.linalg.norm(err) < 1e-7:
                    break
                d = d - err
                d -= a * d.dot(a)
                d /= np.linalg.norm(d)
            else:
                res = resultant()
                miss = np.linalg.norm(res / np.linalg.norm(res) - target)
                if miss > 1e-4:
                    raise ValueError(f"bearing patch could not be turned onto the load ({miss:.1e} rad off)")
            along_d = float(res.dot(target))
            if not along_d > 0:
                raise ValueError("bearing patch resultant does not point along the load")
            cb, sb = math.cos(math.radians(incline_deg)), math.sin(math.radians(incline_deg))
            p0 = np.linalg.norm(Frad) / (along_d * cb)
            cdir = np.zeros(3) if incline_dir is None else np.asarray(incline_dir, float) / np.linalg.norm(incline_dir)

            @LinearForm
            def pin(v, w):
                s, rn = shape(w.x)
                tr = cb * rn + sb * cdir[:, None, None]
                return p0 * s * (tr[0] * v[0] + tr[1] * v[1] + tr[2] * v[2])

            self.f += asm(pin, fb_v)
            if sb:
                extra_axial = cdir * sb * p0 * float(Functional(lambda w: shape(w.x)[0]).assemble(fb_s))
        if np.linalg.norm(Fax) > 0:
            t = Fax / float(Functional(lambda w: 1.0 + 0.0 * w.x[0]).assemble(fb_s))

            @LinearForm
            def ax(v, w):
                return t[0] * v[0] + t[1] * v[1] + t[2] * v[2]

            self.f += asm(ax, fb_v)
        return extra_axial

    def applied_resultant(self):
        """Sum of the assembled load vector per direction (a check that loads add up as intended)."""
        return np.array([self.f[self.comp[i]].sum() for i in range(3)])

    def fix_points(self, points, components):
        """Hold the mesh vertex nearest each point in the listed directions (a 3-2-1 support for a
        self-equilibrated load set: the reactions must come out near zero)."""
        xyz = self.sb.doflocs
        nv = self.mesh.p.shape[1]                       # P2 dofs start with the vertices
        sel, where = [], []
        for p, comps in zip(points, components):
            i = int(np.argmin(np.linalg.norm(xyz[:, :nv] - np.asarray(p, float)[:, None], axis=0)))
            sel += [int(self.comp[c][i]) for c in comps]
            where.append(xyz[:, i].copy())
        self.fixed = np.union1d(self.fixed, np.array(sel, dtype=np.int64))
        self.support_xyz = np.array(where)              # the vertices actually held (for excluding their stress)
        return sel

    # ---- solve and post-process ----
    def solve(self, method="direct", tol=1e-9):
        """'direct' (SuperLU) for small models; 'amg': conjugate gradients with a smoothed-aggregation
        multigrid preconditioner given the six rigid-body modes (needed for large 3D meshes)."""
        if method == "direct":
            self.u = solve(*condense(self.K, self.f, D=self.fixed))
            return self.u
        import pyamg
        from scipy.sparse.linalg import cg
        n = self.K.shape[0]
        free = np.setdiff1d(np.arange(n), self.fixed)
        A = self.K[free][:, free].tocsr()
        b = self.f[free]
        xyz = self.sb.doflocs
        Bm = np.zeros((n, 6))
        for i in range(3):
            Bm[self.comp[i], i] = 1.0
        for j, (p, q) in enumerate(((1, 2), (2, 0), (0, 1))):       # rotations about x, y, z
            Bm[self.comp[p], 3 + j] = -xyz[q]
            Bm[self.comp[q], 3 + j] = xyz[p]
        ml = pyamg.smoothed_aggregation_solver(A, B=Bm[free], symmetry="hermitian")
        x, info = cg(A, b, rtol=tol, maxiter=2000, M=ml.aspreconditioner())
        if info != 0:
            raise RuntimeError(f"CG did not converge (info {info})")
        self.u = np.zeros(n)
        self.u[free] = x
        return self.u

    def qp_stress(self):
        """Stress tensor (3, 3, elements, points) and coordinates (3, elements, points) at the
        quadrature points: the raw P2 stress field, no smoothing."""
        if not self.chunked:
            return self.stress(), self.ib.global_coordinates().value
        S, X = [], []
        for b in self._chunk_bases():
            S.append(np.asarray(linear_stress(self.lam, self.mu)(sym_grad(b.interpolate(self.u)))))
            X.append(np.asarray(b.global_coordinates().value))
            del b
        return np.concatenate(S, axis=2), np.concatenate(X, axis=1)

    @staticmethod
    def von_mises_of(s):
        return np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                       + 3.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))

    def reactions(self):
        """Support reaction per direction (N), from K u - f at the fixed dofs, summed (the net force)."""
        r = self.K @ self.u - self.f
        return np.array([r[np.intersect1d(self.comp[i], self.fixed)].sum() for i in range(3)])

    def support_reactions(self):
        """Reaction at every held dof (N). For a self-equilibrated load on a 3-2-1 support each one must be
        near zero: the summed reaction cannot show a moment imbalance, these can."""
        return (self.K @ self.u - self.f)[self.fixed]

    def displacement(self, points):
        """(n x 3) displacement at points (n x 3)."""
        P = self.sb.probes(np.asarray(points, float).T)
        return np.stack([P @ self.u[self.comp[i]] for i in range(3)], axis=1)

    def stress(self):
        if self.chunked:
            return self.qp_stress()[0]
        return linear_stress(self.lam, self.mu)(sym_grad(self.ib.interpolate(self.u)))

    def von_mises(self):
        if self.chunked:
            raise NotImplementedError("nodal projection needs the full basis: use qp_stress on large meshes")
        s = self.stress()
        vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                     + 3.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))
        return self.sb.project(vm)

    def component(self, i, j):
        if self.chunked:
            raise NotImplementedError("nodal projection needs the full basis: use qp_stress on large meshes")
        return self.sb.project(self.stress()[i, j])

    def probe(self, field, points):
        return self.sb.probes(np.asarray(points, float).T) @ field

    def node_xyz(self):
        """Coordinates of the scalar-basis dofs (vertices and edge midpoints), 3 x n."""
        return self.sb.doflocs
