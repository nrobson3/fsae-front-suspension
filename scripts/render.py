"""Offline renderer for the write-up's images (no OpenGL here): an orthographic z-buffer in numpy.

render(V, F, colors, eye, target, up) -> an RGB image (float, 0-1). Faces are lit in WORLD space (a key and a fill
light from the camera's side, two-sided, so a mesh's facet orientation does not matter); the camera basis is only
used to project. Outlines are drawn where the depth jumps (silhouettes) or the surface folds by more than CREASE_DEG
between neighbouring pixels' faces. Rendered at SSAA x the size and averaged down.
"""
import numpy as np

CREASE_DEG = 35.0


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def render(V, F, colors, eye, target, up=(0.0, 0.0, 1.0), width=1600, height=1000, ssaa=2, bg=(1.0, 1.0, 1.0),
           margin=0.06, outline=True, spec=0.22):
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    colors = np.broadcast_to(np.asarray(colors, float), (len(F), 3))
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = _unit(target - eye)
    side = np.cross(fwd, up)
    if np.linalg.norm(side) < 1e-9 * np.linalg.norm(up):     # looking straight along `up`: any square direction does
        side = np.cross(fwd, (1.0, 0.0, 0.0) if abs(fwd[0]) < 0.9 else (0.0, 1.0, 0.0))
    right = _unit(side)
    upv = np.cross(right, fwd)
    W, H = width * ssaa, height * ssaa
    # projection: screen x along `right`, screen y along `upv` (image rows grow downwards), depth along fwd
    sx, sy, sz = V @ right, V @ upv, V @ fwd
    lo, hi = np.array([sx.min(), sy.min()]), np.array([sx.max(), sy.max()])
    scale = min(W * (1 - 2 * margin) / (hi[0] - lo[0]), H * (1 - 2 * margin) / (hi[1] - lo[1]))
    cx, cy = (lo + hi) / 2
    px = (sx - cx) * scale + W / 2
    py = H / 2 - (sy - cy) * scale
    # world-space shading, two-sided: a normal is flipped to face the viewer
    n = _unit(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]))
    n = np.where((n @ -fwd)[:, None] < 0, -n, n)
    key = _unit(-fwd + 0.6 * upv + 0.5 * right)
    fill = _unit(-fwd - 0.3 * upv - 0.8 * right)
    half = _unit(key - fwd)
    lam = 0.34 + 0.52 * np.clip(n @ key, 0, None) + 0.18 * np.clip(n @ fill, 0, None)
    shade = np.clip(colors * lam[:, None] + spec * np.clip(n @ half, 0, None)[:, None] ** 40, 0, 1)
    zbuf = np.full((H, W), np.inf)
    fid = np.full((H, W), -1, np.int64)
    X, Y, Z = px[F], py[F], sz[F]
    order = np.argsort(Z.mean(axis=1))                       # near first: fewer overwrites
    for t in order:
        x, y, z = X[t], Y[t], Z[t]
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), W - 1)
        y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-12:
            continue
        a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        c = 1 - a - b
        inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        if not inside.any():
            continue
        depth = a * z[0] + b * z[1] + c * z[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (depth < sub)
        sub[win] = depth[win]
        fid[y0:y1 + 1, x0:x1 + 1][win] = t
    img = np.empty((H, W, 3))
    img[:] = bg
    hit = fid >= 0
    img[hit] = shade[fid[hit]]
    if outline:
        edge = np.zeros((H, W), bool)
        zf = np.where(hit, zbuf, np.nan)
        span = np.nanmax(zf) - np.nanmin(zf) if hit.any() else 1.0
        cos_c = np.cos(np.radians(CREASE_DEG))
        for dy, dx in ((0, 1), (1, 0)):
            A, B = fid[:H - dy, :W - dx], fid[dy:, dx:]
            sil = (A >= 0) != (B >= 0)
            both = (A >= 0) & (B >= 0) & (A != B)
            crease = both & ((n[np.maximum(A, 0)] * n[np.maximum(B, 0)]).sum(-1) < cos_c)
            e = sil | crease
            edge[:H - dy, :W - dx] |= e
            edge[dy:, dx:] |= e
        # a part in front of another: the depth's second difference spikes there, but not on a smooth surface seen
        # at a grazing angle (a first difference would draw a thick band along it)
        with np.errstate(invalid="ignore"):                  # inf off the part
            for ax in (0, 1):
                d2 = np.abs(np.diff(zf, 2, axis=ax))
                e = np.nan_to_num(d2, nan=0.0) > 0.01 * span
                if ax == 0:
                    edge[1:-1, :] |= e
                else:
                    edge[:, 1:-1] |= e
        img[edge] = img[edge] * 0.25
    return img.reshape(height, ssaa, width, ssaa, 3).mean(axis=(1, 3))
