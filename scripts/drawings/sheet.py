"""Drawing-sheet framework for code-generated part drawings (ezdxf DXF -> vector PDF).

Everything is drawn in model space in millimetres at 1:1 on an A1 landscape sheet. The PDF is rendered
from the same DXF (matplotlib backend; the PyMuPDF backend writes a broken content stream with the
installed PyMuPDF), so the two cannot disagree. GD&T symbols are drawn as geometry, not glyphs, so they
print the same with any font.
"""
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Vec2

SHEETS = {"A2": (594.0, 420.0), "A1": (841.0, 594.0)}
SHEET_W, SHEET_H = SHEETS["A1"]            # landscape
FRAME = (20.0, 10.0, SHEET_W - 10.0, SHEET_H - 10.0)      # left binding margin 20, others 10
FONT = "arial.ttf"
TEXT_W = 0.85
H = 3.5

LAYERS = {
    "OUTLINE": (7, 50, "CONTINUOUS"),
    "THIN": (7, 25, "CONTINUOUS"),
    "HIDDEN": (7, 25, "HID"),
    "CENTER": (7, 18, "CTR"),
    "CUTPLANE": (7, 18, "CTR"),
    "HATCH": (7, 18, "CONTINUOUS"),
    "DIM": (7, 18, "CONTINUOUS"),
    "NOTE": (7, 25, "CONTINUOUS"),
    "BORDER": (7, 70, "CONTINUOUS"),
    "TBLOCK": (7, 35, "CONTINUOUS"),
    "TBTHIN": (7, 18, "CONTINUOUS"),
}


def new_doc():
    doc = ezdxf.new("R2018", setup=False)
    doc.units = ezdxf.units.MM
    doc.header["$MEASUREMENT"] = 1
    doc.header["$INSUNITS"] = 4
    doc.header["$LTSCALE"] = 1.0
    doc.linetypes.add("HID", pattern=[4.5, 3.0, -1.5], description="Hidden __ __ __")
    doc.linetypes.add("CTR", pattern=[24.0, 16.0, -3.0, 2.0, -3.0], description="Centre ____ _ ____")
    for name, (col, lw, lt) in LAYERS.items():
        doc.layers.add(name, color=col, lineweight=lw, linetype=lt)
    doc.styles.new("GEO", dxfattribs={"font": FONT})
    ds = doc.dimstyles.new("GEO")
    ds.dxf.dimtxsty = "GEO"
    for k, v in dict(dimtxt=H, dimasz=2.5, dimexe=1.5, dimexo=1.0, dimgap=0.8, dimtad=1, dimjust=0, dimtih=0, dimtoh=0,
                     dimdec=1, dimzin=8, dimdsep=ord("."), dimclrd=7, dimclre=7, dimclrt=7, dimlwd=18, dimlwe=18,
                     dimlfac=1.0, dimtix=0, dimtofl=1).items():
        ds.set_dxf_attrib(k, v)
    doc.header["$DIMSTYLE"] = "GEO"
    doc.header["$TEXTSTYLE"] = "GEO"
    doc.header["$LWDISPLAY"] = 1
    return doc, doc.modelspace()


# ---------------------------------------------------------------- primitives
def line(msp, a, b, layer="OUTLINE"):
    msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": layer})


def pline(msp, pts, layer="OUTLINE", close=False):
    msp.add_lwpolyline([tuple(p) for p in pts], close=close, dxfattribs={"layer": layer})


def rect(msp, x0, y0, x1, y1, layer="OUTLINE"):
    pline(msp, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer, close=True)


def circle(msp, c, r, layer="OUTLINE"):
    msp.add_circle(tuple(c), r, dxfattribs={"layer": layer})


def text(msp, s, x, y, h=H, layer="NOTE", align="LEFT", rot=0.0):
    if not s:
        return None
    t = msp.add_text(s, height=h, rotation=rot, dxfattribs={"layer": layer, "style": "GEO", "width": TEXT_W})
    t.set_placement((x, y), align=getattr(TextEntityAlignment, align))
    return t


def text_width(s, h=H):
    from ezdxf.fonts import fonts
    font = fonts.make_font(FONT, h, TEXT_W)
    return font.text_width(s)


def lines_block(msp, rows, x, y, h=2.5, gap=1.55, layer="NOTE"):
    for r in rows:
        text(msp, r, x, y, h, layer)
        y -= h * gap
    return y


def arrowhead(msp, tip, frm, size=2.5, layer="DIM"):
    d = (Vec2(tip) - Vec2(frm)).normalize()
    n = Vec2(-d.y, d.x)
    base = Vec2(tip) - d * size
    w = size / 6.0
    msp.add_solid([tuple(tip), tuple(base + n * w), tuple(base - n * w)], dxfattribs={"layer": layer})


def leader(msp, target, knee, label=None, shoulder=4.0, h=H):
    line(msp, target, knee, "DIM")
    arrowhead(msp, target, knee)
    if label:
        sgn = 1 if knee[0] >= target[0] else -1
        end = (knee[0] + sgn * shoulder, knee[1])
        line(msp, knee, end, "DIM")
        text(msp, label, end[0] + sgn * 1.0, knee[1], h, "NOTE", "MIDDLE_LEFT" if sgn > 0 else "MIDDLE_RIGHT")
        return end
    return knee


def centre_mark(msp, c, size=4.0):
    line(msp, (c[0] - size, c[1]), (c[0] + size, c[1]), "CENTER")
    line(msp, (c[0], c[1] - size), (c[0], c[1] + size), "CENTER")


def dim_lin(msp, p1, p2, base, angle=0.0, txt="<>", text_at=None):
    """Linear dimension; text_at moves the text (e.g. beside a dimension too narrow to hold it)."""
    d = msp.add_linear_dim(base=tuple(base), p1=tuple(p1), p2=tuple(p2), angle=angle, text=txt, dimstyle="GEO",
                           location=None if text_at is None else tuple(text_at), dxfattribs={"layer": "DIM"})
    d.render()
    return d


def hatch_polygon(msp, exterior, holes=(), angle=45.0, scale=0.8):
    h = msp.add_hatch(dxfattribs={"layer": "HATCH"})
    h.set_pattern_fill("ANSI31", scale=scale, angle=angle - 45.0)
    h.paths.add_polyline_path([tuple(p) for p in exterior], is_closed=True)
    for hole in holes:
        h.paths.add_polyline_path([tuple(p) for p in hole], is_closed=True)
    return h


# ---------------------------------------------------------------- GD&T (ASME Y14.5 style)
def _symbol(msp, kind, cx, cy, s=2.2):
    """Characteristic symbol drawn as geometry, centred at (cx, cy), about 2 s tall."""
    if kind == "pos":
        circle(msp, (cx, cy), s * 0.75, "NOTE")
        line(msp, (cx - s, cy), (cx + s, cy), "NOTE")
        line(msp, (cx, cy - s), (cx, cy + s), "NOTE")
    elif kind == "cyl":
        circle(msp, (cx, cy), s * 0.6, "NOTE")
        for dx in (-s * 0.6, s * 0.6):
            line(msp, (cx + dx - s * 0.45, cy - s), (cx + dx + s * 0.45, cy + s), "NOTE")
    elif kind == "perp":
        line(msp, (cx - s, cy - s * 0.9), (cx + s, cy - s * 0.9), "NOTE")
        line(msp, (cx, cy - s * 0.9), (cx, cy + s), "NOTE")
    elif kind == "flat":
        pline(msp, [(cx - s, cy - s * 0.6), (cx + s * 0.5, cy - s * 0.6), (cx + s, cy + s * 0.6), (cx - s * 0.5, cy + s * 0.6)],
              "NOTE", close=True)
    elif kind == "par":
        for dx in (-s * 0.45, s * 0.45):
            line(msp, (cx + dx - s * 0.4, cy - s), (cx + dx + s * 0.4, cy + s), "NOTE")
    elif kind == "runout":
        line(msp, (cx - s * 0.7, cy - s), (cx + s * 0.7, cy + s), "NOTE")
        arrowhead(msp, (cx + s * 0.7, cy + s), (cx - s * 0.7, cy - s), size=s * 0.9, layer="NOTE")


def _fcf_cells(x, tol_s, datums, h):
    hh = h * 2.0
    xs = [x, x + hh, x + hh + text_width(tol_s, h) + 3.0]
    for dlab in datums:
        xs.append(xs[-1] + max(hh, text_width(dlab, h) + 3.0))
    return xs


def fcf_width(tol, datums=(), h=3.0, diameter=False):
    xs = _fcf_cells(0.0, ("Ø" if diameter else "") + tol, datums, h)
    return xs[-1]


def fcf(msp, x, y, kind, tol, datums=(), h=3.0, diameter=False):
    """Feature control frame with its lower-left corner at (x, y). Returns the right end x."""
    hh = h * 2.0
    tol_s = ("Ø" if diameter else "") + tol
    xs = _fcf_cells(x, tol_s, datums, h)
    rect(msp, xs[0], y, xs[-1], y + hh, "NOTE")
    for xx in xs[1:-1]:
        line(msp, (xx, y), (xx, y + hh), "NOTE")
    _symbol(msp, kind, (xs[0] + xs[1]) / 2, y + hh / 2, s=h * 0.62)
    text(msp, tol_s, (xs[1] + xs[2]) / 2, y + hh / 2, h, "NOTE", "MIDDLE_CENTER")
    for dlab, a, b in zip(datums, xs[2:-1], xs[3:]):
        text(msp, dlab, (a + b) / 2, y + hh / 2, h, "NOTE", "MIDDLE_CENTER")
    return xs[-1]


def datum_feature(msp, letter, on_pt, out_dir, length=8.0, box=6.0, h=3.5):
    """Datum feature symbol: filled triangle on the feature at on_pt, a stem along out_dir, a boxed letter."""
    d = Vec2(out_dir).normalize()
    n = Vec2(-d.y, d.x)
    p = Vec2(on_pt)
    msp.add_solid([tuple(p - n * 1.6), tuple(p + n * 1.6), tuple(p + d * 2.6)], dxfattribs={"layer": "NOTE"})
    top = p + d * length
    line(msp, p + d * 2.6, top, "NOTE")
    c = top + d * box / 2
    rect(msp, c.x - box / 2, c.y - box / 2, c.x + box / 2, c.y + box / 2, "NOTE")
    text(msp, letter, c.x, c.y, h, "NOTE", "MIDDLE_CENTER")


def surface_finish(msp, tip, value):
    t = Vec2(tip)
    a = t + Vec2(-2.5, 4.3)
    b = t + Vec2(5.0, 8.7)
    pline(msp, [a, t, b], "NOTE")
    line(msp, a, a + Vec2(5.0, 0), "NOTE")
    text(msp, value, b.x + 0.8, b.y - 2.0, 2.5, "NOTE")


def view_title(msp, title, scale, x, y):
    t = text(msp, title, x, y, 4.0, "NOTE", "BOTTOM_CENTER")
    w = text_width(title, 4.0) if t else 0
    line(msp, (x - w / 2, y - 1.0), (x + w / 2, y - 1.0), "NOTE")
    text(msp, scale, x, y - 5.0, 2.5, "NOTE", "BOTTOM_CENTER")


def section_marks(msp, a, b, letter, look):
    """Cutting-plane line from a to b with arrows showing the viewing direction `look` (unit 2D vector)."""
    line(msp, a, b, "CUTPLANE")
    lk = Vec2(look).normalize()
    for p in (Vec2(a), Vec2(b)):
        tip = p + lk * 7
        line(msp, p, tip, "DIM")
        arrowhead(msp, tip, p, size=3.0)
        text(msp, letter, (tip + lk * 3).x, (tip + lk * 3).y, 4.0, "NOTE", "MIDDLE_CENTER")


def third_angle_symbol(msp, x, y, s=6.0):
    """Third-angle projection symbol (truncated cone: side view left of the end view)."""
    pline(msp, [(x, y - s * 0.35), (x + s * 1.2, y - s * 0.6), (x + s * 1.2, y + s * 0.6), (x, y + s * 0.35)], "NOTE", close=True)
    cx = x + s * 2.4
    circle(msp, (cx, y), s * 0.6, "NOTE")
    circle(msp, (cx, y), s * 0.35, "NOTE")
    line(msp, (x - 1.5, y), (cx + s * 0.9, y), "CENTER")


# ---------------------------------------------------------------- sheet furniture
def frame(msp):
    x0, y0, x1, y1 = FRAME
    rect(msp, 0, 0, SHEET_W, SHEET_H, "TBTHIN")
    rect(msp, x0, y0, x1, y1, "BORDER")
    nc, nr = 12, 8
    for i in range(nc):
        xm = x0 + (x1 - x0) * (i + 0.5) / nc
        text(msp, str(i + 1), xm, y1 + 2.5, 3.0, "TBTHIN", "MIDDLE_CENTER")
        text(msp, str(i + 1), xm, y0 - 2.5, 3.0, "TBTHIN", "MIDDLE_CENTER")
        if i:
            x = x0 + (x1 - x0) * i / nc
            line(msp, (x, y1), (x, y1 + 5), "TBTHIN")
            line(msp, (x, y0), (x, y0 - 5), "TBTHIN")
    for j in range(nr):
        ym = y0 + (y1 - y0) * (j + 0.5) / nr
        text(msp, "ABCDEFGH"[j], x0 - 2.5, ym, 3.0, "TBTHIN", "MIDDLE_CENTER")
        text(msp, "ABCDEFGH"[j], x1 + 2.5, ym, 3.0, "TBTHIN", "MIDDLE_CENTER")
        if j:
            y = y0 + (y1 - y0) * j / nr
            line(msp, (x0, y), (x0 - 5, y), "TBTHIN")
            line(msp, (x1, y), (x1 + 5, y), "TBTHIN")


def title_block(msp, info, w=200.0, hgt=72.0):
    """Title block in the lower-right corner. info: title lines, number, rev, fields [(label, value)], revisions."""
    x1, y0 = FRAME[2], FRAME[1]
    x0, y1 = x1 - w, y0 + hgt
    rect(msp, x0, y0, x1, y1, "TBLOCK")
    # title (top band)
    line(msp, (x0, y1 - 16), (x1, y1 - 16), "TBLOCK")
    ty = y1 - 5.0
    for ln in info["title"]:
        text(msp, ln, x0 + 3, ty, 4.0, "TBTHIN", "MIDDLE_LEFT")
        ty -= 6.0
    # fields grid (2 columns)
    fields = info["fields"]
    rows = (len(fields) + 1) // 2
    fh = (y1 - 16 - (y0 + 14)) / rows
    xm = x0 + w / 2
    line(msp, (xm, y0 + 14), (xm, y1 - 16), "TBTHIN")
    for i, (k, v) in enumerate(fields):
        col, row = i % 2, i // 2
        xa = x0 if col == 0 else xm
        yy = y1 - 16 - fh * (row + 1)
        line(msp, (x0, yy), (x1, yy), "TBTHIN")
        text(msp, k, xa + 1.5, yy + fh - 1.8, 1.8, "TBTHIN", "MIDDLE_LEFT")
        text(msp, v, xa + 1.5, yy + fh / 2 - 1.0, 2.6, "TBTHIN", "MIDDLE_LEFT")
    # number and rev (bottom band)
    line(msp, (x0, y0 + 14), (x1, y0 + 14), "TBLOCK")
    rx = x1 - 22
    line(msp, (rx, y0), (rx, y0 + 14), "TBLOCK")
    text(msp, "DRAWING No.", x0 + 1.5, y0 + 12.0, 1.8, "TBTHIN", "MIDDLE_LEFT")
    text(msp, info["number"], (x0 + rx) / 2, y0 + 6.0, 6.0, "TBLOCK", "MIDDLE_CENTER")
    text(msp, "REV", rx + 1.5, y0 + 12.0, 1.8, "TBTHIN", "MIDDLE_LEFT")
    text(msp, info["rev"], (rx + x1) / 2, y0 + 6.0, 6.0, "TBLOCK", "MIDDLE_CENTER")
    # revision table above the title block
    ry = y1
    for rev, date, desc in info.get("revisions", []):
        rect(msp, x0, ry, x1, ry + 6, "TBTHIN")
        text(msp, rev, x0 + 4, ry + 3, 2.3, "TBTHIN", "MIDDLE_CENTER")
        text(msp, date, x0 + 20, ry + 3, 2.3, "TBTHIN", "MIDDLE_CENTER")
        text(msp, desc, x0 + 34, ry + 3, 2.3, "TBTHIN", "MIDDLE_LEFT")
        line(msp, (x0 + 8, ry), (x0 + 8, ry + 6), "TBTHIN")
        line(msp, (x0 + 32, ry), (x0 + 32, ry + 6), "TBTHIN")
        ry += 6
    text(msp, "REV   DATE           DESCRIPTION", x0 + 1.5, ry + 2.5, 2.0, "TBTHIN", "MIDDLE_LEFT")
    return x0, y0, x1, y1


def table(msp, x0, y_top, header, rows, widths, rh=5.5, h=2.4):
    """Simple grid table with a header row. Returns the bottom y."""
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)
    n = len(rows) + 1
    yb = y_top - rh * n
    rect(msp, xs[0], yb, xs[-1], y_top, "TBLOCK")
    for xx in xs[1:-1]:
        line(msp, (xx, yb), (xx, y_top), "TBTHIN")
    for i in range(1, n):
        line(msp, (xs[0], y_top - rh * i), (xs[-1], y_top - rh * i), "TBLOCK" if i == 1 else "TBTHIN")
    for lab, a, b in zip(header, xs[:-1], xs[1:]):
        text(msp, lab, (a + b) / 2, y_top - rh / 2, h, "TBTHIN", "MIDDLE_CENTER")
    for i, row in enumerate(rows):
        yy = y_top - rh * (i + 1.5)
        for val, a, b in zip(row, xs[:-1], xs[1:]):
            text(msp, val, (a + b) / 2, yy, h, "TBTHIN", "MIDDLE_CENTER")
    return yb


# ---------------------------------------------------------------- output
def save(doc, out_base, title="", author="N. Robson", subject=""):
    """Write the DXF and a vector PDF rendered from that same DXF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext, config
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend, to_matplotlib_path
    from matplotlib.patches import PathPatch

    class Backend(MatplotlibBackend):
        # detect_holes=True turns the overlapping contours of a glyph like Ø into holes (half-black)
        def draw_filled_paths(self, paths, properties):
            try:
                patch = PathPatch(to_matplotlib_path(paths, detect_holes=False), color=properties.color, linewidth=0,
                                  fill=True, zorder=self._get_z())
            except ValueError:
                return
            self.ax.add_patch(patch)

    msp = doc.modelspace()
    msp.reset_limits((0, 0), (SHEET_W, SHEET_H))
    msp.reset_extents((0, 0, 0), (SHEET_W, SHEET_H, 0))
    doc.saveas(out_base + ".dxf")
    fig = plt.figure(figsize=(SHEET_W / 25.4, SHEET_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    backend = Backend(ax, adjust_figure=False)
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE, color_policy=config.ColorPolicy.COLOR,
                               lineweight_policy=config.LineweightPolicy.ABSOLUTE, lineweight_scaling=72.0 / 25.4,
                               min_lineweight=0.13)
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(msp, finalize=False)
    ax.set_xlim(0, SHEET_W)
    ax.set_ylim(0, SHEET_H)
    ax.set_aspect("equal")
    fig.savefig(out_base + ".pdf", format="pdf", metadata={"Title": title, "Author": author, "Subject": subject})
    fig.savefig(out_base + ".png", dpi=110)
    plt.close(fig)

