"""Bill of materials with cost for the left front corner and the front axle.

Prices are researched inputs (data/bom.csv, with source, date and status per line). Quantities that
depend on the design are filled from the analysis outputs, so the BOM follows the design:
  blank_kg   the upright stock blank: the STEP envelope + 3 mm a side, on the thinnest standard plate that holds it
  rod_ends   rod ends per corner, counted from data/misalignment.csv
  tube:<OD x wall>   feet of that tube: member lengths (data/member_checks.csv) less the rod ends, + 50 mm cut each
Totals: "firm" = lines priced at the exact part; "estimated" adds the closest and estimate lines. Quote
lines are listed, not totalled. The right corner differs only in the caliper (RH part number and price).

Run: python bom.py   Writes data/bom_costed.csv and notes/bom.md
"""
import csv
import hashlib
import json

import stamps
from common import DATA, NOTES, ROOT, read_version
from forces import FOS_ULTIMATE


def bom_sha():
    """bom.csv's content fingerprint (line endings ignored): bom_costed.csv's header records it, and the requirements
    matrix refuses a costed BOM made from another bom.csv."""
    return hashlib.sha256((DATA / "bom.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]

FX = {"USD": 1.0, "CAD": 0.72, "GBP": 1.27, "AUD": 0.66, "EUR": 1.08}   # to USD, ASSUMED (not looked up)
PLATE_IN = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)                    # common 7075 plate thicknesses, in
STOCK_MM = 3.0                                                           # machining allowance a side
RHO_7075 = 2810.0
CUT_MM = 50.0                                                            # cut allowance per tube
MM_M8 = {"part": "Aurora MM-M8", "ult_N": 13591.0, "usd": 36.96,
         "source": "https://www.motion.com/products/sku/00140172 (price from a search snippet; Timken lists the CM-M8 at the same 13591 N)"}
S03_MIN = FOS_ULTIMATE


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def upright_blank():
    """Stock blank for the upright: axis-aligned envelope of the STEP (the spin axis is within 1.3 deg of
    global Y, the plate thickness direction) + STOCK_MM a side; thickness rounded up to a standard plate."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(ROOT / "cad" / "upright" / "upright_L.step"))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    big = max(vols, key=lambda v: gmsh.model.occ.getMass(*v))
    x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(*big)
    gmsh.finalize()
    env = sorted([x1 - x0, y1 - y0, z1 - z0])
    need_t = env[0] + 2 * STOCK_MM
    t_in = next(t for t in PLATE_IN if t * 25.4 >= need_t)
    w, length = env[1] + 2 * STOCK_MM, env[2] + 2 * STOCK_MM
    kg = t_in * 25.4 * w * length * 1e-9 * RHO_7075
    return {"envelope_mm": env, "plate_in": t_in, "w_mm": w, "l_mm": length, "kg": kg}


def rod_end_count():
    return sum(1 for r in read_rows(DATA / "misalignment.csv") if "rod end" in r["part"].lower())


def tube_feet(members, checks):
    """Feet of tube per section key 'OD x wall' (mm, as in members.csv). A wishbone leg has one rod end (its
    outboard end is welded to the ball-joint node); the tie rod and pushrod have one at each end."""
    length = {c["member"]: float(c["length_mm"]) for c in checks}
    feet = {}
    for m in members:
        key = f"{float(m['tube_od_mm']):g} x {float(m['tube_wall_mm']):g}"
        n_ends = 1 if m["out"] in ("UCA_O", "LCA_O") else 2
        tube_mm = length[m["member"]] - n_ends * float(m["rod_end_shank_mm"]) + CUT_MM
        feet[key] = feet.get(key, 0.0) + tube_mm / 304.8
    return feet


@stamps.producer("bom")
def main():
    rows = read_rows(DATA / "bom.csv")
    members = read_rows(DATA / "members.csv")
    checks = read_rows(DATA / "member_checks.csv")
    blank = upright_blank()
    n_re = rod_end_count()
    feet = tube_feet(members, checks)
    out = []
    for r in rows:
        q = r["qty"]
        if q == "blank_kg":
            qty = blank["kg"]
        elif q == "rod_ends":
            qty = n_re
        elif q.startswith("tube:"):
            qty = feet[q.split(":", 1)[1]]
        else:
            qty = float(q)
        unit_usd = float(r["unit_price"]) * FX[r["currency"]]
        out.append(dict(r, qty=qty, unit_usd=unit_usd, ext_usd=qty * unit_usd))

    def total(group_filter, statuses):
        return sum(o["ext_usd"] for o in out if group_filter(o["group"]) and o["status"] in statuses)

    corner = lambda g: g not in ("car", "right")            # noqa: E731
    car = lambda g: g == "car"               # noqa: E731
    firm = {"priced", "in_house"}
    est = firm | {"closest", "estimate"}
    lh_firm, lh_est = total(corner, firm), total(corner, est)
    rh_delta = next(o["ext_usd"] for o in out if o["id"] == "H4R") - next(o["ext_usd"] for o in out if o["id"] == "H4")
    car_firm, car_est = total(car, firm), total(car, est)
    axle_firm = 2 * lh_firm + rh_delta + car_firm
    axle_est = 2 * lh_est + rh_delta + car_est
    quotes = [o for o in out if o["status"] == "quote"]
    part_kg = json.loads((ROOT / "cad" / "upright" / "upright_build.json").read_text())["masses_kg"]["upright"]
    big4 = sum(o["ext_usd"] for o in out if o["id"] in ("W1", "W2", "D1", "J5")) / lh_est

    # cost-down check: the plain MM-M8 rod end against S-03 on the six members the force solver checks
    peak = max(max(abs(float(c["max_tension_N"])), abs(float(c["max_compression_N"]))) for c in checks)
    crit = max(checks, key=lambda c: max(abs(float(c["max_tension_N"])), abs(float(c["max_compression_N"]))))["member"]
    fos_mm = MM_M8["ult_N"] / peak
    re_line = next(o for o in out if o["id"] == "J5")
    saving = n_re * (re_line["unit_usd"] - MM_M8["usd"])

    with open(DATA / "bom_costed.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# generated by scripts/bom.py; hardpoints {read_version(DATA / 'hardpoints.csv')}; bom.csv {bom_sha()}; "
                "USD at the FX in bom.py (assumed)\n")
        w = csv.writer(f)
        w.writerow(["id", "group", "part", "spec", "qty", "unit", "unit_usd", "ext_usd", "status", "source"])
        for o in out:
            w.writerow([o["id"], o["group"], o["part"], o["spec"], f"{o['qty']:.6g}", o["unit"], f"{o['unit_usd']:.4f}",
                        f"{o['ext_usd']:.2f}", o["status"], o["source"]])

    L = ["# Bill of materials with cost", "",
         "Generated by `scripts/bom.py` from `data/bom.csv` (prices, sources, status) and the analysis outputs "
         "(quantities). Do not edit by hand.", "",
         f"Prices read on 2026-10-27, in USD. Other currencies are converted at ASSUMED rates "
         f"({', '.join(f'{k} {v}' for k, v in FX.items() if k != 'USD')} USD), not looked up. Machining time, freight, "
         "tax and duty are not included.", "",
         "## Totals", "",
         "| Scope | Firm (exact part priced) | Estimated (adds closest and estimate lines) |",
         "|---|---|---|",
         f"| Left front corner | {lh_firm:,.0f} | {lh_est:,.0f} |",
         f"| Right front corner (RH caliper {rh_delta:+.2f}) | {lh_firm + rh_delta:,.0f} | {lh_est + rh_delta:,.0f} |",
         f"| Per car (front axle): rack, retainer | {car_firm:,.0f} | {car_est:,.0f} |",
         f"| **Front axle** | **{axle_firm:,.0f}** | **{axle_est:,.0f}** |", "",
         f"Not in either total ({len(quotes)} quote lines): " + "; ".join(f"{o['id']} {o['part']}" for o in quotes) + ".", "",
         "## Left front corner and per-car lines", "",
         "Group `right` is the right corner's only different part (not in the left total).", "",
         "| Id | Part | Spec | Qty | Unit USD | Ext. USD | Status |", "|---|---|---|---|---|---|---|"]
    for o in out:
        qty = f"{o['qty']:.1f} {o['unit']}" if o["unit"] in ("kg", "ft") else f"{o['qty']:g}"
        ext = "-" if o["status"] == "quote" else f"{o['ext_usd']:,.2f}"
        unit = "-" if o["status"] == "quote" else f"{o['unit_usd']:,.2f}"
        L.append(f"| {o['id']} | {o['part']} | {o['spec']} | {qty} | {unit} | {ext} | {o['status']} |")
    L += ["", "## How the design quantities were found", "",
          f"- **Upright blank:** STEP envelope {' x '.join(f'{e:.1f}' for e in blank['envelope_mm'])} mm, + {STOCK_MM:g} mm "
          f"a side -> {blank['plate_in']:g} in plate, {blank['w_mm']:.0f} x {blank['l_mm']:.0f} mm, {blank['kg']:.2f} kg "
          f"of 7075 for a {part_kg:.2f} kg part (about {blank['kg'] / part_kg:.0f} to 1 buy-to-fly; normal for a 3+2-axis "
          "machined upright).",
          f"- **Rod ends:** {n_re} per corner (the rod-end rows of `data/misalignment.csv`: wishbones 4, tie rod 2, "
          "pushrod 2, drop link 2). Each takes one M8 10.9 bolt (length from the bolt schedule in `notes/fasteners.md`), an "
          "all-metal lock nut (nylon inserts are limited to 80 C) and a jam nut on its shank (R-10); the drop link's lever "
          "end also takes two 24 mm washers (R-08). Bolts into parts are locked by a screw-locking Heli-Coil (the lower "
          "ball joint) or drilled heads wired in pairs (the caliper and rack bracket bolts) (R-09).",
          "- **Tube:** member length from `data/member_checks.csv` less the two rod-end shanks, + "
          f"{CUT_MM:g} mm cut allowance per tube: " + ", ".join(f"{k} mm {v:.1f} ft" for k, v in feet.items()) + ".", "",
          "## Findings from pricing", "",
          "- **Rod-end part number corrected.** The data said Aurora `AM-M8`, which does not exist. The 33135 N rating "
          "the checks use belongs to the **AM-M8T** (PTFE lined, heat-treated alloy steel, 15.88 mm ball; Timken item "
          "page). `data/members.csv` and `data/joints_misalignment.csv` now name the AM-M8T; no FoS changes.",
          f"- **Cost-down option (not taken):** the plain {MM_M8['part']} ({MM_M8['ult_N']:.0f} N ultimate, {MM_M8['usd']:.2f} USD) "
          f"against the highest rod-end load in the six checked members ({peak:.0f} N, {crit}) gives FoS {fos_mm:.2f} "
          f"on S-03's >= {S03_MIN} target ({'passes' if fos_mm >= S03_MIN else 'fails'}), saving about "
          f"{saving:,.0f} USD a corner. Not taken: it has an unlined steel-on-steel race (greased, wears and gains play) "
          "and a low-carbon body, and the drop-link loads are not in the force solver. Worth a quote on the AM-M8T first.",
          "- **Spring rate (settled 2026-11-04):** the spring had been picked as 25 N/mm (143 lb/in) from an assumed "
          "2.5 N/mm metric list, softer than anything sold for the TTX25: Ohlins and Hyperco both start at 150 lb/in. "
          "`vehicle.json` now holds the real lb/in steps, so the calcs pick 150 lb/in (26.3 N/mm, a Hyperco FSAE spring): "
          "2.91 Hz against the 2.66-2.94 Hz window, and the loads, FEA and fatigue were re-run on it.",
          f"- **Biggest lines:** wheel, tyre, damper and rod ends are {big4:.0%} of the estimated corner. The 3 in "
          "7075-T7351 plate needs a quote (only T651 is listed retail).", ""]
    (NOTES / "bom.md").write_text("\n".join(L), encoding="utf-8")
    print(f"BOM: left corner {lh_firm:,.0f} firm / {lh_est:,.0f} estimated USD; front axle {axle_firm:,.0f} / "
          f"{axle_est:,.0f}; {len(quotes)} quote lines; blank {blank['plate_in']:g} in, {blank['kg']:.1f} kg; "
          f"{n_re} rod ends; MM-M8 FoS {fos_mm:.2f}")


if __name__ == "__main__":
    main()
