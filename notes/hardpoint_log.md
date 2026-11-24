# Hardpoint iteration log

`data/hardpoints.csv` is the single source of truth (PLAN s5). Each version below says what moved, why, and what it did to the targets. Snapshots of earlier versions are kept as `data/hardpoints_v<version>.csv`.

Axes: ISO 8855 (x forward, y left, z up). Origin: ground plane at static ride height, on the front-axle line, on the vehicle centreline. Left front corner; the right corner is the mirror image (y -> -y).

---

## v1.0: hand-derived first set (2026-10-24)

**How the points were set**

- **Wheel and kingpin:** contact patch at y = 610 (half of the 1220 mm track) and static camber -1.25 deg, so WC = (0, 605.6, 200.0).
  - The kingpin passes x = -2.5 mm at hub height, with caster 5.0 deg and KPI 7.0 deg.
  - Ball joints sit at z = 110 / 290, about 90 mm from the hub axis, to stay inside a 10 in rim.
  - Result: scrub 16.5 mm and trail 15.0 mm, both by construction.
- **Wishbones (2D front-view construction):**
  - The targets were a 35 mm roll centre and about 70 % camber recovery, which gives a front-view swing arm (FVSA) of t / (2 x 0.7) = 870 mm. The instant centre therefore sits at (y, z) = (-260, 50).
  - The lower arm line runs through LCA_O and the IC; its inboard pivot is at y = 200, z = 82.9.
  - The upper arm line runs through UCA_O and the IC; its inboard pivot is at y = 230, z = 193.8.
  - The inboard axes are parallel to x, so anti-dive is zero.
- **Steering:**
  - Front-steer arm 57 mm long, TR_O at (60, 586, 140), low rack at x = +60.
  - TR_I came from the "tie rod on the line through the two inboard pivots and through the IC" construction.
- **Pushrod / rocker:**
  - The pushrod picks up on the upright at PR_O (-15, 557, 145), F1 style, so all six links are true two-force members.
  - Rocker axis is along x at (-15, 225, 400). Pushrod arm 55 mm; damper arm 75 mm, pointing up.
  - The damper lies transversely on top of the chassis.
- **ARB:** a U-bar along y at (45, *, 330). The drop link runs from a 40 mm rocker arm to a 60 mm lever.

**Results** (`kinematics.py`):

| Metric | Result | Target / status |
|---|---|---|
| Camber recovery | 72 % | pass |
| RC height | 37 mm | pass |
| Caster / KPI / scrub / trail | | as designed |
| Bump steer | 0.35 deg | FAIL |
| Ackermann | 16 % | FAIL (target 40-60 %) |
| RC migration vs ground | 35 mm over ±25 mm | FAIL |
| Motion ratio | 1.01 | damper stroke 60.3 mm, more than the TTX25's 57 mm |
| ARB installation ratio | 10.9 deg bar twist per deg roll | bar would need only 2 N m/deg, impractically soft |

## RC-migration study (between v1.0 and v1.1)

- **First optimiser run** (`tune_hardpoints.py arms`): the inboard pickups were moved inside chassis bounds (LCA y >= 180, UCA y <= 270), with the ball joints sliding along the kingpin. The best ground-referenced migration over ±25 mm was 25.5 mm.
- **Trade-off sweep** (`tradeoff_rc_camber.py`, `plots/k08_tradeoff_rc_vs_camber.png`): the best achievable migration stays at roughly 25-28 mm for camber recovery anywhere from 40 % to 80 %. So within these bounds the limit is not a camber trade-off.
- **Chassis-width probe** (`probe_rc_vs_chassis_width.py`, `data/tradeoff_rc_chassis_width.csv`): the gate is the upper pickup.
  - Ground-referenced migration drops to about 15 mm only when the UCA inboard pickups move out to about 330 mm from the centreline, i.e. a 660 mm-wide upper frame at the front axle with a very short (230 mm) upper arm.
  - With the upper pickups there, the result depends on how close the LCA pickups come in: 15.6 mm at y = 180, 9.9 mm at 150, 5.7 mm at 120.
  - That narrow-bottom, wide-top frame section at the driver's feet is not what a conventional FSAE nose gives.
- **Key finding:** measured from the ground, most of the RC "migration" in heave is the car itself moving.
  - In 25 mm of bump the sprung mass, and its CG, drop 25 mm.
  - What sets roll per g is the CG-to-RC arm, so the migration target is judged relative to the sprung mass. v1.0 gives ±10 mm on that basis.
  - The ground-referenced value is still reported, with this explanation.
- **Decision:** keep the v1.0 wishbones. The narrow/wide chassis option is recorded for the Week 2 ghost-chassis check. **Navy to confirm the judging basis** at the checkpoint.

## v1.1: steering, rocker, ARB (2026-10-24)

| Point | v1.0 | v1.1 | Why |
|---|---|---|---|
| TR_O | (60, 586.0, 140) | (60, 601.6, 140) | Steering-arm angle about 24 deg outboard of x gives 50 % Ackermann at the hairpin lock (`tune_hardpoints.py steer`). The rod end now sits 4 mm inboard of the wheel centre plane, so the wheel moved to ET 35 to push the disc outboard. |
| TR_I | (60, 203.0, 94.0) | (60, 214.5, 99.6) | Re-solved for zero bump steer after the TR_O move (least squares on toe over ±25 mm). |
| PR_I | (-15, 264.5, 438.2) | (-15, 270.7, 440.3) | Pushrod arm 54.9 -> 61.2 mm and rotated -2.7 deg (`rocker`, `rockershape`). Static MR 0.93 with ±1.5 % variation keeps MR >= 0.90 over ±25 mm and the damper stroke at 55.4 / 57 mm. |
| DMP_C | (-15, 53.5, 475) | (-15, 53.3, 475) | Centres the ±30 mm travel inside the TTX25's 143-200 mm eye-to-eye range (0.8 mm margin each end). |
| ARB_R / ARB_L / ARB_P / ARB_A | 40 mm rocker arm, 60 mm lever | 18 mm rocker arm, 70 mm lever, bar axis at (55, *, 330) | Installation ratio 10.9 -> 3.9 deg/deg, so the required bar is about 17 N m/deg (a 3/4 in x 0.049 4130 tube class). Provisional; finalised with the Week 2 bar sizing. |

**Results:** every PLAN s3 target passes. See `notes/kinematics_results.md`.

## v1.2: pushrod pickup on the kingpin axis (2026-10-24)

QA found that under a pure vertical wheel load the tie rod carried about 25 % of the pushrod force (-672 N in LC1).

**Cause:** PR_O sat about 12 mm off the kingpin axis. The pushrod therefore put a steering moment about the kingpin, so every bump fed torque into the rack. It also meant steering changed the pushrod length, so ride height and motion ratio varied with steer.

| Point | v1.1 | v1.2 | Why |
|---|---|---|---|
| PR_O | (-15.0, 557.0, 145.0) | (1.874, 575.089, 150.0) | Exactly on the kingpin axis (LCA_O -> UCA_O) at z = 150, 40 mm above the lower ball joint. A force through the axis has no moment about it, and a point on the axis does not move when the wheel steers. |
| PR_I | (-15, 270.7, 440.3) | (-15, 269.0, 441.5) | Rocker re-tuned for MR 0.93 static, +2 % progression (`rockershape`). |

**Effect:**
- LC1 tie-rod force drops from 672 N to 134 N. What remains is the ground force's own moment about the kingpin (scrub and trail offsets).
- Motion ratio over ±25 mm of rack travel stays within 0.930-0.934.
- The cornering and braking cases now show the true steering-moment load. Before, the pushrod offset partly cancelled it.
  - Tie rod rises to 851 N tension in LC4.
  - LCA rear leg rises to 4.12 kN compression in LC4.
- All checks still pass.
- The wishbones did not change, so the v1.1 RC trade studies still apply.

## Code review and QA follow-ups (2026-10-24)

- **RC migration** is now scored the conventional way, from the ground: 34.8 mm over ±25 mm, **FAIL**, with the trade-off evidence above. The CG-to-RC arm change is reported separately as `rc_arm_change_heave`: 9.7 mm, PASS, and the proposed basis.
- **Buckling** now uses the Johnson parabola below the transition slenderness. All six members have L/r of 66-82, so Euler had been 20-40 % optimistic. Worst case: LCA rear leg, FoS 2.52 (target 2.0).
- **Envelope:** the force checks now keep each member's tension and compression extremes over both poses separately.
- **Solver:** iterates on an increment from the warm start, so round-off-sized rotations no longer break the finite-difference Jacobian. Warm starts are only used within 2 mm.
- **Damper force** is labelled an upper bound in LC2/LC4: the ARB link is assumed unloaded.
- **Brake output** now splits pad tangential force (into the caliper mount) from pad clamp force (μ = 0.45 assumed).

## Week 1 checkpoint decisions (2026-10-24, first pass, partly superseded below)

Navy asked for best judgement. The first pass judged RC migration on the CG-to-RC arm, picked the Braid ET 35 wheel, kept the low rack, and set the ARB baseline on the middle blade. The wide audit that followed overturned the RC basis and changed the ARB sizing (see the next sections and requirements.md D8/D9).

## Wide audit before Week 2 (2026-10-24)

Three independent auditors ran in parallel: engineering fundamentals, goal fit and plan, and traceability. The findings acted on:

- **Pushrod pickup:** on the upright, steering swung its lower rod end 23-30 deg, beyond the 18 deg rating. This came up while building the CAD poses, and the fundamentals audit found it independently. Fixed in v1.3.
- **RC basis:** the sprung-mass basis was reverted. RC height above ground sets the geometric load transfer: the balance moves ±2.1 points of TLLTD over ±25 mm. K-08a is now a recorded deviation, DV-1.
- **Ride frequency:** front 3.2 -> 2.8 Hz (requirement change RC-2). The roll and balance targets can now hold together, and 2.8 Hz sits inside public practice for aero-less cars.
- **Load cases:** inside wheel, pothole, steering effort, kerb strike and reverse braking added.
- **New checks:**
  - joint misalignment over the full envelope;
  - stepped-column buckling with the rod-end shanks;
  - bolts sheared on the thread root;
  - an independent 2D hand calc of LC1, which agrees with the 3D pushrod force to 0.1 %.
- **Tie-rod clearance:** the tie rod cleared the lower wishbone's front leg by only 4.6 mm at full bump and full lock. Fixed in v1.4.
- **Traceability:** requirement statuses are now generated (`notes/requirements_matrix.md`), every output carries a hardpoint version, and stale outputs are refused.

## v1.3: pushrod on the lower-wishbone node (2026-10-24)

| Point | v1.2 | v1.3 | Why |
|---|---|---|---|
| PR_O | (1.874, 575.089, 150.0) on the upright | (3.798, 556.102, 135.522) on the LCA | 35 mm from the lower ball joint, on the line from the ball joint to PR_I. The pushrod force passes through the ball-joint centre, steering cannot move the pickup, and the lower rod end tilts at most 0.2 deg. |
| PR_I | (-15, 269.0, 441.5) | (-15, 271.0, 440.0) | Rocker re-tuned for MR 0.93 static with +2 % progression (`tune_hardpoints.py`, `stage_pushrod_on_lca`). |

## v1.4: steering arm raised (2026-10-24)

| Point | v1.3 | v1.4 | Why |
|---|---|---|---|
| TR_O | (60, 601.6, 140) | (60, 600.7, 150) | Raised 10 mm. The gap between the tie rod and the LCA front leg grows from 4.6 to 12.9 mm at worst (full bump, full lock). Re-solved for 50 % Ackermann. |
| TR_I | (60, 214.5, 99.6) | (60, 216.8, 105.4) | Re-solved for zero bump steer. |

**Results on v1.4:** the requirements matrix shows 32 pass, 8 open (later weeks), 1 deviation (DV-1) and 1 action (helper spring).

**Still open for Week 2:**
- the rack and ARB against an explicit cockpit/driver envelope;
- rim and disc clearance with ET 35.

## v1.5: rim clearance at full lock, from the CAD sweep (2026-10-26)

**Found by the CAD.** The first mesh interference sweep (`scripts/interference.py`, 35 poses) showed the inboard lip of the 10 x 7 in rim swinging into the lower-wishbone legs from about 14 deg of steer, the upper-wishbone rear leg at full inner lock, and the tie rod at full outer lock. The ball joints sit 45-65 mm outboard of the lip, so at 25-29 deg of lock the lip moves tens of millimetres relative to the legs. The ARB bar also ran through the assumed driver's feet.

**What was tried.**
- Moving only the lower wishbone (ball-joint height, pickup height, leg spread) could clear the lip, but pushed the roll centre to about 76 mm and camber recovery to 86 %.
- A differential-evolution optimiser over both wishbones with rim clearance as a constraint (`tune_hardpoints.py`, `stage_arms_rim`) found no 7 in solution inside the chassis bounds (every variable at its limit, legs still touching).
- With a 6 in rim (inside Hoosier's current 6-8 in range for this tyre) it found a solution that keeps every kinematic target.

| Point | v1.4 | v1.5 | Why |
|---|---|---|---|
| LCA_F / LCA_R | (+/-200, 200, 82.9) | (+/-171.4, 195.0, 104.6) | Legs leave the rim higher and closer to the bottom centre, where steering moves the lip least. |
| UCA_F / UCA_R | (160 / -180, 230, 193.8) | (160 / -180, 208.8, 196.6) | Restores roll centre and camber recovery with the new lower arm. |
| LCA_O | (5.37, 580.0, 110) | (3.45, 577.3, 132) | Moved 22 mm up the kingpin axis (caster, KPI, scrub and trail unchanged). |
| UCA_O | (-10.37, 557.9, 290) | (-9.15, 559.6, 276) | Moved down the kingpin axis. Kingpin length 180 -> 144 mm. |
| TR_O / TR_I | (60, 600.7, 150) / (60, 216.8, 105.4) | (60, 593.2, 178.4) / (45.3, 208.65, 132.95) | Re-tuned for 45 % Ackermann with the tie rod 19 mm clear of the rim and the wishbone legs; zero bump steer. |
| PR_O / PR_I | 35 mm from the LBJ | 45 mm from the LBJ; PR_I (-15, 268.2, 440.7) | Rod end clear of the upright at full droop; rocker re-tuned (MR 0.93, +2 %). |
| ARB_P / ARB_A / ARB_L / ARB_R | bar at x = +55 (footwell), 18 mm rocker arm | bar at x = -135, 120 mm lever, 32 mm rocker arm | Out of the feet; rod end clear of the rocker pivot boss; ratio 4.0. |

**Consequences, handled.**
- Shorter kingpin and narrower legs raise the loads: the LCA rear leg went to 5.3 kN compression in LC4. Tube upsized to 3/4 x 0.035 in (buckling FoS 2.40). The lower ball joint carries 5.0 kN (FoS 5.75 on the COM-5).
- Camber recovery 61.9-62.6 % (was 70.5 %), roll centre 39 mm (was 35), still inside their bands.
- The last 1.4 mm of rack travel inside a +2 deg stop margin brought the rim within 2.8 mm of the lower wishbone at full bump; the stop moved to lock + 1 deg (RC-7).

**Result:** the CAD sweep shows 0 collisions and every non-jointed moving clearance >= 5 mm over heave -30..+30 mm x roll +/-2 deg x steer to the stop. The requirements matrix: 35 pass, 7 open (Week 3), 1 deviation (DV-1), 1 action (helper spring).

**Superseded in part by the Week 2 close audit (below):** the sweep's joint rules were per part, so the "0 collisions" above could hide a clash on a jointed part, and the caliper was never checked. The v1.5 optimiser also had no roll-centre migration in its cost.

## Week 2 close audit (2026-10-26)

Four independent AI review agents, each with a different question: re-derive the numbers from scratch; can the CAD sweep hide a clash; does the package serve its goal, follow its rules and say only true things; is the data-to-conclusion chain traceable. Every finding was re-checked before acting.

| Finding | Verdict | What was done |
|---|---|---|
| Every headline number (lock angle, corner load, ride frequency, LCA buckling, rack force) re-derived within 1 % | Confirmed good | none |
| The 6 in rim is outside Hoosier's range for the 43075 | Rejected: Hoosier's current table says 6-8 in (the agent used the 2018 launch table, 7-8 in) | Both sources cited; confirm with Hoosier before ordering |
| Damper has 0.8 mm of stroke left at full bump, no room for a bump rubber | Confirmed | v1.6 below; V-03 rewritten (RC-10) |
| ARB torsion bar size in no data file, never checked | Confirmed: only 1.24x the stiff-blade requirement | 1.0 x 0.049 in bar, S-08 / S-09 (RC-11) |
| Pushrod bracket in the CAD loads its plates sideways, unlike the force model | Confirmed | Rebuilt as knee plates welded to the node (D5) |
| Sweep joint rules per part; caliper never checked; coarse grid; sweep needs Fusion | Confirmed | Per-body rules, in-wheel checks, 279 poses with warp, committed meshes with an input fingerprint (RC-12) |
| DV-1 rests on a stale study; the design is 8 mm above the "best achievable" | Confirmed, then explained: the frontier needs upper pickups 60-70 mm wider. At the design's width the limit is 35.4 mm and v1.6 is on it | Both studies re-run at v1.6 with the rim constraint; k08 redrawn; DV-1 rewritten |
| SUMMARY hides that the work was done with an AI assistant; "audits" were AI agents; "a test so it cannot come back" untrue for the CAD defects | Confirmed | Disclosure added; wording fixed; `tests/test_week2.py` (9 regression tests) |
| The upright is a layout part, so "CAD done" overstates it; Week 3 overloaded | Confirmed | Status reworded; Week 3 re-planned (requirements.md s9) |
| Stale values (camber recovery range, 2 deg stop), typed rule values in the matrix | Confirmed | Fixed; R-01 to R-04 now read from data (RC-14) |

## v1.6: room for a bump rubber (2026-10-26)

| Point | v1.5 | v1.6 | Why |
|---|---|---|---|
| DMP_C | (-15, 53.3, 475) | (-15, 48.8, 475) | 4.5 mm further along the damper's own axis, so the motion ratio and every kinematic metric are unchanged. The damper is 4.5 mm longer at every position: 1.3 mm of stroke is left at +30 mm with a 4 mm bump rubber solid, and the damper tops out at 25.9 mm of droop, which becomes the droop stop (rule: 25 mm). |

**CAD detail found by the stricter sweep (same hardpoints).**
- **Caliper:** the v1.5 envelope sat inside the disc radius and overlapped the hub and disc. Rebuilt to straddle the disc edge, 35 mm wide. The disc moved onto a 5 mm hat, 14 mm outboard of the wheel centre, centred in the caliper. Result: caliper 3.5 mm inside the wheel face; disc 3.7 mm from the upright.
- **Pushrod bracket:** the knee plates clear the upright by more than 5 mm in every pose (the old bracket was at 1.7 mm).
- **Ball-joint spacers:** 2 -> 5 mm each side. The clevis ears had come within 1.3 mm of the wishbone legs, and at 2 mm the ear would have touched the node at about 8 deg of misalignment, below the joint's 12 deg rating.
- A shorter clevis was tried for disc clearance and rejected: its web then hit the node at full droop and full lock.

**Result:** 0 collisions, 0 warnings over 279 poses under per-body rules. Requirements matrix: 38 pass, 7 open (Week 3), 1 deviation (DV-1), no actions. Unsprung mass 9.62 kg. 131 tests pass.

## Second Week 2 audit: improvements and a re-check (2026-10-26)

Asked to look for room to improve, not only defects. Four AI review agents, each with a different question: an FSAE suspension lead's gap review; the upright and FEA plan; repository quality; an independent re-check of every v1.6 number. A research agent found real caliper data. Every finding was re-checked before acting.

| Finding | Verdict | What was done |
|---|---|---|
| v1.6 numbers (motion ratio, damper lengths, ARB stiffness and roll torque, knee offset, stack-up arithmetic) | Re-derived independently, all agree, except: | |
| Pushrod bracket knee stress is 161 MPa, not 70 (3 mm plates; the 70 was computed for 6 mm and never updated) | Confirmed | Plates 4 mm (about 123 MPa + stress concentration, Week 3 hand calc); D5 corrected (RC-18) |
| ARB bar's worst case is full warp, not 1.6 g roll: FoS 1.28 | Confirmed (174 N m) | 1.0 x 0.065 in, S-09 covers warp, FoS 1.58 (RC-17) |
| A 4 mm bump rubber stops travel at +31.4 mm, outside the checked envelope; v1.5 would have lost travel, not bottomed; droop only 0.9 mm over the rule | Confirmed | v1.7 below; V-03b checks where travel ends; D10 corrected (RC-15) |
| Stack-up missing chassis-pickup, rack-play, flex and rim tolerances (worst case about 5.3 mm vs a 5.2 mm gap) | Confirmed | Stop margin 1 -> 0.5 deg (gap 6.4 mm); full table in requirements s6; Week 3 Monte Carlo (RC-16) |
| Upright loads never computed: forces.py treats upright, hub and wheel as one body | Confirmed | `upright_loads.py`, Week 3a item 1 (started) |
| Lower ball-joint bolt in bending is overstressed if the clamped stack is ignored (the 5 mm spacers lengthened the span) | Confirmed as a risk | Week 3a item 2: COM-6 / AN6 or a clamped-stack proof |
| Caliper envelope was an assumption | Confirmed | Wilwood PS-1 from its published drawing, 180 mm disc (RC-19) |
| No aligning moment in the loads; LC4 "20 % beyond the ellipse" is about 10 %; "25 N m at the wheel" ambiguous | Confirmed | Pneumatic trail 15 mm (D12); wording fixed (RC-20) |
| Advanced additions: compliance and camber budget, tolerance Monte Carlo, all eight load cases in the FEA, stiffness requirements, fatigue on a spectrum, a test plan, a recruiter page | Adopted | Week 3a/3b plan (requirements s9) |
| Repository: no pinned dependencies, no CI, duplicated fingerprint code, stale docstrings, tests reading generated files without saying so, sweep text that could drift | Confirmed | `requirements.txt`, `.github/workflows/ci.yml`, `.gitignore`, `scripts/fingerprint.py`, docstrings and tests fixed |
| Use CalculiX from FreeCAD/PrePoMax | Not adopted: not installed and not pip-installable | gmsh + scikit-fem, cantilever benchmark within 0.7 % (D4, RC-21) |

## v1.7: droop margin and a real caliper (2026-10-26)

| Point | v1.6 | v1.7 | Why |
|---|---|---|---|
| DMP_C | (-15, 48.8, 475) | (-15, 49.8, 475) | 3.5 mm (not 4.5) along the damper axis from v1.5: droop to the top-out 27.0 mm (2 mm over the rule); a bump rubber with a 4.3-8.9 mm solid height stops travel between 25 and 30 mm (6 mm assumed: +28.1 mm). |

**Other v1.7 changes (same hardpoints otherwise):** steering stop at lock + 0.5 deg (rim gap 6.4 mm); ARB bar 1.0 x 0.065 in; pushrod clevis plates 4 mm; Wilwood PS-1 caliper on a 180 mm disc, its envelope trimmed to the drawing's outside radius (a straight box's corners reached into the rim barrel: the sweep caught it); hub flange r 60 mm; tyre aligning moment in the load cases (LCA rear leg 5.47 kN in LC4, buckling FoS 2.31).

**Result:** 0 collisions, 0 warnings over 279 poses. Requirements matrix: 37 pass, 7 open, 1 deviation (DV-1), 1 action (helper spring, slack 1.1 mm). Unsprung mass 9.49 kg. 134 tests pass.

## Week 3 starts: upright interface loads (2026-10-26)

`scripts/upright_loads.py` splits the wheel, hub and upright (one body in `forces.py`) into hub, caliper and upright free bodies for every load case, plus an upright version of the steering-abuse case (a kerb at the tyre's front, scaled to the full rack force). Every case balances to about 1e-14, which checks the split against the six-link solution.

**Finding:** the layout's SKF 71908 angular-contact pair carried 9-12 kN per row in LC2 and LC4 (static rating 8.5 kN): static safety 0.69. The rows are only about 38 mm apart while the tyre's side force acts on a 200 mm lever. **Fix:** an SKF 32008 X tapered roller pair, back to back (C0 71 kN each): worst s0 5.3 (D13, RC-22, new requirement S-10). Same hardpoints; the CAD hub and housing follow `vehicle.json` (bore 68 mm, housing -26.5..+17.5 mm). Sweep still 0 / 0; unsprung mass 9.79 kg; matrix 38 pass, 7 open, 1 deviation, 1 action; 135 tests pass.

## Week 3a: bearings, bolts, the detailed upright and its FEA (2026-10-26/27)

Same hardpoints (v1.7). What changed and why, in the order it was found:

| Finding | Evidence | Fix |
|---|---|---|
| Hub bearing life and fits never checked | Week 3a plan | `bearings.py`: L10 over an assumed season spectrum is millions of km (static capacity, not life, sized them); P6 cup seats go slack at +60 C, so Loctite 641 + a central rib hold the cups; hoop stress 30 MPa (D16, S-11, S-12) |
| Lower ball-joint bolt bends: FoS 0.8 | `bolts.py`, lug-pin model, LC4 | COM-6 + 3/8 in NAS6606: FoS 1.83 in bending, 1.55 with the preload tension (D15, S-13, RC-27) |
| FEA loads out of balance by up to 344 N | in-run check on the applied resultant | Tapered-roller rows modelled as a traction tilted onto the pressure centre; every cosine patch turned until its resultant lies on its load (0.02 N left) |
| Upright v1 (layout sizes) peaks at over 600 MPa (lower web root) | first coarse FEA | v2: post 12 -> 20 mm, web 10 -> 20 mm, both wider |
| Housing ring bends at its free inboard end: 383 MPa at the bore edge (LC4), 542 MPa principal at the lower web's corner (LC2) | coarse FEA hot spots | Wall 7 -> 9 mm (the inboard row's pressure centre sits inboard of every arm) |
| Caliper lugs twist under braking: 471 MPa at a root (LC4) | coarse FEA; the pads sit 26 mm outboard of the bolts, so one bolt pushes and the other pulls | One plate (bridge + fill between the lugs), 14 mm, then flush with the housing's inboard end (19.4 mm): about 160 MPa on the coarse mesh |
| Peaks at the top of 5 mm root fillets | coarse FEA | Root fillets 8 mm (6 mm on the post, the largest OCC accepts there) |
| A stress that grew 124 -> 222 MPa as the mesh was refined | base-mesh run, LC1 | A singularity: the lower clevis's upper ear grazed the housing by about 1 mm after the wall grew. 8 mm boss on the ear over the housing's length only (a full-length boss came within 0.5 mm of the pushrod's rod end: sweep) |
| Ear roots are sharp re-entrant corners | same | Ears end 1 mm inside the part they join (a tangent end defeated the fillets); concave junctions found by point classification and filleted where OCC accepts (2 mm on two ears); the other four are recorded as sharp edges, kept out of the FEA peaks and checked by hand (S-14) |
| Independent code review: FAIL | reviewer agent | Fixed: equilibrium now gates the run (force and moment residual, loaded-face areas); FEA fingerprint covers its code and inputs (matrix shows STALE otherwise); mass and sweep outputs carry the CAD fingerprint; Goodman amplitude from the tensor range; nominal hole bearing P/(d t); the cylinder seam no longer counts as a junction; folded-element straightening re-measures; validation sign |
| Beam theory does not hold anywhere on the upright (steering arm 20 x 32 mm: FEA vs the beam formula -14 / +43 %) | validation attempt | Method validated on a slender pin-loaded bar with the same mesher, pin load and stress recovery: 0.04 % (RC-26) |
| FEA run used 12 GB and thrashed | memory | scikit-fem's full basis (about 8 GB) replaced by chunked assembly and stress evaluation (identical to 1e-9); the benchmark solves with multigrid, not SuperLU |

Sweep after all of it: 0 collisions, 0 warnings over 279 poses. **FEA result:** worst 166 MPa in LC4 (yield FoS 2.35), converged to 0.9 % over three meshes, method within 0.04 % of beam theory, hole bearing FoS 13. **Open:** the fatigue screen (unlimited life at peak 1.6 g left-right cornering) gives FoS 1.15 (cornering) and 1.38 (braking) at the housing's inboard bore edge and the lower ear: action, Miner's rule over the duty spectrum (Week 3b). Upright with clevis and shims 0.86 kg (target 0.6: action, lightening pass); unsprung mass 10.25 kg. Matrix after the audit fixes below: 45 pass, 12 open, 2 deviations, 3 actions; 150 tests pass. (Numbers above are from the corrected run: cups loaded over their own width.)

### Wide audit at the end of Week 3a (2026-10-27)

Three independent reviewers: engineering assumptions, documents against the generated outputs, and fit against the goal. Every finding was re-checked before acting.

| Finding | Verdict | Action |
|---|---|---|
| The FEA loaded each cup seat over the bearing's 19 mm overall width; the 32008 X cup is 14.5 mm, against the rib, so the load sat 4.5 mm too far toward the housing's free end, where both governing results were | Confirmed (cup width C = 14.5 mm) | Cups loaded over their own width; tilt re-derived from the pressure centre (now about 15 deg, close to the bearing's 14.2 deg contact angle); FEA re-run (RC-27) |
| The lower ball-joint bolt cannot be fitted: no room for a nut between the upper ear and the housing end; fasteners are not in the sweep | Confirmed | Upper ear tapped for a steel insert, bolt from below (D15); fasteners join the sweep in Week 3b (S-18) |
| "Light preload set by the axle nut" cannot work against a solid spacer and rib | Confirmed | Preload set by a ground or shimmed cone spacer (D13, D16) |
| Bolt check ignored the preload tension (about +145 MPa at the AN torque on the lower bolt) | Confirmed | Torques specified, tension added: lower 1.55, upper 1.51 (S-13) |
| Fatigue: braking also fails the screen (1.38); press-fit mean stress, temperature and a true 1 g static state missing; "about 1e7 cycles" claim had no calculation | Confirmed | S-07a's action covers both pairs and the missing inputs; the claim removed |
| Yield FoS quoted without saying the four sharp roots are excluded; S-14 had no fatigue | Confirmed | Reported as "excluding the roots left sharp"; S-14 adds fatigue |
| RC-26 read as moving the goalposts | Accepted | The original plain-section check stays (S-06f) and fails as deviation DV-2; the bar benchmark is an added check |
| No hub-spindle, brake-heat or bolted-clevis checks | Confirmed | S-15, S-16, S-17 open for Week 3b |
| Document numbers out of step (mass 0.84, "eight" cases, 700 MPa with no source, -11 / +43 %, "1.7x stiffer", rim gap called the tightest moving gap, stale "Week 3") | Confirmed | Corrected |
| The upright build's own mass record used aluminium for the steel shims; LC4 labelled "20 %" beyond the ellipse | Confirmed | Fixed at the same rebuild |
| Week 3b should cut the tolerance Monte Carlo and the compliance study, and bring the drawing, the BOM and Navy's own hand calcs forward | Recorded for Navy | A plan decision, not made here |

## Week 3b: upright v3, its drawing and BOM, hand-calc packets (2026-10-27)

Same hardpoints (v1.7). Navy re-ordered Week 3b (RC-28): the upright's revision first, then the drawing and BOM, then her own hand calcs.

| Finding | Evidence | Fix |
|---|---|---|
| The upper clevis's ear roots and the lower web's sliver blocked fillets | OCC refused them (v2) | v3: upper-clevis ears filleted R3 once the ears (r 13) and their base (half-width 18) no longer ran flush; the lower web made flat-bottomed (its trimmed end left a sliver the kernel could not fillet past). Two roots stay sharp: one lower ball-joint ear and one steering-arm ear (S-14) |
| The fatigue screen (unlimited life at 1.6 g left-right) fails: 1.14 cornering, 1.36 braking | FEA v3, Goodman | Miner's rule over the duty spectrum (`upright_fatigue.py`, S-07c): factor 3.18 on every stress before the 3,000 km life is used up (4.28 at the hole edges). The screen stays as DV-3 |
| Upright machined parts 0.845 kg against 0.6 kg | mass budget | Accepted as DV-4: the tapered bearings (D13) set a 0.27 kg housing ring; M-01 still passes (10.2 kg). v3 took 0.024 kg out |
| The rod end named in the data ("Aurora AM-M8") does not exist | pricing the BOM | It is the AM-M8T (PTFE lined, the 33135 N rating the checks used): name corrected, no FoS changes |
| The chosen spring (25 N/mm) is softer than Öhlins sells for the TTX25 (from 150 lb/in, 26.3 N/mm) | pricing the BOM | Open: the spring list in `vehicle.json` should be the real catalogue steps; a rerun moves the ride frequency by under 2.5 % |
| The lower ball-joint insert (KNH624J, 12.7 mm long) does not fit its ear, which is 6 mm thick at the hole (the boss stops at the housing end, 1 mm from the hole) | sizing the hole for the HC-7 lug check | Open for the fastener sweep (S-18); the drawing carries a HOLD on H1 |
| The hand-calc ear model reads 50 % low against the FEA at the steering-arm root the FEA covers | HC-6 | The FEA's peak is in the housing wall where the arm joins; HC-7 scales the ear model by that ratio (x 1.98) before judging S-14 |
| Independent code review of the hand calcs and BOM: FAIL | reviewer agent | Fixed: the ear model calibrated; fatigue cycles keep the sign of a reversing load; a missing sharp-edge list raises instead of passing; the results carry an input fingerprint (the matrix shows STALE otherwise); duty states are checked against the spectrum; the bending Kt table (one exponent had come from the torsion table) with the largest fit taken; the rocker's out-of-plane bending at the net section and its bearing on the pin loads; the ARB tube end on the welded allowable; tube lengths per rod end; the right caliper as its own line |
| Hub spindle passes by 1 % over the target with the 1 mm fillet the cone chamfer allows (k 1.52) | HC-4 | A shoulder ring behind the outboard cone so the shaft gets an R3 fillet: k 1.94 (D19) |

**FEA v3:** worst 169 MPa in LC4 (yield FoS 2.31), converged (166, 166, 169 MPa over three meshes: 2.2 %), method within 0.04 % of beam theory, hole bearing FoS 13. **Drawing:** FS1-UPR-001 rev A (D18). **BOM:** about 3,590 USD a corner estimated, 8,050 USD the front axle, 8 quote lines (D19). **Hand calcs:** seven packets and a separate answer key in `notes/handcalcs/`; rocker FoS 5.9 (S-06e), ear roots 1.80 (S-14), spindle 1.94 with the ring (S-15), knee 1.80 with a 3 mm inside radius, ARB ends a 2 mm weld and a stub of at least 17.4 mm.

## Upright v4: the fastener sweep (S-18) (2026-11-03)

Same hardpoints (v1.7). Every bolt between designed parts was put into the sweep (`scripts/fasteners.py`): washers, heads, nuts and thread inserts as bodies over all 279 poses, a straight socket on each joint's turned end, a ring spanner on its held end, and each bolt's insertion path, with the wheel off.

| Finding | Evidence | Fix |
|---|---|---|
| The lower ball joint's KNH624J insert (12.7 mm long) does not fit: the upper ear is 6 mm thick at the hole | first sweep (insert outside the material) | A 24 x 8 mm boss round the bolt axis on the upper ear (ear + boss 14 mm); the bolt axis is the steering axis, so the boss turns about itself as the wheel steers |
| That boss came within 2.9 mm of the pushrod's lower rod end at full droop and full lock | interference sweep (V-07 warning) | A shorter insert instead of a bigger boss: a 1D Heli-Coil (1191-6CN375, 9.5 mm) in an 18 x 4 mm boss (ear + boss 10 mm); pull-out FoS about 6 on the bolt's preload (HC-7) |
| A nut above the lower ball joint hits the housing | first sweep | No nut: the bolt threads into the insert (as specified), turned from below |
| The bolted upper clevis's two M8 heads sit inside its own ears and against the upper ball-joint node; no hex-key path | first sweep (24 FAIL) | Bolts moved above and below the ears (M6 at 28 mm): the top head still sat on the ear's R3 root fillet, the keys hit the ears and the upper joint's bolt was in the way; clearing all of it needs about 33 mm, 5 mm from the end of a base that cannot grow (the rim). The clevis is now one piece with the upright, camber set at the upper wishbone's chassis rod ends (D20) |
| As a friction joint the bolted clevis would have slipped under braking (slip FoS 1.04 at 12 N m) | HC-8 (S-17) | Moot with the one-piece clevis; S-17 retired (RC-30). A shear dowel had fixed the slip (FoS 10.7) but not the access |
| The upper ball joint's nut cannot take a straight socket (the hub is below it); the pushrod's lower nut neither (the tie rod) | first sweep | Each joint is turned at the end a socket reaches and held at the other with a ring spanner: the upper joint and the pushrod's lower joint at their heads |
| The sweep's own bookkeeping could hide a fault: both washers of a joint had one name, so one result overwrote the other; a tool wholly inside a closed body read as clear; the insertion path was only the shank's width; the result was stamped with the CAD's fingerprint only, so a changed washer or tool size left it looking current; the drawing's HOLD lifted unless a row named the lower ball joint as FAIL | independent code review | Names by end; containment counts as blocked; the path is as wide as the head or washer; a fastener fingerprint (CAD, hardware and tool sizes, poses, sweep code) that the matrix and the drawing check; the HOLD lifts only on a current file with no FAIL and both inserts in material |
| The left upright's Fusion mesh is two closed shells sharing an edge, so point-in-solid tests on it are unreliable: the first insert result could not be trusted | a watertight guard added after the review | Inserts are checked on the upright's STEP, classified by OpenCASCADE on the exact solid, after checking its volume against the mesh (0.1 %) |
| The CAD capture job wrote no images while another session's Fusion held a client document, and reported success | `saveAsImageFile` returned True without a file | The job now activates its own document, raises if no file appears and hands the previous document back its focus; the renders stay at the 10-27 (v3) set until Fusion is free |

**Sweep after the fixes:** 0 FAIL, 0 WARN over 308 checks (279 poses); both inserts 144 of 144 wall points in solid material; interference 0 collisions, 0 clearance warnings. **Upright v4:** one piece, 0.778 kg (v3: 0.845 kg as three parts); the post's root takes an 8 mm fillet (v3: 6 mm). **FEA (nine cases):** worst 170 MPa in LC4 (v3: 169), yield FoS 2.30, converged 2.0 % (166 / 167 / 170 MPa); fatigue screen 1.12 (DV-3), Miner 3.24 (4.35 at the hole edges); the two sharp ear roots by hand 1.78 (HC-7, calibrated to this FEA).

## Main spring on the real catalogue (2026-11-04)

Same hardpoints (v1.7). `vehicle.json` now holds the spring rates actually sold for the TTX25 MkII (Ohlins' list, 150-650 lb/in; Hyperco's FSAE line, 150-400 lb/in) in place of an assumed 2.5 N/mm metric list (D21).

| Finding | Evidence | Fix |
|---|---|---|
| The 25 N/mm spring is softer than anything sold for this damper | pricing the BOM (10-27); the retailer and Hyperco listings (read 11-04) | 150 lb/in (26.3 N/mm), a Hyperco FSAE spring, 55 USD: 2.91 Hz (V-01, 2.66-2.94), roll gradient 0.99 deg/g (V-02), ARB baseline 155 N m/deg and bar FoS 1.73 (S-09, was 1.58) |
| The stiffer spring sits 1.3 mm less compressed at static, so it goes slack over the last 2.3 mm of droop (was 1.1 mm) | vehicle calcs (V-05) | Still the helper-spring action; the droop stop stays the damper top-out (27.0 mm), because cutting droop to 24.7 mm would miss the 25 mm target |
| The load cases move with the wheel rate (each case's heave and roll pose) | vehicle calcs: LC1's bump-stop share 656 N | Loads, bearings, bolts, FEA and fatigue re-run on the new loads: no change at the reported precision (worst 170 MPa in LC4, yield FoS 2.30, converged 2.0 %; Miner 3.24, 4.36 at the hole edges) |

## Step B: brake heat, welds and threads, free play, FEA benchmarks (2026-11-04)

Same hardpoints (v1.7). Hand calcs HC-8 to HC-10 (`scripts/week3b_checks.py`, inputs in `data/checks_week3b.json`) and two FEA benchmarks (`scripts/fea/benchmark_peaks.py`). Decision D22, change record RC-31.

| Finding | Evidence | Fix |
|---|---|---|
| The disc's solid hat ring (0.27 K/W) carries about the disc's whole mean power into the hub and bearings; the upright can shed only about 110 W at 120 C, so the mount needs at least 4.1 K/W | HC-8 (assumed convection and contact values) | S-16 is an action: a thermal-break disc mount in step C (a plain floating disc on six bobbins gives about 1.8 K/W, not enough alone) and a housing temperature test in step D |
| The cups' press fit opens at only +16 K on the loosest cup (13 µm interference, 0.81 µm/K) | HC-8, from `bearings.json` | Above that the cups are held by the Loctite 641 (150 C); its limit, not the fit, bounds the housing |
| Nylon-insert nuts on the rod-end bolts, and no nut at all for the upper ball-joint bolt in the BOM | the 80 C nylon-insert limit (T.8.2.3, second-hand) against HC-8 | All-metal lock nuts on every rod-end bolt (ISO 7042) and an MS21042L5 on the AN5 bolt |
| `vehicle.json` takes 75 mm of rack per steering-wheel turn; the Kaz rack in the BOM is 119.6 mm | pricing data against the steering data | Kept: 75 mm gives the larger LC6 rack force and the larger free play, so both checks stay conservative |
| The member-end welds and rod-end threads pass in fatigue: Miner 3.2 (weld root, IIW FAT 36) and 3.0 (EN 1993-1-9 detail 50), stripping FoS 6.9 | HC-9, member forces in every duty state | Drawing rules: weld throat at least the tube wall; thread engagement 1.5 d with a jam nut. The IIW classes were read second-hand |
| The first hole benchmark asked for 0.35 mm elements and got 1.2 mm: the mesher holds every element at no less than a fifth of the global size | reading `fea._gmsh_mesh` before trusting the result | Quarter model with a 1.75 mm global size; the full plate then ran out of memory on the shared machine, so it is 200 mm long (L/W 2) |
| The FEA method recovers a stress concentration and the cup seat | `notes/fea_benchmarks.md` | Hole Kt +0.51 % (Pilkey's fit to Howland), ring hoop stress +0.03 % and growth -0.00 % (Lame); both gated on their reactions |

## Step C: the floating disc and the chassis interface (2026-11-04, built 2026-11-08)

Same hardpoints (v1.7). Decisions D23 and D24, change record RC-32. Checked offline: the new parts rebuilt with trimesh from the same definition, against the existing meshes over all 279 poses (only the disc had changed in them).

| Finding | Evidence | Fix |
|---|---|---|
| The heat budget counted the upright only; the hub flange sheds some too | HC-8 revisited | Both counted (one temperature): the mount needs 3.0 K/W, not 4.1 |
| Stainless floating-disc bobbins fail at the high end of the contact range (1.9 K/W) | HC-8 | Titanium bobbins with a 6 mm path: 3.6 K/W at the high end, margin 1.20 (D23) |
| A carrier at 46-54 mm came within 1.5 mm of the upright's outboard end; the bobbins' sweep within 2.6 mm | offline clearance check | Carrier 50-54 mm (3.8 mm), bobbins at 56 mm with a 0.5 mm inboard head (3.4 mm) |
| Lower wishbone tabs, damper tabs and the rack web bent to FoS 0.8, 0.8 and 1.3: the joint forces run across them | HC-11 | Widened (40 mm, stepped 16 to 36 mm, 30 mm): FoS 2.16 at least |
| Damper tabs touched the damper body (it starts 9 mm from the eye) | offline pre-check | 8 mm round the bolt, leaning 35 mm away: 1 mm clear (on the assumed eye geometry) |
| The Kaz rack's clevis (1/2 in slot) leaves no room for the rod end's misalignment spacers; its top hats do not fit an 8 mm bore | pre-check, Kaz's page | A custom 4130 clevis with a 22 mm slot on the 1/4-28 extension; its reach cut to 4 mm to clear the housing at full lock (5.6 mm). Its web needs the rack's extensions about 5 mm longer than assumed |
| The ghost chassis has no tube within 127 mm of the dampers' chassis eyes, and its top rail lies in the rocker's swing | nearest-tube search; code review | Two assumed tubes, recorded as chassis interface requirements: a cross tube under the eyes, a rail below each rocker |
| A rocker bracket centred on the axis midpoint, not on RKR_P (where the rocker is) | caught while building the bracket | Centred on RKR_P |
| The rocker bracket to the top rail ran through the drop-link bolt's swing; the rack bracket's bolt washers sat on its web | independent code review | Bracket down to the assumed rail with 12 mm spacers; bolts 30 mm apart |
| Bobbins tangent to the carrier's bore and a coincident carrier and hat would break the CAD union | independent code review | 11 mm bobbins; carrier and hat as one tube |
| A first double-shear check passed any through-bolt (two opposite seat faces), so it could not show single shear | own check of the check | Kept as a record of the clevis joints; R-08 stays open until the drop link's lower end is detailed |
| First build (11-08): 10 collisions, all between fixed parts the offline pre-check had skipped: front wishbone tabs inside the pedal-box keep-out, the rack web in the rack housing, the two damper tabs meeting at the centreline | interference sweep | Tab ends on the tube's surface; the web below its pad; the damper tabs lean 25 mm, not 35. The pre-check now covers fixed parts and left against right |
| The pedal-box keep-out (+/-170 mm) overlapped the side rails themselves (inner faces at 162 mm): a contradiction in the assumed context | the same | Keep-out +/-150 mm, recorded in cockpit_envelope.json |
| 10 tool-access fails: the upper wishbone bolts tilt towards their rail (normal to the leg), so a head, nut or socket on the rail side sat 0.4-1.5 mm from it; the spanner on the damper eye's bolt hit the damper body | fastener sweep | Each wishbone bolt turned from the side away from its rail and held by a ring spanner on the rail side; the assumed upper rail 10 mm inboard; 16 mm spacers either side of the damper eye (10 mm did not help: the body starts 9 mm from the eye along its axis, so the stack had to move outside the body's 40 mm width) |
| Final build (11-08): 0 collisions, 0 clearance warnings; fastener sweep 0 FAIL, 0 WARN over 612 checks | both sweeps | Matrix 57 pass, 4 deviations, 1 action, 3 open (R-08's drop-link joint, R-09, R-10); renders refreshed |

## Step C close-out: the drop link's lever end, locking and bolt lengths (2026-11-08)

Same hardpoints (v1.7). Decision D25, change record RC-33. Pre-checked offline (the existing meshes with the lever moved and the rod end turned, through the real interference sweep: 0 collisions, 0 warnings), then rebuilt (inputs 303cbc9f0874c2ea).

| Finding | Evidence | Fix |
|---|---|---|
| The drop link's lever-end bolt lay in the blade's plane (the CAD and the misalignment axis were normal to the link and the rocker's axis): not buildable | planning R-08 | Bolt along the bar's axis through the blade; the lever 14 mm inboard; the rod end on a spacer beside it, a 24 mm washer outside (captured single shear); misalignment 2.1 deg of 18 |
| A blade ARB adjusted by rotating the blade cannot carry a through-bolted rod end | the same | For step D: holes along a stiff lever, the bar sized for the rate |
| The CAD lever (6 x 18 mm) was never sized: FoS 0.95 at the warp torque (bending plus the twist from the 14 mm offset) | HC-12 | S-21 is an action for step D; the bolt passes (S-20, FoS 1.99) |
| The caliper bolts in the BOM (M8x25) engaged the caliper 5.6 mm: the lug grew to 19.4 mm during the FEA iterations | the bolt schedule | M8x30 through all of the ear's 10.4 mm thread; the tip 0.2 mm past it, 8.5 mm from the disc |
| Wilwood lists only a 0.32 in mount hole for the PS-1; whether the ears are tapped is not published | Wilwood's page (read 11-08) | Taken as tapped through; an interface item (WARN rows) |
| Threads in aluminium need about 2 d; the Kaz trunnion block's thread depth is unpublished (the modelled collar is an envelope with about 10 mm of wall over the bolts) | own check; code review | 2 d scheduled; an interface item on the block (WARN rows) |
| Tapped bolts had no tip in the sweep; insertion paths used grip + 15 mm, not the real bolt; a list filter passed with a named check missing; any 'capture' passed R-08 | independent code review | Tip bodies past through-tapped parts; scheduled lengths; every named check must have rows; the capture washer must be larger than the ball |
| Final build (11-08): 0 collisions, 0 clearance warnings; fastener sweep 0 FAIL, 8 WARN (the interface rows) over 712 checks | both sweeps | Matrix 61 pass, 4 deviation, 2 action, 1 retired; renders refreshed |

## Step D: the ARB's rate in the bar, DFMEA and test plan (2026-11-18)

Same hardpoints (v1.7). Decision D26, change record RC-34. Rebuilt (inputs b7e2ec4e0af11953).

| Finding | Evidence | Fix |
|---|---|---|
| A blade ARB cannot work with the drop link bolted through the lever: rotating the blade turns the bolt past the rod end's 18 deg; and a blade soft enough to set the rate would see about 1500 MPa at the warp torque | D25's finding; hand sizing | The bar sets the rate: three solid 4340 bars (10.2 / 12.4 / 13.9 mm), swapped to adjust, between stiff 8 x 25 mm levers |
| Lever holes as the adjustment instead | geometry | Rejected: a hole 23 mm along the lever tilts the drop link 18 deg at the rocker's rod end |
| The bar's modelled length (hub to hub, 356 mm) was 26 mm longer than its CAD body: every bar about 6 % stiffer than modelled, more than S-22's 3 % | independent code review | The model counts the 330 mm body and each 13 mm upset to its hub's centre; the bars re-sized |
| No check of the shoulder where the body steps up to the splined end, and none in fatigue on a part that reverses every corner | independent code review | An R4 shoulder; HC-5 adds an infinite-life check at the largest cornering roll: FoS 1.51 (S-23; the cut spline governs, the ground shoulder 1.67) |
| The DFMEA's coverage check passed on a stale matrix and on any status starting 'PASS'; a warning needed no closing entry; ratings were off their own scales; ten failure modes were missing (spindle, links, ears and tabs, ARB lever, ARB set-up, joint bind, steering abuse, bearings, disc and bobbins, wheel studs) | independent code review | The check fails closed on every one of these; the ratings follow the scales; the rows added; numbers are placeholders filled from the generated results |
| Final build (11-18): 0 collisions, 0 clearance warnings; fastener sweep 0 FAIL, 8 WARN (the interface rows) over 712 checks | both sweeps | Matrix 64 pass, 4 deviation, 1 action, 1 retired; DFMEA 21 rows, 8 tests; renders refreshed |

## Step E: the write-up and the final wide audit (2026-11-21)

Same hardpoints (v1.7). Decision D27, change record RC-35. Rebuilt (inputs 237fb82d0cc0ae23). Four independent audit agents (numbers and claims; engineering soundness; honesty and presentation; the pipeline's own claims) checked the first write-up against the files. Every finding was re-checked against the files before anything changed.

| Finding | Evidence | Fix |
|---|---|---|
| The bar's fatigue check left out the 99 % reliability factor the upright's check uses; with it the 150 ksi bar's machined spline failed (1.23) | engineering audit; `scripts/fea/upright_fatigue.py` | 4340 at 180 ksi, 25 mm upset ends, ground body and fillets: fatigue 1.58, static 2.11 (S-23, S-09) |
| Toe could be set only in half-turn steps of 0.6 deg (two right-hand rod ends, an eye that must stay aligned) | engineering audit | A left-hand rod end and a left-hand jam nut at each tie rod's outer end: the tube is a turnbuckle |
| DV-1's cause blamed the pickups' width, and 'the fix belongs to the chassis' did not follow: no layout or width tried reaches the target | `data/tradeoff_rc_camber.csv` | DV-1 restated; the balance shift labelled a front-only sensitivity |
| T-06 measured the balance with damper potentiometers, which see only the elastic load transfer; T-04's fixture did not match the FEA's supports; the 500 km interval had no basis | engineering audit | A tilt table; a fixture that loads the part as the car does, with the FEA to be re-run on the fixture's supports for the strain predictions; the interval called a judgement |
| Fretting at the ear holes and under the spindle's cones, and the brake's line pressure and temperatures, were not in the DFMEA | engineering audit | F-22, F-12 (O 6), F-23 (98 bar at LC4) |
| 'Mesh-converged to 2 %' overstated: the peak still creeps up between the finest meshes (166, 167, 170 MPa) | `notes/upright_fea.md` | Worded as the change between the two finest meshes, not asymptotic |
| 'The tightest gap' was only the rim's; 'every moving pair at least 5 mm' ignored the 3 mm limit inside the wheel; the rim check's script defaults still held the old 7 in rim and printed a negative gap | `data/interference.csv`, `scripts/rim_clearance.py` | Worded as the rim's gap; both limits and minima quoted; the script reads the data and writes `data/rim_clearance.json` (6.4 mm analytic, 6.3 mm in the sweep) |
| Hand-calc methods described as independent of the FEA, but HC-6 and HC-7 are calibrated to it; 'hand method' read as worked by hand | `notes/handcalcs/answer_key.md` | Said so; nothing is claimed as checked by hand until Navy's paper packets are done |
| The FEA report gave a 3 mm radius for the two sharp roots; the drawing and S-14 say R2 | `notes/upright_fea.md` | Report text corrected; the FEA re-run from its checkpoints (a report-only change) |
| The pipeline claims were stronger than the code: the write-up could print a stale number, the FEA's area check runs on the first case only, the upright's STEP is outside the CAD fingerprint, 'run_all rebuilds every output' | pipeline audit | `writeup.py` now refuses open, failed or stale items and stale inputs, and any leftover placeholder; the area check made to match its claim (next row); the other claims reworded to what the code does |
| Checking that claim showed the gap was real: the rib face the thrust bears on flips between cases (LC1 loads one face; LC2, LC4 and LC7 the other), so one loaded face's area was never checked | a check on the coarsest mesh: both rib faces 612.6 mm2 against 612.6 | Both rib faces checked; every face any case can load is area-checked on the base mesh in a stage of its own, so the check is part of every result |
| The FEA report printed a Richardson extrapolation from meshes whose change grows with refinement (166, 167, 170 MPa: observed order -4.4, a negative 'GCI'); its Goodman screen (1.12) had no pointer to DV-3 | found in the final run | `richardson()` gives no estimate unless the changes shrink (a test added), and the report says why; the screen's note points to DV-3 and Miner |
| An independent code review of the step E code FAILED it: the write-up trusted the matrix and the DFMEA page by their hardpoint label (both were stale after the bar change); toe per turn used the steering arm's distance from the kingpin axis instead of the tie rod's moment arm (6 % low); placeholder, link and number-format gaps; the rim check's sampling could overstate the gap; the bar torques used the target rate, not the stiff bar's as built | code review, then a re-review (PASS); every finding re-checked against the code and, for toe and camber, the kinematic solver | Both pages rebuilt and compared before the write-up is written; the setup table now from the kinematic solver, with the lower ball joint held as the untouched spring perch holds it (camber 0.55 deg per turn with the toe reset, matching the three-sphere closed form; each turn also moves the corner 0.3 mm), toe 2.4 deg per turn against its closed form; stray braces, exact-name links, whole-number counts and exponents caught; the rim's closest point refined; the bar's as-built rate; a left-hand jam nut in the BOM, and the costed BOM stamped with its source's fingerprint, which the matrix checks |
| Final (11-21): 0 collisions; fastener sweep 0 FAIL, 8 WARN (interface) over 712 checks | both sweeps | Matrix 64 pass, 4 deviations, 1 action, 1 retired; DFMEA 23 rows, 8 tests; `WRITEUP.md` 3410 words; renders refreshed |

## Step E follow-ups: the pipeline's freshness gaps (2026-11-22)

Same hardpoints (v1.7). No design change. No FEA or fatigue input touched (their fingerprints are still 9bc2359936b9cdab and f5a9eafc901c633e), and no Fusion rebuild (the CAD fingerprint is still 237fb82d0cc0ae23). The step E pipeline audit's three open follow-ups, closed:

| Finding | Evidence | Fix |
|---|---|---|
| The upright STEP was fingerprinted by the FEA but not tied to the CAD. A STEP regenerated without a CAD rebuild passed the sweeps, and so did a CAD rebuilt on a STEP older than its inputs | `fingerprint.CODE_FILES` names `upright_geom.py`, never the STEP it writes; only the fastener sweep compared the two volumes (0.5 %) | Two links. The mesh index records the STEP the meshes were exported with (`fingerprint.upright_step_sha`, written by `job_export_meshes.py`), and `common.load_mesh_index` refuses meshes exported with another STEP. The STEP's own stamp says today's inputs still reproduce it: `run_all.py` first regenerates the upright in a scratch folder holding only its declared inputs, and stamps the committed files only on an exact match (`stamps.verify_upright`; about 40 s, only when something changed). Adding the STEP to `cad_inputs_hash`, or editing `upright_geom.py` or `job_build.py` to record it, was not done: each is a CAD input, so each would have made every committed mesh stale until Fusion rebuilt them |
| Recording the STEP hash in today's index without a re-export needed proof the meshes came from today's STEP | Regenerated from today's inputs in a scratch copy: the three STEPs' DATA sections and `upright_build.json` are identical. The upright meshes lie within 0.08 mm of the STEP solids, and the solids within 0.14 mm of the meshes (both sides; the same mesh moved 0.5 mm reads 0.54 mm). The STEP was written on 11-03 and the meshes on 11-21 | Recorded. `test_the_committed_upright_meshes_are_the_step_solids` repeats the geometric check on every test run |
| One swapped or edited mesh was accepted: the index named the files but not their content | `cad/meshes/index.json` | Each mesh's hash is in the index: 108 added from the committed files, after checking each file's triangle count and size against the index. `common.load_mesh_index` checks every file, and `common.load_mesh` checks the bytes it parses. The interference sweep, the fastener sweep and the mass budget read meshes only through them |
| Most outputs carried only the hardpoint label, so an edited `members.csv`, a changed load case or new code left them looking current | `requirements_matrix.load_sources` | `data/stamps.json` holds, per producer, the content hash of each input it read (its declared data files and the code it imports) and of each output it wrote. The matrix refuses any mismatch, following generated inputs back to the script that made them. The stamps live in a sidecar rather than in each file, because four of these outputs are FEA or fatigue inputs whose bytes are fingerprinted; an in-file stamp would have marked the FEA stale. A run that reads a data file it does not declare fails (an audit hook), so the declarations cannot fall behind the code |
| `mass_budget.py` read `vehicle_derived.json` and never used it | the read trace of every stage | Removed; it would have been a false input |
| An independent code review of the new checks FAILED them. `code_files` missed `from package import module` and relative imports. The scratch regeneration inherited `PYTHONPATH`, which can import from the real tree. A move into place was not seen as a write. An unindexed mesh or a `.tmp` cache could be read unhashed. Nested runs were tracked by equality. The stamp write had no retry under OneDrive. A missing input compared as current | code review; a re-review FAILED the fixes once more, on three points: an unreadable STEP compared equal to itself, a failed trace could still stamp, and the FEA and hand-calc fingerprints cut a STEP at any `DATA;`. A third review PASSED; its three low notes were fixed as well | All fixed, with a test each. The STEP cut now uses `fingerprint.step_data` everywhere, and the FEA, fatigue and hand-calc fingerprint values are unchanged. A QA agent's 59 black-box tests of `stamps.py` all pass. A new test imports each producer in a fresh module state and checks that every project file Python loads is in its stamp |
| Left open: the export records the STEP on disk, not the one the build imported. The FEA and fatigue fingerprints omit `scripts/common.py`, which both import. Reads outside Python's open (gmsh reading a STEP in `bom.py` and `fasteners.py`) are declared by hand. The generated `notes/` pages carry no stamp | the code | Recording the imported STEP means editing `job_build.py`, a CAD input: do it at the next rebuild. Adding `common.py` to the FEA fingerprint would mark the FEA and fatigue stale (about 1.5 h): do it at the next FEA run |
| Final (11-22): `run_all.py` (61 min on a loaded machine): upright STEP stamp current; 0 collisions; fastener sweep 0 FAIL, 8 WARN over 712 checks; all 15 producers stamped | `run_all.py`, pytest | Matrix unchanged: 64 pass, 4 deviations, 1 action, 1 retired. pytest 241 passed (211 + 30 in `tests/test_freshness.py`) |

## Publishing: machine-independent FEA and fatigue fingerprints (2026-11-23)

Same hardpoints (v1.7). No design change, no FEA or fatigue input changed, no Fusion rebuild (the CAD fingerprint is still 237fb82d0cc0ae23).

| Finding | Evidence | Fix |
|---|---|---|
| The FEA and fatigue fingerprints hashed four files that `run_all.py` rewrites, byte for byte, and their last digits depend on the machine: least-squares balance residuals near 1e-15 (`upright_loads.json`), zeros written as -0.0, whose sign can follow rounding noise (25 in `upright_loads.json`, 4 in `duty_loads.json`) and bearing loads at full precision (`bearings.json`). The CI runner (Linux), or a numpy upgrade here, would have marked the FEA and the fatigue check STALE with nothing changed, and the write-up refuses a STALE requirement | `fingerprint.fea_inputs_hash` and `fatigue_inputs_hash`; found by reading what each value is made from before the first CI run (no Linux machine here to compare on) | `fingerprint.GENERATED_INPUTS` (`upright_loads.json`, `loadcases.csv`, `duty_loads.json`, `bearings.json`) are hashed in a canonical form: non-integer numbers to 7 significant digits, -0.0 as 0.0, and the interface loads' balance residuals left out (a check the FEA does not read; `upright_loads.py` stops when one exceeds 1e-3). A change of one part in 10^6 or more always changes the fingerprint (`tests/test_week3.py`) |
| An independent code review and a QA run (100 edge cases) of the first version, at 9 digits | The review put the chance that a full-precision bearing load straddles a 9-digit rounding boundary between machines at up to about 1 in 100 per regeneration. `float()` read label-like CSV cells (`007`, `Infinity`, long numeric codes) as numbers. A malformed CSV raised `csv.Error`, which the matrix does not catch. A key named `residual` was dropped from all four files | 7 digits; only plain decimals are numbers (integers kept exact); `csv.Error` becomes `ValueError`, so the matrix reads STALE; residuals are dropped from the interface loads' cases only. The review's one major finding, a stale hand-calc record, was checked and did not hold: the hand calcs had been re-run straight after the change |
| The recorded fingerprints had to follow the new method without re-running an hour and a half of FEA and fatigue | Before the switch, the old method still gave the recorded `9bc2359936b9cdab` (FEA) and `f5a9eafc901c633e` (fatigue) from today's files, and each version of the new method its own recorded values before the next | The same files under the final method: FEA `eeca6594bbacfb8f`, fatigue `f22cc6bed299c7c4`, written into `upright_fea.json`, `fea_lc4_plot.json` and `upright_fatigue.json`. The hand calcs (whose fingerprint hashes `upright_fea.json`), the matrix, the DFMEA and the write-up were re-run, with the same results. The FEA's stage checkpoints in `.tmp/fea` are keyed on the old method, so a re-run of the FEA starts from scratch |
