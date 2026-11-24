# FSAE-class front suspension: design and verification

*Navy Robson, Mechanical Designer (engineering student). Independent design from public data only: no team data, hardpoints or CAD.*

*How it was made: I set the brief and the four main design choices and reviewed each checkpoint. The analysis code, the Fusion 360 build scripts and the detailed technical decisions were produced with an AI coding assistant (Claude); every decision is recorded with its reason in [notes/requirements.md](notes/requirements.md).*

![The upright, outboard and inboard](plots/hero_upright.png)

*The upright from outboard (left) and inboard (right): one machined 7075-T7351 part, {req:M-02} kg, shaped by the FEA. Rendered from the same STEP the FEA meshes and the drawing is made from.*

## The brief

A front suspension for an FSAE-class car, from requirements to drawings, where every number can be traced to the file that produced it. The scope is the whole system past the chassis, both front corners:

- wishbones with spherical rod ends;
- a pushrod, a rocker and an inboard coilover (Öhlins TTX25);
- an anti-roll bar;
- the steering linkage to a Kaz rack;
- the upright, the hub and its bearings, the brake (a Wilwood PS-1 caliper on a floating disc);
- the tabs and brackets that join all of this to an assumed chassis.

The chassis itself is context only.

The car is my own reference vehicle, inside published FSAE ranges:

- {vin:wheelbase_mm:.0f} mm wheelbase, {vin:track_front_mm:.0f} mm front track;
- {vin:mass_total_kg:.0f} kg with a {vin:driver_mass_kg:.0f} kg driver, {vin:front_weight_fraction:pct0} on the front, CG at {vin:cg_height_mm:.0f} mm;
- 10 in wheels on 16 x 7.5-10 tyres, no aero.

The targets came from public practice. The main ones:

- camber recovery in roll {tgt:K-02};
- bump steer {tgt:K-09};
- a roll gradient of {tgt:V-02};
- {vin:tlltd_front_target_assumed:pct0} of the total lateral load transfer at the front (TLLTD, which sets the handling balance);
- a front ride frequency of {tgt:V-01}.

That last one is lowered from 3.2 Hz by a recorded requirement change (RC-2), because the roll and ride targets conflicted.

![The assembly](plots/cad/r01_assembly_iso.png)

*The CAD is generated from one hardpoint file and posed by the kinematic solver. The grey frame is context, not designed.*

## Architecture, and why

- **Pushrod to a rocker and an inboard coilover.** This takes the spring and damper off the wheel (less unsprung mass), lets the motion ratio be designed ({req:K-10a}-{req:K-10b} over ±25 mm, rising {req:K-10c} %), and gives the anti-roll bar a clean place to connect.
- **10 in wheels.** They are lighter, but they make the upright, bearings and brake hard to package; that packaging is most of the work.
- **The upright is code.** One script builds the solid for both the Fusion assembly and the FEA, so the part analysed is the part drawn. There is one exception: two ear roots the CAD kernel could not fillet are sharp in the FEA and carry an R2 radius on the drawing, so they are checked separately.
- **Real parts throughout:** Öhlins TTX25 damper, Aurora rod ends and spherical bearings, SKF 32008 X tapered-roller hub bearings, and a Wilwood PS-1 caliper taken from its published drawing.
- **An anti-roll bar set by the bar.**
  - **The design:** three solid 4340 bars ({vin:arb_bar.bars_mm.soft:g} / {vin:arb_bar.bars_mm.middle:g} / {vin:arb_bar.bars_mm.stiff:g} mm), swapped to adjust, between stiff levers.
  - **Why not blades:** an earlier design adjusted the rate with rotating blades, which could not work. Rotating a blade would turn the bolted drop link past its rod end's rating. A blade soft enough to set the rate would also be overstressed in warp (one wheel at full bump, the other at full droop).

## Kinematics

| | Result | Target |
|---|---|---|
| Static camber / caster / kingpin inclination | {req:K-01} / {req:K-03} / {req:K-04} deg | {tgt:K-01} / {tgt:K-03} / {tgt:K-04} |
| Scrub radius / mechanical trail | {req:K-05} / {req:K-06} mm | {tgt:K-05} / {tgt:K-06} |
| Camber recovery in roll (outside wheel, 0.5-2 deg) | {req:K-02}-{kin:camber_recovery_max:.1f} % | {tgt:K-02} |
| Bump steer over ±25 mm | {req:K-09} deg (effectively zero) | {tgt:K-09} |
| Ackermann, 5 deg to the stop | {req:K-11a}-{req:K-11b} % | {tgt:K-11a} |
| Front roll-centre height / its movement in roll | {req:K-07} / {req:K-08b} mm | {tgt:K-07} / {tgt:K-08b} |
| Roll-centre movement in heave above the ground, ±25 mm | **{req:K-08a} mm** | {tgt:K-08a}: **deviation DV-1** |
| Ride frequency / roll gradient at the middle bar | {req:V-01} Hz / {req:V-02} deg/g | {tgt:V-01} / {tgt:V-02} |

![Roll centre](plots/k03_roll_centre.png)

*Left: the front roll centre's height above the ground (judged, DV-1) and relative to the body, over the wheel travel. Right: in roll its height stays within {req:K-08b} mm; its sideways shift ({kin:rc_lateral_roll:.0f} mm over ±2 deg) is reported for information.*

**The one kinematic miss is roll-centre movement in heave.** It is judged above the ground, as the plan set it: {req:K-08a} mm over ±25 mm against {tgt:K-08a}. It stays a failed requirement (deviation DV-1), not a redefined one.

- **No layout tried reaches the target.** Across the trade study (run on hardpoints {trade:version}; v1.7 moved only the damper's chassis eye), the best migration stays between {trade:mig_min:.0f} and {trade:mig_max:.0f} mm at any camber recovery from {trade:rec_min:.0f} to {trade:rec_max:.0f} %, even with the upper pickups {trade:uca_y_max:.0f} mm out. This design sits at {req:K-08a:.0f} mm because its upper pickups are also held to the assumed chassis's width, with the rim's clearance at full lock. At that width no geometry tried does better, and each extra 10 mm of pickup width buys only about {trade:per_10mm:.0f} mm. The target, as set, cannot be met within these constraints.
- **Relative to the body it moves {kin:rc_arm_change_heave:.1f} mm.** That is the change in the CG-to-roll-centre arm, which is what drives the load transfer.
- **The effect is a sensitivity, not a prediction.** If only the front roll centre moved, with the rear assumed fixed, the front's share of the load transfer would range from {veh:tlltd_front_bump25:pct1} to {veh:tlltd_front_droop25:pct1} over ±25 mm. In braking pitch the rear moves the other way and adds to it. A tilt-table test at two ride heights is planned to measure it.

## Loads

- **Load cases** (`data/loadcases.csv`):
  - {lc:LC1};
  - {lc:LC2}, and the inside wheel;
  - {lc:LC3};
  - {lc:LC4};
  - {lc:LC5};
  - {lc:LC7};
  - {lc:LC8};
  - the steering effort with the wheels blocked (LC6).
- **What the cases leave out.** No case combines a bump with cornering, and no dynamic factor is applied. The margins cover that judgement, and the pushrod strain gauges in the test plan will check it.
- **Member forces.** A 3D six-link solver gives the forces in every member. A separate 2D closed-form method on the 3 g bump agrees with it: pushrod compression {hc:lc1_pushrod_3d_N:abs.0f} N against {hc:lc1_pushrod_hand_N:abs.0f} N.
- **Member checks.** Every link passes:
  - tube yield (FoS {req:S-01});
  - column buckling with the slimmer rod-end shanks included (FoS {req:S-02b});
  - the rod ends on their ultimate rating ({req:S-03});
  - the ball joints on their limit rating ({req:S-04}).

## The upright

The upright started as a layout part at over 600 MPa. The FEA shaped its next two versions and the fastener sweep the fourth.

- **Method.** The FEA is open source: gmsh and scikit-fem, quadratic tetrahedra, {fea:dofs_base:,} degrees of freedom on the base mesh. It loads the part the way it is really loaded:
  - pin-bearing pressure in every clevis and bolt hole;
  - tilted tapered-roller pressure on the cup seats;
  - the thrust on the central rib.
- **Fail-safes.**
  - Every solve stops unless the applied loads balance.
  - Every face a load case can load (each hole, both cup seats, both faces of the rib) has its area checked against the build record, within 5 %.
  - The mesh is refined where a coarse pre-pass found the stress.
- **Validation.**
  - The method matches beam theory on a slender pin-loaded bar to within {req:S-06c:.2f} %.
  - It recovers a hole's stress concentration to about {req:S-06g:.1f} % and a thick ring under bore pressure to within {req:S-06h:.2f} %.
- **Mesh refinement.** The governing peak changed {req:S-06d} % between the two finest meshes ({fea:convergence.0.peaks_MPa._global:.0f}, {fea:convergence.1.peaks_MPa._global:.0f}, {fea:convergence.2.peaks_MPa._global:.0f} MPa). It is still creeping up, not yet in the asymptotic range, so the peak carries that uncertainty; the FoS of {req:S-06a:.2f} (target {tgt:S-06a}) allows for it.

![LC4 von Mises](plots/fea02_lc4_von_mises.png)

*{fea:critical_case} governs: {lc:LC4}. Its body peak is {fea:cases.LC4.peaks_MPa._global:.0f} MPa on the base mesh, FoS {req:S-06a:.2f} on the design yield.*

*The plot is a re-solve on a mesh of the convergence study's level-1 size: peak {lc4:peak_MPa:.0f} MPa, against {lc4:stored_level_peak_MPa:.0f} MPa in the study. The grey zones are kept out of the peak and checked separately:*

- *pin contact at the holes, checked by bearing stress;*
- *the held points (the patch on the outboard face is one of them);*
- *two sharp corners the CAD kernel could not fillet, checked by a hand-calculation method calibrated to the FEA (HC-7: FoS {req:S-14}).*

What the FEA changed:

- the bearing housing's wall;
- one plate for the caliper mount instead of two lugs (the caliper's offset twisted separate lugs);
- the fillets;
- a boss where an ear grazed the housing. A stress that kept growing as the mesh got finer gave that one away.

Fatigue (7075 has no endurance limit):

- **Miner's rule:** over an assumed {fat:life_km:,.0f} km duty spectrum, the factor on every stress is {req:S-07c}. The S-N line is Shigley's estimate, with a machined-surface factor made for steels (commonly applied to 7075). Fretting at the bolted ear holes is not modelled; the DFMEA covers it.
- **Unlimited life:** the stricter screen, unlimited life at the peak cornering load, gives only {req:S-07a}. That is recorded as deviation DV-3. It is managed by a proof test and an inspection interval, and the interval stays a judgement until the measured load spectrum supports one.

Two more deviations:

- **DV-4, mass:** the part weighs {req:M-02} kg against its target ({tgt:M-02}), a target set before the bearings had to grow. The corner still meets its unsprung budget: {req:M-01} kg against {tgt:M-01}.
- **DV-2, no direct hand check:** the upright has no slender section, so the plan's hand-calc check of the FEA at a plain section cannot be made. The method's validation on the slender bar, the hole and the ring stands in for it now; a strain-gauged proof test in the test plan will check the part itself.

The part has a GD&T drawing generated from the same solid ([FS1-UPR-001](drawings/FS1-UPR-001.pdf)):

- datums on the bearing axis, the rib and the lower ball-joint hole;
- a hole table for the compound-angle holes.

## Fit, clearance and every bolt

- **CAD sweep.** The CAD is posed by the kinematic solver over {intf:poses} combinations of heave, roll, steer and warp, and every pair of parts is checked:
  - {req:V-06} collisions;
  - parts that move relative to each other stay apart by {tgt:V-07}: smallest {intf:moving_min:.1f} mm, and {intf:intra_min:.1f} mm inside the wheel.
- **Rim clearance.** The rim's tightest gap is its lip to the lower wishbone at full bump and full lock. The CAD sweep gives {intf:rim_gap:.1f} mm. An analytic check of the same pair (`scripts/rim_clearance.py`: the barrel as a cylinder of assumed inner radius, the leg as a tube, the closest point refined) gives {rim:min_mm:.1f} mm; the two models agree within half a millimetre. The worst-case tolerance stack-up of every source of error is {log:stack_worst_mm:.1f} mm (notes/requirements.md, section 6), so the parts stay apart even if every source goes the wrong way at once.
- **Fastener sweep.** Every washer, head, nut and thread insert goes through the same poses, with the socket, the spanner and the path to pull each bolt out: {fast:fails} failures over {fast:checks} checks.
- **What that sweep found:**
  - a thread insert that could not fit its ear;
  - a bolted clevis no key could reach;
  - wishbone bolts no socket could reach beside their rail;
  - a caliper bolt that engaged its thread by only 5.6 mm after the caliper mount grew during the FEA work.
- **The rules' joint requirements.** Every joint is locked positively (T.8.3), every bolt length comes from a schedule that leaves two threads past its nut (T.8.2.5), and every rod end is in double shear or captured (V.3.1.5). The rule numbers are checked against the official 2026 rules (version 1.0).
- **Open warnings.** The {fast:warns} remaining warnings are threads inside bought parts (the caliper's mount holes, the rack's mounting blocks) whose depths the makers do not publish.

![Full lock at full bump](plots/cad/r05_full_lock_bump_top.png)

*Top view at full lock and full bump: the pose where the rim comes closest to the lower wishbone.*

## Setup and adjustment

| Setting | Baseline | How it is adjusted |
|---|---|---|
| Camber | {req:K-01} deg | Upper wishbone's chassis rod ends: about {calc:camber_per_turn:.2f} deg per turn of both, with the toe reset afterwards (each turn also moves toe about {calc:camber_toe_per_turn:.1f} deg, and the corner's height about {calc:camber_height_per_turn:.1f} mm) |
| Toe | set on the alignment rig | Turn the tie-rod tube: it has a left-hand rod end at one end, so it works as a turnbuckle. About {calc:toe_per_turn:.1f} deg per turn, set continuously; jam nuts lock it |
| Caster, kingpin inclination | {req:K-03} / {req:K-04} deg | Fixed by the pickups |
| Ride height | as designed (the hardpoints are at static ride height) | The coilover's threaded spring perch: about {calc:ride_per_mm:.1f} mm at the wheel per mm of perch |
| Spring | {veh:spring_chosen_lbf_in:g} lb/in Hyperco on the TTX25 ({req:V-01} Hz) | Spring swap; this rate is the softest sold for the TTX25 |
| Anti-roll bar | middle bar: front TLLTD {veh:arb_middle_tlltd:pct1}, {veh:arb_middle_roll_gradient_deg_g:.2f} deg/g | Swap the bar: soft {veh:arb_soft_tlltd:pct1}, stiff {veh:arb_stiff_tlltd:pct1}; drop links off: {veh:tlltd_front_no_arb:pct1} |
| Bump steer | effectively zero as designed | Shims at the rack-end clevis if the built car needs them |
| Steering stop | the hairpin's lock + 0.5 deg | Fixed |
| Damping | not studied | TTX25 clickers (a quarter-car model is the next study) |

## What was caught and fixed

Most of the value is in what the checks found before anything was made. Each finding is in [notes/hardpoint_log.md](notes/hardpoint_log.md), with what it led to.

- **Kinematics and fit:**
  - a pushrod mount that steering would have twisted beyond its rod end's rating;
  - a rim that hit the wishbones at full lock (fixed with a 6 in rim and a re-optimisation inside the rim's clearance);
  - no room for a bump rubber.
- **Structure:**
  - hub bearings overloaded in cornering (static safety 0.69), found by the first free body of the upright and fixed with a tapered-roller pair;
  - a lower ball-joint bolt bending past yield;
  - a hub spindle that passed its fatigue check by 1 % (now a shoulder ring with an R3 fillet).
- **Brakes:**
  - a disc on a solid steel hat that would have passed its heat into the bearings (now a floating disc on titanium bobbins, margin {req:S-16} on an estimate);
  - nylon-insert lock nuts where the rules' 80 C limit would not hold.
- **Anti-roll bar:**
  - a first bar too soft for its blades and too weak in warp;
  - later, an adjustment by blades that could not carry a bolted drop link;
  - in the final audit, a fatigue check of the new bar without the reliability factor the upright's check uses. With it the bar failed, so it is now 4340 at 180 ksi with larger splined ends (FoS {hc:arb_bar_fatigue_fos:.2f}).
- **The checks themselves.** The analysis's own checks and independent code reviews found defects in the analysis, each fixed before its result was used:
  - FEA loads that did not balance, caught by the run's own equilibrium check;
  - a hand-calc model that read 50 % low;
  - a fastener sweep with gaps in its own bookkeeping;
  - a bar 26 mm longer in the analysis than in the CAD;
  - a risk table whose coverage check passed on stale results.

## Verification still to do

The design is checked by analysis; the car has not been built.

- **DFMEA.** It has {dfmea:rows} failure modes and says what would close each requirement that is not passing. The highest risk priority number is {dfmea:rpn_before} before the planned actions and {dfmea:rpn_after} after ([notes/dfmea.md](notes/dfmea.md)).
- **Test plan.** {dfmea:tests} tests:
  - alignment and a bump-steer gauge;
  - strain gauges on the pushrods to check the load model;
  - steering free play and stops ({req:R-06} deg predicted, against {tgt:R-06});
  - a proof test of one upright with strain rosettes, in a fixture that loads it as the car does, with the FEA to be re-run on the fixture's supports for the comparison. It is the direct check DV-2 lacks, and gives the baseline for the inspection interval (DV-3);
  - hub, housing and disc temperatures over an endurance-length run;
  - the anti-roll bar's rate per bar, and a tilt-table test of the load-transfer balance at two ride heights;
  - a fastener torque audit;
  - an inspection schedule.
- **Open items:**
  - A helper spring is still to be sourced: the main spring goes slack over the last {req:V-05} mm of droop.
  - The wheel studs are not detailed.
  - The brake's line pressure reaches about {veh:brake_line_pressure_max_bar:.0f} bar in the combined case. It is still to be checked against Wilwood's rating, with the pad and fluid temperatures.
  - Fretting is not modelled under the press-fitted bearing cones or at the bolted ear holes.
- **Assumptions,** each marked where it is used:
  - the load cases and the duty spectrum;
  - the brake's contact conductances;
  - the anti-roll bar's spline Kt (stress-concentration factor).

## Status of the requirements

The status is generated from the analysis outputs:

- {mx:PASS} pass;
- {mx:DEVIATION} accepted deviations ({mx:dv_ids});
- {mx:ACTION} action ({mx:actions});
- {mx:RETIRED} retired.

None is open, failed or stale; `scripts/writeup.py` refuses to write this document otherwise ([notes/requirements_matrix.md](notes/requirements_matrix.md)).

The hand-calculation checks quoted here (HC-1 to HC-12, `notes/handcalcs/`) are scripted answer keys: closed-form methods, independent of the solver. Most are also independent of the FEA; HC-6 and HC-7 are calibrated to it. The same twelve packets are set for me to work on paper; until that is done, none of them is claimed as checked by hand.

## How to check it

- **`python scripts/run_all.py`** (35-45 minutes) rebuilds the analysis outputs from the data files and the committed CAD meshes.
- **These run separately**, each recording a fingerprint of its inputs:
  - the upright FEA and fatigue (`scripts/fea/`, about 1.5 hours together);
  - the LC4 plot;
  - the upright's geometry and drawing;
  - the Fusion build and renders.

  The matrix and this document refuse their results once they are stale.
- **The CAD meshes** carry a fingerprint of the hardpoints, the data and the build scripts, so the sweeps refuse meshes built from other inputs. The upright's STEP, generated separately, is fingerprinted by the FEA.
- **The regression tests** run with `pytest tests`.
- **This document** is generated from a template whose results are filled in from those outputs (`scripts/writeup.py`); its other figures (inputs, targets and history) are quoted from the data files and the decision log.

## What I'd do next

- Build and test: the test plan above, starting with the pushrod strain gauges and the upright proof test.
- A quarter-car model to choose damping, and steering effort against lateral g.
- Design the chassis, and with it revisit the roll-centre target.
