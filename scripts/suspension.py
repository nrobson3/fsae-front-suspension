"""Kinematic model of one double-wishbone, pushrod-rocker corner (LEFT front).

The upright is a rigid body held by five fixed-length links (4 wishbone legs + tie rod).
Wheel-centre height (body frame) is the driving coordinate, rack travel moves the tie-rod
inboard point in y. The rocker and ARB lever are each one-DOF bodies solved after the
upright. Right-corner results come from mirror symmetry (y -> -y), see kinematics.py.

Sign conventions (left wheel, ISO 8855 x fwd / y left / z up):
  z travel   + = bump (wheel up relative to body)
  camber     - = top of wheel leans inboard
  steer      + = wheel heading rotates to the left (CCW seen from above); toe-in = -steer
  caster     + = top of kingpin rearward;  KPI + = top of kingpin inboard
  scrub      + = kingpin hits ground inboard of contact patch centre
  trail      + = kingpin hits ground ahead of contact patch centre
  motion ratio = damper compression / wheel bump (both mm)
"""
import math

import numpy as np
from scipy.optimize import fsolve

from common import EZ, MIRROR, rotate_about, rotvec_matrix, skew, unit

UPRIGHT_PTS = ("UCA_O", "LCA_O", "TR_O", "WC", "WA")  # + PR_O when the pushrod picks up on the upright
ROCKER_PTS = ("PR_I", "DMP_R", "ARB_R")
LINKS = (("UCA_O", "UCA_F"), ("UCA_O", "UCA_R"), ("LCA_O", "LCA_F"), ("LCA_O", "LCA_R"), ("TR_O", "TR_I"))


class SolveError(RuntimeError):
    pass


def axis_angle(p, axis_a, axis_b, p_new):
    """Angle [rad] of the rotation about the fixed axis a->b that takes point p to p_new."""
    k = unit(axis_b - axis_a)
    r0, r1 = p - axis_a, p_new - axis_a
    r0 = r0 - np.dot(r0, k) * k
    r1 = r1 - np.dot(r1, k) * k
    return math.atan2(np.dot(np.cross(r0, r1), k), np.dot(r0, r1))


class Corner:
    def __init__(self, hp, tyre_radius, pushrod_on="lca"):
        """pushrod_on: 'lca' (v1.3+: pickup on the lower wishbone, moves with it) or 'upright'
        (v1.0-v1.2: pickup on the upright, moves with it)."""
        if pushrod_on not in ("lca", "upright"):
            raise ValueError(pushrod_on)
        self.hp = {k: np.asarray(v, float).copy() for k, v in hp.items()}
        self.R = float(tyre_radius)
        self.pushrod_on = pushrod_on
        self.upright_pts = UPRIGHT_PTS + (("PR_O",) if pushrod_on == "upright" else ())
        h = self.hp
        self.c0 = h["WC"].copy()
        self.lengths = np.array([np.linalg.norm(h[a] - h[b]) for a, b in LINKS])
        self.rk_axis = unit(h["RKR_A"] - h["RKR_P"])
        self.arb_axis = unit(h["ARB_A"] - h["ARB_P"])
        self.L_push = np.linalg.norm(h["PR_I"] - h["PR_O"])
        self.L_drop = np.linalg.norm(h["ARB_L"] - h["ARB_R"])
        self.L_damper0 = np.linalg.norm(h["DMP_R"] - h["DMP_C"])

    # ---------------- upright ----------------
    def upright(self, q):
        R = rotvec_matrix(q[:3])
        return {k: self.c0 + R @ (self.hp[k] - self.c0) + q[3:] for k in self.upright_pts}

    def _residual(self, q, z, rack):
        P = self.upright(q)
        r = np.empty(6)
        for i, (a, b) in enumerate(LINKS):
            pb = self.hp[b] + np.array([0.0, rack, 0.0]) if b == "TR_I" else self.hp[b]
            r[i] = np.linalg.norm(P[a] - pb) - self.lengths[i]
        r[5] = P["WC"][2] - (self.c0[2] + z)
        return r

    def _solve_q(self, z, rack, q0):
        # Solve for the increment dq from q0 (start dq = 0). fsolve's finite-difference step scales
        # with |x|, so iterating on q itself breaks down when a component is round-off small (1e-21).
        res = lambda dq: self._residual(q0 + dq, z, rack)
        dq, _, ier, msg = fsolve(res, np.zeros(6), xtol=1e-12, full_output=True)
        q = q0 + dq
        # judge on the closure residual: fsolve can report "no progress" once already at round-off
        if np.abs(self._residual(q, z, rack)).max() > 1e-8:
            raise SolveError(f"upright did not assemble at z={z:.2f} rack={rack:.2f}: {msg}")
        return q

    def solve_q(self, z, rack=0.0, q0=None, step=2.0):
        """Upright pose q = (rotation vector, translation). Marches from q0 (default static)
        in steps <= `step` mm so the solver stays on the physical assembly branch."""
        if q0 is not None:
            try:
                return self._solve_q(z, rack, q0)
            except SolveError:
                pass
        n = max(1, math.ceil(max(abs(z), abs(rack)) / step))
        q = np.zeros(6)
        for i in range(1, n + 1):
            q = self._solve_q(z * i / n, rack * i / n, q)
        return q

    # ---------------- rocker and ARB (1-DOF each) ----------------
    @staticmethod
    def _newton_1d(f, x0, what):
        x = x0
        for _ in range(60):
            fx = f(x)
            d = (f(x + 1e-7) - f(x - 1e-7)) / 2e-7
            if d == 0:
                break
            dx = fx / d
            dx = max(-0.2, min(0.2, dx))
            x -= dx
            if abs(dx) < 1e-13:
                break
        if abs(f(x)) > 1e-8:
            raise SolveError(f"{what} did not close (residual {f(x):.2e} mm)")
        return x

    def rocker_angle(self, pr_o, th0=0.0):
        h = self.hp
        f = lambda th: np.linalg.norm(rotate_about(h["PR_I"], h["RKR_P"], self.rk_axis, th) - pr_o) - self.L_push
        return self._newton_1d(f, th0, "pushrod loop")

    def arb_angle(self, arb_r, be0=0.0):
        h = self.hp
        f = lambda be: np.linalg.norm(rotate_about(h["ARB_L"], h["ARB_P"], self.arb_axis, be) - arb_r) - self.L_drop
        return self._newton_1d(f, be0, "ARB drop-link loop")

    # ---------------- full pose ----------------
    def pose(self, z, rack=0.0, warm=None):
        """Solve the whole corner. `warm` = a previous pose dict near this one (optional)."""
        if warm is not None and max(abs(z - warm["z"]), abs(rack - warm["rack"])) > 2.0:
            warm = None  # too far away to trust the warm start: march from static instead
        q = self.solve_q(z, rack, None if warm is None else warm["q"])
        P = self.upright(q)
        h = self.hp
        lca_ang = axis_angle(h["LCA_O"], h["LCA_F"], h["LCA_R"], P["LCA_O"])
        if self.pushrod_on == "lca":
            P["PR_O"] = rotate_about(h["PR_O"], h["LCA_F"], unit(h["LCA_R"] - h["LCA_F"]), lca_ang)
        th = self.rocker_angle(P["PR_O"], 0.0 if warm is None else warm["th"])
        for k in ROCKER_PTS:
            P[k] = rotate_about(h[k], h["RKR_P"], self.rk_axis, th)
        be = self.arb_angle(P["ARB_R"], 0.0 if warm is None else warm["be"])
        P["ARB_L"] = rotate_about(h["ARB_L"], h["ARB_P"], self.arb_axis, be)
        for k in ("UCA_F", "UCA_R", "LCA_F", "LCA_R", "RKR_P", "RKR_A", "DMP_C", "ARB_P", "ARB_A"):
            P[k] = h[k].copy()
        P["TR_I"] = h["TR_I"] + np.array([0.0, rack, 0.0])
        out = {"z": z, "rack": rack, "q": q, "th": th, "be": be, "lca_angle": lca_ang, "P": P}
        out.update(wheel_geometry(P, self.R))
        out["damper_len"] = np.linalg.norm(P["DMP_R"] - P["DMP_C"])
        return out

    def velocity(self, pose, dz=0.02):
        """Upright twist per mm of wheel-centre travel at `pose` (central difference).
        Returns (w, v_c, c, mr, arb_rate): angular velocity [rad/mm], velocity of point c [mm/mm],
        c = WC, motion ratio (damper compression / wheel bump) and ARB lever rate [rad/mm]."""
        pm = self.pose(pose["z"] - dz, pose["rack"], warm=pose)
        pp = self.pose(pose["z"] + dz, pose["rack"], warm=pose)
        c = pose["P"]["WC"]
        A, b = [], []
        for k in self.upright_pts:
            v = (pp["P"][k] - pm["P"][k]) / (2 * dz)
            A.append(np.hstack([np.eye(3), -skew(pose["P"][k] - c)]))
            b.append(v)
        sol = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
        mr = -(pp["damper_len"] - pm["damper_len"]) / (2 * dz)
        arb_rate = (pp["be"] - pm["be"]) / (2 * dz)  # rad per mm
        return sol[3:], sol[:3], c, mr, arb_rate

    def point_velocities(self, pose, dz=0.02):
        """Velocity [mm per mm of wheel travel] of every solved point, by central difference."""
        pm = self.pose(pose["z"] - dz, pose["rack"], warm=pose)
        pp = self.pose(pose["z"] + dz, pose["rack"], warm=pose)
        return {k: (pp["P"][k] - pm["P"][k]) / (2 * dz) for k in pose["P"]}


def make_corner(hp, meta, tyre_radius):
    """Corner whose pushrod pickup body ('lca' or 'upright') comes from the hardpoint CSV."""
    body = meta.get("PR_O", {}).get("body", "upright")
    return Corner(hp, tyre_radius, pushrod_on=body if body in ("lca", "upright") else "upright")


def wheel_geometry(P, R):
    """Camber, steer, contact patch, kingpin metrics for the LEFT wheel in the body frame."""
    n = unit(P["WA"] - P["WC"])                       # spin axis, outboard
    camber = -math.degrees(math.asin(n[2]))
    h = unit(np.cross(n, EZ))                          # heading (horizontal, forward)
    steer = math.degrees(math.atan2(h[1], h[0]))
    d = unit(-EZ + n[2] * n)                           # down, in wheel plane
    cp = P["WC"] + R * d                               # rigid-wheel contact patch
    k = unit(P["UCA_O"] - P["LCA_O"])                  # kingpin axis, upward
    caster = math.degrees(math.atan2(-k[0], k[2]))
    kpi = math.degrees(math.atan2(-k[1], k[2]))
    g = P["LCA_O"] + (cp[2] - P["LCA_O"][2]) / k[2] * k  # kingpin at ground (contact-patch height)
    lat = np.cross(EZ, h)                              # horizontal, outboard
    return {"n": n, "heading": h, "camber": camber, "steer": steer, "cp": cp, "kingpin": k,
            "caster": caster, "kpi": kpi, "kp_ground": g,
            "scrub": float(np.dot(cp - g, lat)), "trail": float(np.dot(g - cp, h))}


def point_velocity(w, v_c, c, p):
    return v_c + np.cross(w, p - c)


def front_view_ic(w, v_c, c, cp):
    """Front-view instant centre seen from the contact patch.

    In front view the upright turns about the IC at the camber rate w_x, so the
    upright-fixed point at the contact patch moves as v = w_x * ex x (cp - IC).
    Inverting that gives the IC. Using the contact-patch velocity (not the raw screw
    axis) keeps the roll centre consistent with its kinematic definition, RC height =
    half-track * (lateral scrub rate), even when there is some bump steer.
    """
    v = point_velocity(w, v_c, c, cp)
    if abs(w[0]) < 1e-12:  # no camber rate (e.g. parallel equal arms): IC at infinity
        return np.array([cp[0], np.nan, np.nan])
    return np.array([cp[0], cp[1] - v[2] / w[0], cp[2] + v[1] / w[0]])


def front_view_line(w, v_c, c, cp):
    """Direction (dy, dz) of the contact-patch -> IC line, from the contact-patch velocity alone.
    Stays finite when the IC is at infinity, so roll centres can always be intersected."""
    v = point_velocity(w, v_c, c, cp)
    return np.array([-v[2], v[1]])


def mirror(p):
    return p * MIRROR


def line_intersect_2d(p1, d1, p2, d2):
    """Intersection of two lines in the (y, z) plane: p + t d."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1
