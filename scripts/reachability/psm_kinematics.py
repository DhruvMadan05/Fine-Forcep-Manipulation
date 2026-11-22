#!/usr/bin/env python3
"""PSM kinematics from the dVRK DH specification (pure math, no simulator).

Model
-----
config/psm_kinematics.yaml gives the dVRK PSM modified-DH table, jaw length
and joint limits. The DH chain is converted once (at q = 0) into a product
of exponentials (PoE) in the RCM/base frame:

    T_base_jaw(q) = exp([S1] q1) ... exp([S6] q6) M

q = (yaw, pitch, insertion, roll, wrist pitch, jaw), jaws closed. The jaw
tip is the last frame's origin + jaw_length * its +y axis, which is also
the jaw pointing axis. The RCM is the base-frame origin.

The PoE form gives a cheap analytic space Jacobian, used by the batched,
joint-limit-aware IK below (jaw tip on a point, jaw axis inside a cone).

Run directly for a self-check (DH vs PoE agreement, IK round trips):

    python3 scripts/reachability/psm_kinematics.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "config" / "psm_kinematics.yaml"
JOINT_LABELS = ["yaw", "pitch", "insertion", "roll", "wrist_pitch", "jaw"]


# ----------------------------------------------------------------------------
# SE(3) helpers (batched over leading dims)
# ----------------------------------------------------------------------------
def skew(w):
    w = np.asarray(w)
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1),
                     np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def exp_screw(S, theta):
    """S: (6,) screw [w, v]; theta: (...,). Returns (..., 4, 4)."""
    theta = np.asarray(theta, dtype=float)
    w, v = S[:3], S[3:]
    T = np.zeros(theta.shape + (4, 4))
    T[..., 3, 3] = 1.0
    if np.linalg.norm(w) < 1e-9:          # prismatic
        T[..., :3, :3] = np.eye(3)
        T[..., :3, 3] = theta[..., None] * v
        return T
    W = skew(w)
    W2 = W @ W
    s, c = np.sin(theta)[..., None, None], np.cos(theta)[..., None, None]
    R = np.eye(3) + s * W + (1 - c) * W2
    G = np.eye(3) * theta[..., None, None] + (1 - c) * W + (theta[..., None, None] - s) * W2
    T[..., :3, :3] = R
    T[..., :3, 3] = G @ v
    return T


def adjoint(T):
    R, p = T[..., :3, :3], T[..., :3, 3]
    A = np.zeros(T.shape[:-2] + (6, 6))
    A[..., :3, :3] = R
    A[..., 3:, 3:] = R
    A[..., 3:, :3] = skew(p) @ R
    return A


def dh_modified(alpha, a, theta, d):
    """Craig modified DH: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0, 0, 0, 1.0]])


def rpy_to_R(r, p, y):
    """AMBF 'orientation: {r, p, y}' = fixed-axis XYZ roll-pitch-yaw."""
    return Rotation.from_euler("xyz", [r, p, y]).as_matrix()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class PSMKinematics:
    def __init__(self, screws, M, jaw_length, limits, rcm):
        self.S = np.asarray(screws, dtype=float)       # (6, 6) in base frame
        self.M = np.asarray(M, dtype=float)            # (4, 4) in base frame
        self.jaw_length = float(jaw_length)
        self.limits = np.asarray(limits, dtype=float)  # (6, 2)
        self.rcm = np.asarray(rcm, dtype=float)        # (3,) in base frame

    # ---- construction -----------------------------------------------------
    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        """Build from the DH spec in config/psm_kinematics.yaml."""
        spec = yaml.safe_load(Path(path).read_text())
        rows = spec["dh"]
        if [r["joint"] for r in rows] != JOINT_LABELS:
            sys.exit(f"ERROR: {path} dh joints must be ordered {JOINT_LABELS}")
        A = np.eye(4)
        screws = []
        for r in rows:
            A = A @ dh_modified(r["alpha"], r["a"], r["theta"], r["d"])  # frame i at q=0
            z, p = A[:3, 2], A[:3, 3]
            screws.append(np.r_[np.zeros(3), z] if r["type"] == "prismatic"
                          else np.r_[z, -np.cross(z, p)])
        model = cls(screws, A, spec["jaw_length"],
                    [spec["limits"][k] for k in JOINT_LABELS], np.zeros(3))
        model.dh = rows
        return model

    def fk_dh(self, q):
        """Direct DH evaluation (single q), used only to self-check the PoE."""
        A = np.eye(4)
        for r, qi in zip(self.dh, q):
            if r["type"] == "prismatic":
                A = A @ dh_modified(r["alpha"], r["a"], r["theta"], r["d"] + qi)
            else:
                A = A @ dh_modified(r["alpha"], r["a"], r["theta"] + qi, r["d"])
        return A

    # ---- forward kinematics -----------------------------------------------
    def fk_all(self, q):
        """q: (..., 6). Returns list of cumulative products P_i = e^S1q1..e^Siqi
        (each (..., 4, 4)) and T = P_6 M."""
        q = np.asarray(q, dtype=float)
        P = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        prods = []
        for i in range(6):
            P = P @ exp_screw(self.S[i], q[..., i])
            prods.append(P)
        return prods, P @ self.M

    def fk(self, q):
        return self.fk_all(q)[1]

    def tip_and_axis(self, q):
        """Jaw tip position and jaw pointing axis, both (..., 3), base frame."""
        T = self.fk(q)
        axis = T[..., :3, 1]
        return T[..., :3, 3] + self.jaw_length * axis, axis

    def shaft_points(self, q):
        """RCM, wrist-pitch point (end of shaft) and jaw tip, base frame.
        The wrist point is where the shaft ends: RCM + (insertion offset) * dir,
        computed from the roll screw's axis (shaft axis)."""
        prods, T = self.fk_all(q)
        tip = T[..., :3, 3] + self.jaw_length * T[..., :3, 1]
        # wrist pitch axis point, transformed by joints 1..4 (it's on the
        # wrist-pitch screw axis; precomputed point in zero config)
        wp = (prods[3] @ np.append(self._wrist_point0, 1.0))[..., :3]
        return np.broadcast_to(self.rcm, wp.shape), wp, tip

    @property
    def _wrist_point0(self):
        if not hasattr(self, "_wp0"):
            w, v = self.S[4, :3], self.S[4, 3:]
            # point on the axis closest to the gripper origin at zero config
            r0 = np.cross(w, v) / np.dot(w, w)
            g = self.M[:3, 3]
            self._wp0 = r0 + np.dot(g - r0, w) * w / np.dot(w, w)
        return self._wp0

    def space_jacobian(self, q):
        prods, T = self.fk_all(q)
        J = np.empty(np.shape(q)[:-1] + (6, 6))
        J[..., :, 0] = self.S[0]
        for i in range(1, 6):
            J[..., :, i] = np.einsum("...ij,j->...i", adjoint(prods[i - 1]), self.S[i])
        return J, T

    # ---- inverse kinematics -----------------------------------------------
    def ik(self, tip, approach, cone_deg=20.0, iters=60, seeds=None, damping=1e-4, limits=None):
        """Batched IK for jaw-tip position + jaw axis within a cone.

        tip, approach: (N, 3) in base frame (approach = desired jaw axis).
        The jaw axis only has to lie within cone_deg of approach; at each
        iteration the target axis is the point of the cone closest to the
        current jaw axis (so the solver uses the slack to reduce wrist bend).
        Roll about the jaw axis is free. Joint limits enforced by projection.

        limits: optional (6, 2) override of the joint limits (e.g. tightened
        by a safety margin or a minimum insertion depth).

        Returns dict with q (N, 6), pos_err (N,), ang_err (N,) [deg beyond
        cone], best over the seeds.
        """
        tip = np.atleast_2d(tip).astype(float)
        a = np.atleast_2d(approach).astype(float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cone = np.deg2rad(cone_deg)
        limits = self.limits if limits is None else np.asarray(limits, dtype=float)
        lo, hi = limits[:, 0], limits[:, 1]
        if seeds is None:
            seeds = [0.0, np.pi / 2, -np.pi / 2, np.pi * 0.95]   # roll seeds
        best = None
        for roll0 in seeds:
            q = self._initial_guess(tip, a, roll0, limits)
            for _ in range(iters):
                J, T = self.space_jacobian(q)
                R, p = T[:, :3, :3], T[:, :3, 3]
                axis = R[:, :, 1]
                cur_tip = p + self.jaw_length * axis
                tgt_axis = _project_into_cone(axis, a, cone)
                # orientation error: rotate axis onto tgt_axis (2-DOF, roll free)
                w_err = np.cross(axis, tgt_axis)
                # space-frame twist relating tip velocity: v_tip = v_s + w x tip
                pos_err = tip - cur_tip
                # Task Jacobian rows for [w (3), v_tip (3)]:
                # v_tip = v_s + w x p_tip  ->  Jv_tip = Jv_s - [p_tip]x Jw
                Jw = J[:, :3, :]
                Jv = J[:, 3:, :] - skew(cur_tip) @ Jw
                # remove the roll-about-jaw-axis component from orientation rows
                Pa = np.eye(3) - axis[:, :, None] * axis[:, None, :]
                Jt = np.concatenate([Pa @ Jw, Jv], axis=1)          # (N, 6, 6)
                e = np.concatenate([w_err, pos_err], axis=1)        # (N, 6)
                # scale: 1 rad ~ 10 mm
                Wt = np.array([0.01, 0.01, 0.01, 1, 1, 1])
                Jt = Jt * Wt[None, :, None]
                e = e * Wt[None]
                JtT = Jt.transpose(0, 2, 1)
                H = JtT @ Jt + damping * np.eye(6)
                dq = np.linalg.solve(H, (JtT @ e[..., None]))[..., 0]
                dq = np.clip(dq, -0.3, 0.3)
                dq[:, 2] = np.clip(dq[:, 2], -0.03, 0.03)
                q = np.clip(q + dq, lo, hi)
            res = self._residuals(q, tip, a, cone)
            if best is None:
                best = res
            else:
                better = _score(res) < _score(best)
                for k in best:
                    best[k] = np.where(better[(...,) + (None,) * (best[k].ndim - 1)], res[k], best[k])
        return best

    def _residuals(self, q, tip, a, cone):
        cur_tip, axis = self.tip_and_axis(q)
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", axis, a), -1, 1))
        return {"q": q, "pos_err": np.linalg.norm(cur_tip - tip, axis=1),
                "ang_err": np.rad2deg(np.maximum(0.0, ang - cone))}

    def _initial_guess(self, tip, a, roll0, limits):
        """Point the shaft from the RCM at the wrist point estimate, wrist
        straight, by a tiny 3-DOF solve over (yaw, pitch, insertion)."""
        q = np.zeros((len(tip), 6))
        q[:, 3] = roll0
        wrist_target = tip - (self.jaw_length + 0.009) * a
        q[:, 2] = np.clip(np.linalg.norm(wrist_target - self.rcm, axis=1), *limits[2])
        for _ in range(15):
            J, T = self.space_jacobian(q)
            _, wp, _ = self.shaft_points(q)
            e = wrist_target - wp
            Jv = (J[:, 3:, :3] - skew(wp) @ J[:, :3, :3])
            H = Jv.transpose(0, 2, 1) @ Jv + 1e-6 * np.eye(3)
            d = np.linalg.solve(H, (Jv.transpose(0, 2, 1) @ e[..., None]))[..., 0]
            q[:, :3] = np.clip(q[:, :3] + np.clip(d, -0.3, 0.3), limits[:3, 0], limits[:3, 1])
        return q


def _project_into_cone(axis, a, cone):
    """Closest unit vector to `axis` within `cone` of `a` (rows)."""
    c = np.clip(np.einsum("ij,ij->i", axis, a), -1, 1)
    ang = np.arccos(c)
    inside = ang <= cone
    perp = axis - c[:, None] * a
    n = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.where(n > 1e-9, perp / np.maximum(n, 1e-12), _any_perp(a))
    edge = np.cos(cone) * a + np.sin(cone) * perp
    return np.where(inside[:, None], axis, edge)


def _any_perp(a):
    t = np.where(np.abs(a[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    p = np.cross(a, t)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _score(res):
    return res["pos_err"] * 1000.0 + res["ang_err"]   # mm + deg


# ----------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("-n", type=int, default=2000, help="Random configurations to test")
    args = ap.parse_args()

    m = PSMKinematics.load(args.model)
    rng = np.random.default_rng(0)
    lo, hi = m.limits[:, 0], m.limits[:, 1]
    q = rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), (args.n, 6))
    q[:, 2] = rng.uniform(0.03, 0.22, args.n)

    T_poe = m.fk(q)
    T_dh = np.array([m.fk_dh(qi) for qi in q])
    print(f"PoE vs DH max difference: {np.abs(T_poe - T_dh).max():.2e}")

    tip0, ax0 = m.tip_and_axis(np.zeros(6))
    print(f"q=0: jaw tip {tip0.round(4)} m, jaw axis {ax0.round(3)} (shaft points -z)")

    tip, ax = m.tip_and_axis(q)
    r = m.ik(tip, ax, cone_deg=0.0)
    ok = (r["pos_err"] < 1e-4) & (r["ang_err"] < 0.5)
    print(f"IK round trip (exact axis): {ok.mean() * 100:.1f}% of {args.n} reachable poses, "
          f"median pos err {np.median(r['pos_err']) * 1e3:.4f} mm")
    pert = Rotation.from_rotvec(rng.normal(size=(args.n, 3)) * np.deg2rad(8)).apply(ax)
    r2 = m.ik(tip, pert, cone_deg=20.0)
    ok2 = (r2["pos_err"] < 1e-4) & (r2["ang_err"] < 1e-3)
    print(f"IK round trip (20 deg cone, perturbed axis): {ok2.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
