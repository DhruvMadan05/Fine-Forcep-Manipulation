#!/usr/bin/env python3
"""Search for the PSM RCM pose that best reaches the Task 1 bracket
targets (inside corner + every hole from the inside).

Pure math: bracket geometry from config/task1_targets.yaml (extracted from
the STL) and the dVRK DH model in config/psm_kinematics.yaml. No simulator.

For a candidate placement, each target is checked by:
  1. IK (psm_kinematics.py): jaw tip on the target point with the jaw axis
     within `approach_cone_deg` of the target's approach direction, joints
     inside their limits (tightened by `joint_limit_margin_rad`) and
     insertion >= `min_insertion_m`.
  2. Collision: tool shaft (RCM -> wrist) and wrist/jaws (minus the last
     `tip_clearance_m`) must not pass through the plates (solid-box proxies,
     inflated by the shaft/wrist radius) or below the table (bracket base).
  3. RCM clearance from the plates and table.

The objective is corner_weight * corner_fraction + hole_fraction plus a
small joint-limit-margin tie-break; scipy's differential evolution searches
position (3) + yaw + two tilts, on a hole subsample for speed, then the best
placement is re-evaluated on every target.

Outputs (results/reachability/<name>/):
  placement.yaml  - best RCM pose (bracket frame) + coverage
  targets.csv     - per-target reached / reason / joint solution
  coverage.png    - plate maps of reached holes + 3D view of the setup

Usage:
    python3 scripts/reachability/place_psm.py            # psm1, defaults
    python3 scripts/reachability/place_psm.py --name psm1 --maxiter 80
    python3 scripts/reachability/place_psm.py --evaluate results/reachability/psm1/placement.yaml
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psm_kinematics import JOINT_LABELS, PSMKinematics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "config" / "reachability.yaml"
DEFAULT_TARGETS = REPO_ROOT / "config" / "task1_targets.yaml"
DEFAULT_MODEL = REPO_ROOT / "config" / "psm_kinematics.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "reachability"
PARAM_NAMES = ["x", "y", "z", "yaw_deg", "tilt_y_deg", "tilt_x_deg"]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def segment_hits_boxes(a, b, boxes_min, boxes_max, inflate):
    """a, b: (K, 3) segment endpoints. Boxes (nb, 3). Returns (K,) bool."""
    d = b - a
    hit = np.zeros(len(a), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for bmin, bmax in zip(boxes_min - inflate, boxes_max + inflate):
            t1 = (bmin - a) / d
            t2 = (bmax - a) / d
            tmin = np.where(np.abs(d) < 1e-12,
                            np.where((a >= bmin) & (a <= bmax), -np.inf, np.inf),
                            np.minimum(t1, t2))
            tmax = np.where(np.abs(d) < 1e-12,
                            np.where((a >= bmin) & (a <= bmax), np.inf, -np.inf),
                            np.maximum(t1, t2))
            t_enter = tmin.max(axis=1)
            t_exit = tmax.min(axis=1)
            hit |= (t_enter <= t_exit) & (t_exit >= 0) & (t_enter <= 1)
    return hit


def point_box_distance(p, boxes_min, boxes_max):
    """p: (K, 3). Min distance from each point to any box (0 if inside)."""
    d = np.full(len(p), np.inf)
    for bmin, bmax in zip(boxes_min, boxes_max):
        delta = np.maximum(0, np.maximum(bmin - p, p - bmax))
        d = np.minimum(d, np.linalg.norm(delta, axis=1))
    return d


def placement_R(params):
    """params (..., 6) -> RCM-frame rotation in the bracket frame (..., 3, 3).
    Untilted (all angles 0) = RCM frame z-up with the tool pointing down."""
    params = np.atleast_2d(params)
    R = Rotation.from_euler("ZYX", np.deg2rad(params[:, 3:6])).as_matrix()  # Rz*Ry*Rx
    return R


# ----------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------
class PlacementProblem:
    def __init__(self, cfg, targets_doc, model: PSMKinematics, hole_subsample=1):
        self.cfg = cfg
        self.model = model
        tg = targets_doc["targets"]
        holes = [t for t in tg if t["type"] == "hole"]
        corner = [t for t in tg if t["type"] == "corner"]
        holes = holes[::hole_subsample]
        self.targets = corner + holes
        self.ids = [t["id"] for t in self.targets]
        self.is_corner = np.array([t["type"] == "corner" for t in self.targets])
        self.pts = np.array([t["position"] for t in self.targets])
        self.app = np.array([t["approach"] for t in self.targets])
        self.app /= np.linalg.norm(self.app, axis=1, keepdims=True)
        boxes = targets_doc["collision_boxes"]
        self.bmin = np.array([b["min"] for b in boxes])
        self.bmax = np.array([b["max"] for b in boxes])
        self.base_z = targets_doc["bracket"]["base_z"]
        lim = model.limits.copy()
        m = cfg["joint_limit_margin_rad"]
        for i in (0, 1, 3, 4, 5):
            lim[i] = [lim[i, 0] + m, lim[i, 1] - m]
        lim[2, 0] = max(lim[2, 0], cfg["min_insertion_m"])
        self.limits = lim

    def evaluate(self, params, iters=60, seeds=None, detail=False):
        """params (P, 6). Returns score (P,) and, if detail, per-target info."""
        cfg, model = self.cfg, self.model
        params = np.atleast_2d(params)
        P, N = len(params), len(self.targets)
        R = placement_R(params)              # (P, 3, 3) base->bracket
        t = params[:, :3]                                  # RCM = base origin
        # targets in each base frame
        pts_b = np.einsum("pji,pnj->pni", R, self.pts[None] - t[:, None])
        app_b = np.einsum("pji,nj->pni", R, self.app)
        pts_b, app_b = pts_b.reshape(-1, 3), app_b.reshape(-1, 3)

        # cheap prefilter so IK only runs on plausible rows
        rel = pts_b - model.rcm
        dist = np.linalg.norm(rel, axis=1)
        cos_ua = np.einsum("ij,ij->i", rel, app_b) / np.maximum(dist, 1e-9)
        plausible = ((dist > self.limits[2, 0] - 0.03) & (dist < self.limits[2, 1] + 0.03)
                     & (cos_ua > np.cos(np.deg2rad(95 + cfg["approach_cone_deg"]))))

        q = np.zeros((P * N, 6))
        pos_err = np.full(P * N, np.inf)
        ang_err = np.full(P * N, np.inf)
        if plausible.any():
            r = model.ik(pts_b[plausible], app_b[plausible], cone_deg=cfg["approach_cone_deg"],
                         iters=iters, seeds=seeds, limits=self.limits)
            q[plausible], pos_err[plausible], ang_err[plausible] = r["q"], r["pos_err"], r["ang_err"]
        ik_ok = (pos_err < cfg["position_tolerance_m"]) & (ang_err < 1e-3)

        # collisions (in bracket frame)
        rcm_b, wp_b, tip_b = model.shaft_points(q)
        _, axis_b = model.tip_and_axis(q)
        Rr = np.repeat(R, N, axis=0)
        tr = np.repeat(t, N, axis=0)
        to_br = lambda v: np.einsum("kij,kj->ki", Rr, v) + tr          # noqa: E731
        rcm_w, wp_w, tip_w = to_br(np.broadcast_to(rcm_b, wp_b.shape)), to_br(wp_b), to_br(tip_b)
        jaw_end = tip_w - cfg["tip_clearance_m"] * np.einsum("kij,kj->ki", Rr, axis_b)
        shaft_hit = segment_hits_boxes(rcm_w, wp_w, self.bmin, self.bmax, cfg["shaft_radius_m"])
        wrist_hit = segment_hits_boxes(wp_w, jaw_end, self.bmin, self.bmax, cfg["wrist_radius_m"])
        below_table = (np.minimum(wp_w[:, 2], rcm_w[:, 2]) < self.base_z + cfg["shaft_radius_m"]) | \
                      (jaw_end[:, 2] < self.base_z + cfg["wrist_radius_m"])
        collide = shaft_hit | wrist_hit | below_table
        ok = (ik_ok & ~collide).reshape(P, N)

        # RCM clearance (per placement)
        rcm_clear = (point_box_distance(t, self.bmin, self.bmax) >= cfg["rcm_clearance_m"]) & \
                    (t[:, 2] >= self.base_z + cfg["rcm_clearance_m"])
        ok &= rcm_clear[:, None]

        # joint-limit margin of reached targets, normalised to [0, 0.5]
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        qn = (q - lo) / (hi - lo)
        margin = np.minimum(qn, 1 - qn).min(axis=1).reshape(P, N)
        corner_frac = ok[:, self.is_corner].mean(axis=1)
        hole_frac = ok[:, ~self.is_corner].mean(axis=1)
        mean_margin = np.where(ok.any(axis=1),
                               (margin * ok).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), 0.0)
        score = cfg["corner_weight"] * corner_frac + hole_frac + cfg["margin_weight"] * mean_margin
        if not detail:
            return score
        reason = np.full(P * N, "ok", dtype=object)
        reason[~plausible] = "out_of_range"
        reason[plausible & ~ik_ok] = "ik_fail"
        reason[ik_ok & shaft_hit] = "shaft_collision"
        reason[ik_ok & ~shaft_hit & wrist_hit] = "wrist_collision"
        reason[ik_ok & ~shaft_hit & ~wrist_hit & below_table] = "below_table"
        reason = reason.reshape(P, N)
        reason[~rcm_clear] = "rcm_clearance"
        return score, {"ok": ok, "reason": reason, "q": q.reshape(P, N, 6),
                       "corner_frac": corner_frac, "hole_frac": hole_frac,
                       "margin": margin, "mean_margin": mean_margin,
                       "rcm": t, "wrist": wp_w.reshape(P, N, 3), "tip": tip_w.reshape(P, N, 3)}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_outputs(out_dir, name, params, problem, info, search_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, reason, q = info["ok"][0], info["reason"][0], info["q"][0]
    reasons, counts = np.unique(reason[~ok], return_counts=True)
    summary = {
        "_comment": "Generated by scripts/reachability/place_psm.py (pure math). "
                    "'bracket_frame' is the PSM RCM-frame pose relative to the bracket STL "
                    "frame (m, deg): R = Rz(yaw) Ry(tilt_y) Rx(tilt_x), where the untilted "
                    "RCM frame is z-up with the tool pointing down.",
        "arm": name,
        "bracket_frame": {k: round(float(v), 5) for k, v in zip(PARAM_NAMES, params)},
        "coverage": {
            "corner": f"{int(ok[problem.is_corner].sum())}/{int(problem.is_corner.sum())}",
            "holes": f"{int(ok[~problem.is_corner].sum())}/{int((~problem.is_corner).sum())}",
            "corner_fraction": round(float(info["corner_frac"][0]), 4),
            "hole_fraction": round(float(info["hole_frac"][0]), 4),
            "mean_joint_margin": round(float(info["mean_margin"][0]), 4),
            "unreached_by_reason": {str(r): int(c) for r, c in zip(reasons, counts)},
        },
        "search": search_info,
    }
    (out_dir / "placement.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    with open(out_dir / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "reached", "reason", "joint_margin"] + JOINT_LABELS)
        for i, tid in enumerate(problem.ids):
            w.writerow([tid, int(ok[i]), reason[i], round(float(info["margin"][0, i]), 4)]
                       + [round(float(v), 5) for v in q[i]])
    plot(out_dir / "coverage.png", name, problem, info, summary)
    return summary


def plot(path, name, problem, info, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = info["ok"][0]
    fig = plt.figure(figsize=(15, 5.2))
    plates = sorted({t["plate"] for t in problem.targets if t["type"] == "hole"})
    for k, plate in enumerate(plates):
        ax = fig.add_subplot(1, 3, k + 1)
        idx = [i for i, t in enumerate(problem.targets) if t.get("plate") == plate]
        P = problem.pts[idx] * 1000
        along = 1 if plate == "plate_x" else 0
        ax.scatter(P[:, along], P[:, 2], c=np.where(ok[idx], "tab:green", "tab:red"), s=18)
        cidx = np.flatnonzero(problem.is_corner)
        C = problem.pts[cidx] * 1000
        ax.scatter(C[:, along], C[:, 2], marker="s", s=30,
                   c=np.where(ok[cidx], "tab:green", "tab:red"), edgecolors="k")
        ax.set_title(f"{plate}: {int(ok[idx].sum())}/{len(idx)} holes (squares = corner)")
        ax.set_xlabel(f"{'y' if along == 1 else 'x'} (mm, bracket frame)")
        ax.set_ylabel("z (mm)")
        ax.set_aspect("equal")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for bmin, bmax in zip(problem.bmin, problem.bmax):
        _draw_box(ax3, bmin, bmax)
    rcm = info["rcm"][0]
    sel = np.flatnonzero(ok)[:: max(1, int(ok.sum()) // 25)]
    for i in sel:
        w, tp = info["wrist"][0, i], info["tip"][0, i]
        ax3.plot(*np.c_[rcm, w], color="tab:blue", alpha=0.35, lw=1)
        ax3.plot(*np.c_[w, tp], color="tab:orange", lw=1.5)
    ax3.scatter(*rcm, color="k", s=40)
    ax3.text(*rcm, " RCM")
    pts = np.vstack([problem.bmin, problem.bmax, rcm[None]])
    c, r = pts.mean(0), (pts.max(0) - pts.min(0)).max() / 2 + 0.01
    ax3.set_xlim(c[0] - r, c[0] + r); ax3.set_ylim(c[1] - r, c[1] + r); ax3.set_zlim(c[2] - r, c[2] + r)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("Sample tool poses (blue shaft, orange wrist)")
    cov = summary["coverage"]
    fig.suptitle(f"{name}: corner {cov['corner']}, holes {cov['holes']}  |  "
                 f"bracket frame {summary['bracket_frame']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _draw_box(ax, lo, hi):
    import itertools
    for s, e in itertools.combinations(np.array(list(itertools.product(*zip(lo, hi)))), 2):
        if np.sum(np.abs(s - e) > 1e-12) == 1:
            ax.plot(*zip(s, e), color="0.4", lw=0.8)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="psm1", help="Arm name (results folder name)")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--maxiter", type=int, help="Override search.maxiter")
    ap.add_argument("--popsize", type=int, help="Override search.popsize")
    ap.add_argument("--max-tilt-deg", type=float,
                    help="Override search.tilt_x_deg / tilt_y_deg to +-this value")
    ap.add_argument("--evaluate", help="Skip the search; evaluate the placement in this placement.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    targets_doc = yaml.safe_load(Path(args.targets).read_text())
    model = PSMKinematics.load(args.model)
    sc = cfg["search"]
    if args.max_tilt_deg is not None:
        sc["tilt_x_deg"] = sc["tilt_y_deg"] = [-args.max_tilt_deg, args.max_tilt_deg]
    out_dir = RESULTS_DIR / args.name
    full = PlacementProblem(cfg, targets_doc, model, hole_subsample=1)

    if args.evaluate:
        bf = yaml.safe_load(Path(args.evaluate).read_text())["bracket_frame"]
        best = np.array([bf[k] for k in PARAM_NAMES])
        search_info = {"evaluated_from": str(args.evaluate)}
    else:
        coarse = PlacementProblem(cfg, targets_doc, model,
                                  hole_subsample=sc["hole_subsample"])
        bounds = list(zip(sc["position_min"], sc["position_max"])) + \
            [tuple(sc["yaw_deg"]), tuple(sc["tilt_y_deg"]), tuple(sc["tilt_x_deg"])]
        n_eval = [0]
        t0 = time.time()

        def objective(X):          # X: (6, S) with vectorized=True
            s = coarse.evaluate(X.T, iters=30, seeds=[0.0, np.pi / 2])
            n_eval[0] += X.shape[1]
            return -s

        def progress(xk, convergence=0.0):
            s = coarse.evaluate(xk[None], iters=30, seeds=[0.0, np.pi / 2])[0]
            print(f"  [{time.time() - t0:6.0f}s, {n_eval[0]} evals] best score {s:.4f} "
                  f"at {np.round(xk, 3).tolist()}", flush=True)

        print(f"Searching placement for {args.name}: {len(coarse.targets)} targets "
              f"(holes subsampled 1/{sc['hole_subsample']})", flush=True)
        res = differential_evolution(objective, bounds, popsize=args.popsize or sc["popsize"],
                                     maxiter=args.maxiter or sc["maxiter"], seed=sc["seed"],
                                     vectorized=True, updating="deferred", polish=False,
                                     tol=1e-6, callback=progress)
        best = res.x
        search_info = {"evaluations": int(n_eval[0]), "seconds": round(time.time() - t0, 1),
                       "coarse_score": round(float(-res.fun), 5),
                       "config": str(Path(args.config).resolve().relative_to(REPO_ROOT)),
                       "tilt_bounds_deg": {"x": sc["tilt_x_deg"], "y": sc["tilt_y_deg"]}}

    score, info = full.evaluate(best[None], iters=100, detail=True)
    search_info["final_score"] = round(float(score[0]), 5)
    summary = write_outputs(out_dir, args.name, best, full, info, search_info)
    print(yaml.safe_dump({k: summary[k] for k in ("bracket_frame", "coverage")},
                         sort_keys=False))
    print(f"Wrote {out_dir.relative_to(REPO_ROOT)}/{{placement.yaml,targets.csv,coverage.png}}")


if __name__ == "__main__":
    main()
