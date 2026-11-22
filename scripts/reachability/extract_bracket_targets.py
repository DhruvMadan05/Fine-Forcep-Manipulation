#!/usr/bin/env python3
"""Extract the Task 1 reachability targets from the needle L-bracket STL.

Targets (all expressed in the BRACKET frame, metres, i.e. the STL frame
scaled by the ADF's 0.001):

  * holes  - every through-hole in either plate. Target point is the hole
             centre on the plate's INSIDE face; approach axis is the hole
             axis pointing outward (the direction the jaws/needle travel to
             go through the hole from the inside).
  * corner - points along the inside (concave) vertical edge of the L, found
             by ray-casting along the inside bisector so the inner fillet is
             respected; approach axis is that bisector (into the corner).

Also writes solid-box collision proxies for the two plates, and base_z
(the plane the bracket stands on, treated as the table surface).

The plates are detected from the mesh rather than hard-coded: the two
largest pairs of opposing axis-aligned planar faces. Holes are mesh
components of the hole-wall triangles, with a circle fit per component.

Writes config/task1_targets.yaml (committed; rerun if the STL changes).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_mesh import TriMesh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STL = REPO_ROOT / "meshes" / "task1" / "IROS_FFM_Needle_L_Bracket.stl"
DEFAULT_ADF = REPO_ROOT / "adf" / "task1" / "needle_bracket.yaml"
DEFAULT_OUT = REPO_ROOT / "config" / "task1_targets.yaml"
AXES = "xyz"


def find_plates(mesh: TriMesh):
    """Return plates as dicts: axis index, inner/outer face coords, inward
    normal sign. A plate = two large parallel axis-aligned faces ~thickness
    apart. 'Inner' is the face whose outward normal points toward the other
    plate (into the inside of the L)."""
    area = defaultdict(float)
    for nrm, c, a in zip(mesh.normals, mesh.centroids, mesh.area):
        ax = int(np.argmax(np.abs(nrm)))
        if abs(nrm[ax]) < 0.999 or ax == 2:      # vertical plates only
            continue
        area[(ax, int(np.sign(nrm[ax])), round(c[ax], 3))] += a
    faces = sorted(area.items(), key=lambda kv: -kv[1])[:4]
    plates = []
    for ax in (0, 1):
        f = [k for k, _ in faces if k[0] == ax]
        if len(f) != 2:
            sys.exit(f"ERROR: could not identify a plate normal to {AXES[ax]}")
        plates.append({"axis": ax, "faces": {s: c for _, s, c in f}})
    # The inside of the L lies toward the other plate's far end; use the
    # mesh bbox centre as a robust 'inside' reference.
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    for p in plates:
        ax = p["axis"]
        # face whose coordinate is closer to the bbox centre is the inner one
        (s1, c1), (s2, c2) = p["faces"].items()
        inner, outer = ((s1, c1), (s2, c2)) if abs(c1 - centre[ax]) < abs(c2 - centre[ax]) \
            else ((s2, c2), (s1, c1))
        p["inner_coord"], p["outer_coord"] = inner[1], outer[1]
        p["outward_sign"] = int(np.sign(outer[1] - inner[1]))  # inner -> outer
        p["thickness"] = abs(outer[1] - inner[1])
    return plates


def find_holes(mesh: TriMesh, plate, min_d=0.5, max_d=20.0, max_rms=0.05):
    ax = plate["axis"]
    lo_c, hi_c = sorted((plate["inner_coord"], plate["outer_coord"]))
    v = mesh.tri[:, :, ax]
    within = (v.min(1) >= lo_c - 1e-3) & (v.max(1) <= hi_c + 1e-3)
    wall = within & (np.abs(mesh.normals[:, ax]) < 1e-3) & (np.abs(mesh.normals[:, 2]) > 1e-3)
    if not wall.any():
        return []
    labels = mesh.components(wall)
    idx = np.flatnonzero(wall)
    other = [i for i in range(3) if i != ax]
    holes = []
    for lab in np.unique(labels):
        pts = mesh.tri[idx[labels == lab]][:, :, other].reshape(-1, 2)
        x, y = pts.T
        cx, cy, c = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], x**2 + y**2, rcond=None)[0]
        cx, cy = cx / 2, cy / 2
        r = np.sqrt(c + cx**2 + cy**2)
        rms = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
        if rms > max_rms or not (min_d <= 2 * r <= max_d):
            continue  # not a clean circular hole (e.g. chamfer/edge fragments)
        centre = np.zeros(3)
        centre[ax] = plate["inner_coord"]
        centre[other[0]], centre[other[1]] = cx, cy
        holes.append((centre, 2 * r))
    return holes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", default=str(DEFAULT_STL))
    ap.add_argument("--adf", default=str(DEFAULT_ADF),
                    help="Bracket ADF, used for the mesh scale")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--corner-step-mm", type=float, default=5.0)
    ap.add_argument("--corner-margin-mm", type=float, default=2.5,
                    help="Skip this much at the bottom/top of the corner edge")
    args = ap.parse_args()

    adf = yaml.safe_load(open(args.adf))
    body = adf[adf["bodies"][0]]
    scale = float(body.get("scale", 1.0))

    mesh = TriMesh.from_stl(args.stl)   # work in STL units (mm), convert at the end
    plates = find_plates(mesh)

    targets = []
    for p in plates:
        ax = p["axis"]
        axis = np.zeros(3)
        axis[ax] = p["outward_sign"]
        name = f"plate_{AXES[ax]}"   # plate normal to x / normal to y
        holes = find_holes(mesh, p)
        # stable ordering: by height (z) then along the plate
        along = [i for i in (0, 1) if i != ax][0]
        holes.sort(key=lambda h: (round(h[0][2], 3), round(h[0][along], 3)))
        for k, (c, d) in enumerate(holes):
            targets.append({"id": f"{name}_hole_{k:03d}", "type": "hole", "plate": name,
                            "position": (c * scale).round(6).tolist(),
                            "approach": axis.tolist(), "diameter": round(float(d * scale), 6)})
        print(f"{name}: inner face {AXES[ax]}={p['inner_coord']:.2f} mm, "
              f"thickness {p['thickness']:.2f} mm, {len(holes)} holes", file=sys.stderr)

    # Inside corner: intersection of the two inner face planes, bisector
    # pointing from the inside toward the corner.
    px = next(p for p in plates if p["axis"] == 0)
    py = next(p for p in plates if p["axis"] == 1)
    corner_xy = np.array([px["inner_coord"], py["inner_coord"]])
    bis = np.array([px["outward_sign"], py["outward_sign"], 0.0])
    bis /= np.linalg.norm(bis)
    lo, hi = mesh.bounds
    zs = np.arange(lo[2] + args.corner_margin_mm, hi[2] - args.corner_margin_mm + 1e-9,
                   args.corner_step_mm)
    start = np.c_[np.tile(corner_xy - 20.0 * bis[:2], (len(zs), 1)), zs]
    t = mesh.ray_hits(start, np.tile(bis, (len(zs), 1)))
    if not np.all(np.isfinite(t)):
        sys.exit("ERROR: corner ray-cast missed the bracket")
    for k, (s, tk) in enumerate(zip(start, t)):
        pt = s + tk * bis
        targets.append({"id": f"corner_{k:02d}", "type": "corner",
                        "position": (pt * scale).round(6).tolist(),
                        "approach": bis.round(6).tolist()})
    print(f"corner: {len(zs)} points, surface at {(start[0] + t[0] * bis)[:2].round(2)} mm "
          f"(sharp corner {corner_xy})", file=sys.stderr)

    # Solid-box collision proxies for the plates (holes and fillets are far
    # smaller than the tool shaft, so they don't matter for shaft clearance).
    boxes = []
    for p in plates:
        bmin, bmax = lo.copy(), hi.copy()
        ax = p["axis"]
        bmin[ax], bmax[ax] = sorted((p["inner_coord"], p["outer_coord"]))
        boxes.append({"name": f"plate_{AXES[ax]}", "min": (bmin * scale).round(6).tolist(),
                      "max": (bmax * scale).round(6).tolist()})

    out = {
        "_comment": "Generated by scripts/reachability/extract_bracket_targets.py from the "
                    "bracket STL. Positions in the bracket frame (m); 'approach' is the unit "
                    "direction the jaws point/travel at the target.",
        "bracket": {
            "stl": str(Path(args.stl).resolve().relative_to(REPO_ROOT)),
            "adf": str(Path(args.adf).resolve().relative_to(REPO_ROOT)),
            "scale": scale,
            "base_z": round(float(lo[2] * scale), 6),
        },
        "collision_boxes": boxes,
        "targets": targets,
    }
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None))
    print(f"Wrote {len(targets)} targets to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
