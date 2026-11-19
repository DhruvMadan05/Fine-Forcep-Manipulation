#!/usr/bin/env python3
"""Generate per-arm AMBF Description Files (ADFs) from the stock PSM/MTM
models shipped with AMBF, per config/arms.yaml.

Each generated ADF is a copy of the stock blender-<source_model>.yaml with:
  - a unique "namespace" (so the 4 arms don't collide on ROS topics)
  - every body's world position shifted by "world_offset" (so they don't
    spawn on top of each other)
  - "high/low resolution path" rewritten to an absolute path, since the
    generated file lives outside the AMBF models tree

Run directly to (re)generate everything; prints the comma-separated list of
generated file paths on the last line of stdout, for use by launch_sim.sh.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "adf" / "generated"


def get_ambf_ws() -> Path:
    ambf_ws = os.environ.get("AMBF_WS")
    if not ambf_ws:
        sys.exit("ERROR: AMBF_WS is not set. source scripts/env.sh first.")
    ambf_ws = Path(ambf_ws).expanduser().resolve()
    if not (ambf_ws / "install" / "setup.bash").exists():
        sys.exit(f"ERROR: {ambf_ws} doesn't look like a built AMBF workspace "
                  f"(missing install/setup.bash).")
    return ambf_ws


def stock_model_path(ambf_ws: Path, source_model: str) -> Path:
    path = (ambf_ws / "src" / "ambf" / "core" / "ambf_models" / "descriptions"
            / "multi-bodies" / "robots" / f"blender-{source_model}.yaml")
    if not path.exists():
        sys.exit(f"ERROR: stock ADF not found: {path}")
    return path


def overwrite_namespaces(node, new_namespace):
    """Stock ADFs can carry a per-body 'namespace' override in addition to
    the top-level one (e.g. target_fk in blender-psm.yaml); walk the whole
    doc so no leftover instance gets stamped with the stock namespace."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "namespace":
                node[key] = new_namespace
            else:
                overwrite_namespaces(value, new_namespace)
    elif isinstance(node, list):
        for item in node:
            overwrite_namespaces(item, new_namespace)


def generate_one(ambf_ws: Path, arm: dict, out_dir: Path) -> Path:
    src_path = stock_model_path(ambf_ws, arm["source_model"])
    with open(src_path) as f:
        doc = yaml.safe_load(f)

    src_dir = src_path.parent
    for key in ("high resolution path", "low resolution path"):
        rel = doc.get(key)
        if rel:
            doc[key] = str((src_dir / rel).resolve()) + "/"

    overwrite_namespaces(doc, arm["namespace"])

    offset = arm.get("world_offset", {})
    dx, dy, dz = offset.get("x", 0.0), offset.get("y", 0.0), offset.get("z", 0.0)
    for body_key in doc.get("bodies", []):
        body = doc.get(body_key)
        if not isinstance(body, dict):
            continue
        pos = body.get("location", {}).get("position")
        if pos:
            pos["x"] = pos.get("x", 0.0) + dx
            pos["y"] = pos.get("y", 0.0) + dy
            pos["z"] = pos.get("z", 0.0) + dz

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arm['name']}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-config", default=str(REPO_ROOT / "config" / "arms.yaml"))
    parser.add_argument("--out-dir", default=str(GENERATED_DIR))
    args = parser.parse_args()

    ambf_ws = get_ambf_ws()
    with open(args.arms_config) as f:
        arms = yaml.safe_load(f)["arms"]

    out_dir = Path(args.out_dir)
    generated = []
    for arm in arms:
        out_path = generate_one(ambf_ws, arm, out_dir)
        print(f"Generated {out_path}", file=sys.stderr)
        generated.append(str(out_path))

    print(",".join(generated))


if __name__ == "__main__":
    main()
