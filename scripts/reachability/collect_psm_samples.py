#!/usr/bin/env python3
"""Record PSM joint configurations and the resulting tool-link poses from a
running ambf_simulator. NOT used by the (pure-math) reachability analysis;
kept for comparing the dVRK DH model against the AMBF PSM when the chosen
placement is tested in simulation.

Two kinds of samples are recorded:
  * calibration - the all-zero pose, then each joint on its own at +/-delta
                  (used to extract the joint screw axes)
  * random      - every joint at a random value (held out, used only to
                  validate the fitted model)

The jaws are always kept closed: gripper1 = jaw, gripper2 = -jaw (verified
in sim that this gives both jaws the same orientation).

Each sample streams the command for --settle-time so neither the client
nor the server watchdog releases the joints, then records the measured
joint positions and the world poses of the base and tool links.

Run the simulator WITHOUT the bracket so the tool can't hit it:
    ./scripts/launch_sim.sh --no-scene          # terminal 1
    source scripts/env.sh                       # terminal 2
    python3 scripts/reachability/collect_psm_samples.py --arm psm1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ambf_client import Client

REPO_ROOT = Path(__file__).resolve().parents[2]

# Model joint order used throughout scripts/reachability:
#   yaw, pitch, insertion, roll, wrist pitch, jaw (closed jaws)
JOINTS = ["baselink-yawlink", "yawlink-pitchbacklink", "pitchendlink-maininsertionlink",
          "maininsertionlink-toolrolllink", "toolrolllink-toolpitchlink",
          "toolpitchlink-toolgripper1link"]
JAW2 = "toolpitchlink-toolgripper2link"
# Random-sample ranges, inside the ADF joint limits with margin.
RANGES = [(-1.2, 1.2), (-0.5, 0.6), (0.02, 0.22), (-2.5, 2.5), (-1.3, 1.3), (-1.2, 1.2)]
CAL_DELTA = [0.4, 0.4, 0.15, 0.8, 0.6, 0.6]
LINKS = ["baselink", "maininsertionlink", "toolrolllink", "toolpitchlink",
         "toolgripper1link", "toolgripper2link"]


def pose_of(handle):
    p, q = handle.get_pos(), handle.get_rot()
    return {"pos": [p.x, p.y, p.z], "quat_xyzw": [q.x, q.y, q.z, q.w]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="psm1")
    ap.add_argument("-n", "--num-random", type=int, default=50)
    ap.add_argument("--settle-time", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "psm_fk_samples.json"))
    args = ap.parse_args()

    client = Client("psm_sample_collector")
    client.connect()
    time.sleep(0.5)
    handles = {}
    for link in LINKS:
        h = client.get_obj_handle(f"{args.arm}/{link}")
        if h is None:
            sys.exit(f"ERROR: {args.arm}/{link} not found - is the simulator running?")
        handles[link] = h
    base = handles["baselink"]
    t0 = time.time()
    while not base.get_joint_names() and time.time() - t0 < 5:
        time.sleep(0.2)
    missing = set(JOINTS + [JAW2]) - set(base.get_joint_names())
    if missing:
        sys.exit(f"ERROR: joints not reported by {args.arm}: {sorted(missing)}")

    plan = [("calibration", [0.0] * 6)]
    for i, d in enumerate(CAL_DELTA):
        for s in (+1, -1):
            q = [0.0] * 6
            q[i] = s * d if i != 2 else d * (1 if s > 0 else 0.5)  # insertion can't go < 0
            plan.append(("calibration", q))
    rng = np.random.default_rng(args.seed)
    plan += [("random", [float(rng.uniform(*r)) for r in RANGES]) for _ in range(args.num_random)]

    samples = []
    for k, (kind, q) in enumerate(plan):
        cmd = dict(zip(JOINTS, q))
        cmd[JAW2] = -q[5]
        for _ in range(max(1, int(args.settle_time / 0.05))):
            for n, v in cmd.items():
                base.set_joint_pos(n, v)
            time.sleep(0.05)
        meas = [base.get_joint_pos(n) for n in JOINTS]
        jaw2 = base.get_joint_pos(JAW2)
        err = max(abs(a - b) for a, b in zip(meas, q))
        samples.append({"kind": kind, "q_cmd": q, "q_meas": meas, "jaw2_meas": jaw2,
                        "poses": {link: pose_of(h) for link, h in handles.items()}})
        print(f"[{k + 1}/{len(plan)}] {kind:11s} max |cmd-meas| = {err:.4f}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "joints": JOINTS, "jaw2_joint": JAW2,
                               "samples": samples}, indent=1))
    print(f"Wrote {len(samples)} samples to {out}", file=sys.stderr)
    client.clean_up()


if __name__ == "__main__":
    main()
