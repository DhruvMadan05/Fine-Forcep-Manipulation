#!/usr/bin/env python3
"""Joint-sweep smoke test for psm1, psm2, mtm1, mtm2.

Connects to a running ambf_simulator (launched via launch_sim.sh), finds
each arm declared in config/arms.yaml, and for every joint it reports,
commands a small sweep around its current position (+delta, -delta, back
home), logging commanded vs. measured position. This only checks that
comms and joint control are wired up correctly - it is not a task script.

Usage:
    source scripts/env.sh
    python3 scripts/joint_sweep_test.py
"""
import argparse
import sys
import time
from pathlib import Path

import yaml
from ambf_client import Client

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_arms_config(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)["arms"]


def wait_for_object(client, short_name, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        handle = client.get_obj_handle(short_name)
        if handle is not None:
            return handle
        time.sleep(0.2)
    return None


def wait_for_joint_names(handle, timeout):
    """get_obj_handle() only just asked the simulator to start publishing
    joint names/positions; it takes a couple of publish cycles to show up."""
    deadline = time.time() + timeout
    joint_names = handle.get_joint_names()
    while not joint_names and time.time() < deadline:
        time.sleep(0.2)
        joint_names = handle.get_joint_names()
    return joint_names


def sweep_joint(handle, joint_name, delta, settle_time):
    home = handle.get_joint_pos(joint_name)
    home = 0.0 if home is None else home

    readings = {}
    for label, target in (("plus", home + delta), ("minus", home - delta), ("home", home)):
        handle.set_joint_pos(joint_name, target)
        time.sleep(settle_time)
        readings[label] = (target, handle.get_joint_pos(joint_name))
    return readings


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms-config", default=str(REPO_ROOT / "config" / "arms.yaml"))
    parser.add_argument("--delta", type=float, default=0.05,
                         help="Sweep amplitude in rad (revolute) or m (prismatic) around each joint's current position")
    parser.add_argument("--settle-time", type=float, default=0.5,
                         help="Seconds to wait after each commanded move before reading back")
    parser.add_argument("--discovery-timeout", type=float, default=5.0,
                         help="Seconds to wait for each arm's object to appear")
    args = parser.parse_args()

    arms = load_arms_config(Path(args.arms_config))

    client = Client("joint_sweep_test")
    client.connect()
    time.sleep(0.5)

    print("Objects visible to client:")
    for name in client.get_obj_names():
        print(" -", name)

    failures = []
    for arm in arms:
        root_short_name = arm["root_body"].replace(" ", "")
        short_name = f"{arm['name']}/{root_short_name}"
        print(f"\n=== {arm['name']} ({short_name}) ===")

        handle = wait_for_object(client, short_name, args.discovery_timeout)
        if handle is None:
            print(f"  FAILED to find object '{short_name}' - is ambf_simulator running with this arm loaded?")
            failures.append(arm["name"])
            continue

        joint_names = wait_for_joint_names(handle, args.discovery_timeout)
        if not joint_names:
            print("  No joints reported for this object.")
            failures.append(arm["name"])
            continue

        for joint_name in joint_names:
            readings = sweep_joint(handle, joint_name, args.delta, args.settle_time)
            plus_cmd, plus_meas = readings["plus"]
            minus_cmd, minus_meas = readings["minus"]
            home_cmd, home_meas = readings["home"]
            ok = all(meas is not None for _, meas in readings.values())
            status = "OK" if ok else "NO FEEDBACK"
            print(f"  {joint_name:30s} "
                  f"+{args.delta:.3f} -> cmd {plus_cmd:+.4f} meas {plus_meas}   "
                  f"-{args.delta:.3f} -> cmd {minus_cmd:+.4f} meas {minus_meas}   "
                  f"home -> cmd {home_cmd:+.4f} meas {home_meas}   [{status}]")
            if not ok:
                failures.append(f"{arm['name']}/{joint_name}")

    client.clean_up()
    client.ral.shutdown()

    if failures:
        print(f"\n{len(failures)} issue(s) found:")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)

    print("\nAll joints swept successfully on all arms.")


if __name__ == "__main__":
    main()
