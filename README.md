# Fine Forcep Manipulation

Simulation-side dev environment for a class project built around the IROS
2026 Forceps-Based Fine Manipulation (FFM) Challenge Task 1 (needle
threading). The official FFM challenge is hardware-only (no dVRK/sim
requirement), but without hardware access this project uses AMBF +
simulated dVRK PSMs/MTMs as a stand-in to develop and test motion/task
planning for a modified version of the needle-threading task before (if
ever) porting to real hardware.

This repo does **not** vendor AMBF. It expects a separate, already-built
AMBF ROS 2 workspace (https://github.com/WPI-AIM/ambf, branch `ambf-3.0`)
and points at it via an environment variable, so each team member can keep
their own workspace at whatever path they like.

## Prerequisites

- Ubuntu 24.04 + ROS 2 Jazzy (or adjust `ROS_DISTRO` below)
- A built AMBF workspace: clone `WPI-AIM/ambf` into `<ws>/src/ambf` and
  `colcon build` it. See the AMBF repo's own README for build deps.
- Python 3 with `pyyaml` (`pip install pyyaml` if not already present)

## Setup

```bash
cp .env.example .env
```

Edit `.env` and set `AMBF_WS` to the absolute path of your built
`ros_ambf_ws` (the workspace containing `src/ambf` and `install/`).
`.env` is gitignored — every teammate sets their own path and nothing
machine-specific gets committed.

## What's here

- `config/arms.yaml` — declares the 4 arms spawned for dev/testing: `psm1`,
  `psm2`, `mtm1`, `mtm2`. Each entry names the stock AMBF model it's based
  on (`psm` or `mtm`), a unique ROS namespace, and a world-space offset so
  the two copies of each model don't spawn on top of each other.
- `scripts/env.sh` — source this (`source scripts/env.sh`) to load `.env`
  and source ROS 2 + the AMBF workspace it points at. Every other script
  below assumes this has been sourced, or sources it itself.
- `scripts/generate_adfs.py` — reads `config/arms.yaml` and the stock
  `blender-psm.yaml` / `blender-mtm.yaml` ADFs from `$AMBF_WS`, and writes
  out `adf/generated/{psm1,psm2,mtm1,mtm2}.yaml`: copies with a unique
  namespace, an offset world position, and mesh paths resolved to absolute
  paths (since the generated files live outside the AMBF models tree).
  `adf/generated/` is gitignored and regenerated per machine.
- `scripts/launch_sim.sh` — regenerates the ADFs and launches
  `ambf_simulator` with all 4 arms loaded. Extra args are passed through to
  `ambf_simulator` (e.g. `-g 0` to run headless).
- `scripts/joint_sweep_test.py` — the joint-moving smoke test: connects to
  a running simulator, finds each of the 4 arms, and for every joint it
  reports, commands a small sweep (+delta, -delta, back to its starting
  position) and logs commanded vs. measured position. This only checks
  that comms and joint control are wired up correctly for all 4 arms — it
  is not a task or teleoperation script.

## Running the smoke test

Terminal 1 — launch the simulator (GUI on by default):

```bash
./scripts/launch_sim.sh
```

Terminal 2 — once the simulator window is up, run the sweep:

```bash
source scripts/env.sh
python3 scripts/joint_sweep_test.py
```

Useful flags on the test script:

- `--delta` — sweep amplitude in rad or m (default `0.05`, deliberately
  small/safe around each joint's current position)
- `--settle-time` — seconds to wait after each commanded move before
  reading back (default `0.5`)
- `--discovery-timeout` — seconds to wait for each arm to appear on the ROS
  graph (default `5.0`)

The script exits non-zero and lists which arm/joint failed if anything
doesn't respond.

### Known harmless noise

- `Watch Dog Expired, Resetting ... command` from `ambf_simulator` is
  AMBF's own joint-command watchdog firing between sweeps (expected since
  this test sends one command and waits, rather than streaming
  continuously like a real controller would) — not an error.
- A `Traceback` from `threading.py` may print after
  "All joints swept successfully..." on exit. That's a known thread-safety
  race in AMBF's own `ambf_client` (`Client.clean_up()` clears a dict that
  its background publisher thread may be mid-iteration over) firing during
  shutdown, in a daemon thread, after the script's real work and exit code
  are already done. Harmless; not something to fix in this repo.

## Next steps (not done yet)

- Teleoperation mapping: wire `mtm1 -> psm1` and `mtm2 -> psm2` Cartesian
  pose so real/simulated MTMs can drive the PSMs (the user eventually wants
  to use real MTM hardware as input into this sim).
- Modified Task 1 environment: needle holder + needle ADFs for the
  simulated version of needle threading, and the motion/task planning on
  top of it.
