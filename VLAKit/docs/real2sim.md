# Real-to-sim episodes

`neo_vla.real2sim` turns a measured real robot set-up into simulated demonstration episodes that
you can add to a post-training corpus. You describe your robot (URDF and meshes), your cameras
(calibrated intrinsics and poses), the table and objects (measured sizes, declared masses and
friction) and the task (instruction, controller and a success contract). The pipeline builds a
[SAPIEN](https://sapien.ucsd.edu/) scene from that description, runs a scripted controller in
bounded rollouts, checks every rollout from the files it leaves behind and exports the episodes that
pass in the same raw layout a real recording would have.

Claude Code can do the authoring work: propose the scene and task specifications from your
measurements, calibration files and photos, write a controller plugin when the built-in controllers
do not fit, and explain failed rollouts. It runs through your own Claude Code login, it never writes
files itself, and nothing it produces is used before you have reviewed it. Simulation, recording and
acceptance are ordinary deterministic code; no language model decides whether an episode passed.

The pipeline itself needs only the Python standard library. The simulator runs in its own
environment as a separate process.

## What is supported

| | Supported |
|---|---|
| Robot | one fixed-base arm described by a URDF (relative mesh paths), revolute or prismatic arm joints with limits |
| Gripper | parallel gripper on prismatic finger joints; aperture = offset + Σ scale × joint position |
| Objects | rigid boxes, cylinders, spheres or convex meshes (single hull or pre-decomposed parts) on a table |
| Cameras | pinhole cameras fixed in the world or mounted on a robot link, optional Brown–Conrady distortion |
| Controllers | `joint_waypoints` (smooth moves through reviewed joint configurations, for example a replayed real demonstration), `pick_place` (IK-based grasp, lift, carry, place, release, retreat), or a reviewed Python plugin |
| Success predicates | position in a box (world or relative to an object), axis tilt, contact and no-contact (object, table or robot), speed limits, gripper open, displacement limit, phase |
| Recording | 30 Hz rows, physics at any multiple of 30 Hz, causal image selection, per-row pose and gripper records |
| Rendering | SAPIEN Vulkan on a GPU, or on the CPU through Mesa lavapipe |

A second route, `can_in_tape_reference`, reproduces the can-in-tape simulation of the NEO-VLA
real-robot work on the original task runtime with its original checks. It needs that runtime and its
inputs, which are not distributed here; see [the reference route](#can-in-tape-reference-route).

Not supported or not claimed: reconstructing geometry from a single photograph, deformable or
articulated objects, mobile or bimanual robots, sim-to-real transfer, and any statement about how a
policy trained on these episodes behaves on a real robot.

## Stages

```
check -> plan -> review -> build -> qualify -> collect -> verify -> export [-> admit]
```

| Stage | What it does | Passing means | It does not establish |
|---|---|---|---|
| `check` | validates project.json and every referenced file; lists structured requests for missing inputs | inputs parse, provenance rules hold | anything physical |
| `plan` | writes candidate scene/task/plugin specifications (agent jobs, or your own files) | candidates validate | that the candidates are right |
| `review` | lists every estimate, design choice, image-derived value and agent note; records your decisions | every item decided, none rejected | the correctness of the decisions |
| `build` | hashes the URDF, meshes and reviewed specifications, checks controller limits, samples layouts, stages the runner and probes the simulator | configuration is complete and the simulator renders a frame | physics or rendering of your task (both UNVERIFIED) |
| `qualify` | runs and verifies a few rollouts from a separate pool, saves the first frames | every qualification rollout passes every check | visual similarity to your real cameras |
| `collect` | runs bounded rollout attempts, each in its own process with a deadline | nothing by itself | acceptance |
| `verify` | repeats every check on the persisted files of every attempt | each target episode has exactly one passing attempt | training readiness |
| `export` | hard-links passing episodes into split groups and checks them again | see export levels below | training admission |
| `admit` | records your evidence that the converted corpus passed your training-side checks | you attested it | anything the pipeline ran itself |

Every stage run is a new, sequence-numbered directory under `<workdir>/stages/<stage>/runs/`. It
starts by writing `STARTED.json` with the fingerprint of its exact inputs (file hashes, upstream
receipts, configuration) and ends with a `RECEIPT.json`: the status, the files it vouches for with
their hashes, and what it does not establish. A stage is up to date only when its newest run for the
current inputs passed and its files still hash as recorded. Editing any input invalidates every
later stage. If the newest run for the current inputs failed, the stage is failed; if it was
interrupted, the next command records it as `INCOMPLETE` in place (its files stay as evidence) and
the stage is held until it runs again. An earlier pass for the same inputs is never reused. Paths
are always resolved inside the workdir, whatever directory you run the commands from. One process
writes a workdir at a time (`<workdir>/LOCK`); if a process died, delete the lock after checking
that it is gone.

Exit codes: `0` passed (export: `CONVERSION_READY`), `2` held, incomplete, partial or
`RAW_SCHEMA_VALID`, `3` failed, `4` refused (a precondition is not met).

## Setup

### 1. Code

```bash
git clone https://github.com/EvolvingLMMs-Lab/NEO.git
cd NEO/VLAKit
python3 -m neo_vla.real2sim --help      # standard library only; no installation needed
```

`pip install -e .` also installs a `neo-vla-real2sim` command, but pulls in the training
dependencies.

### 2. Simulator environment

Use a separate environment for the simulator (Linux, Python 3.10):

```bash
python3.10 -m venv ~/envs/real2sim-sim
~/envs/real2sim-sim/bin/pip install sapien==3.0.0b1 numpy==1.26.4 pillow==11.3.0 six==1.17.0
~/envs/real2sim-sim/bin/pip install opencv-python-headless==4.11.0.86   # only for lens distortion
```

These are the versions the pipeline was developed against; other versions have not been checked.
The episodes accepted on the can-in-tape reference were produced with SAPIEN 3.0.0b1, NumPy 1.26 and
Python 3.10 and rendered on the CPU.

`six` is listed because SAPIEN 3.0.0b1's bundled URDF parser imports it, and SAPIEN does not pull
it in. The import happens only when the first URDF loader is created, which is why the probe creates
one. OpenCV is only needed for cameras with lens distortion.

For CPU rendering install Mesa's lavapipe Vulkan driver and
point `simulator.env.VK_ICD_FILENAMES` at its `lvp_icd.x86_64.json`. Some SAPIEN builds need a
newer lavapipe than the one a distribution ships; `simulator.preload_libraries` loads extra shared
libraries before SAPIEN is imported if your set-up needs that. On a GPU machine set
`simulator.renderer` to `gpu`.

After the project exists (step 4), check the environment with:

```bash
python3 -m neo_vla.real2sim probe --project my_task/project.json
```

The probe imports SAPIEN the way a rollout does, creates a URDF loader, and renders one frame of a
lit table. It says nothing about your task.

The probe (and the probe inside `build`) passes only when **all** of these hold:

- its process started and exited with status 0 on its own, within its deadline and output limit;
- both output streams were captured to the end;
- no process of its group was left behind — observable on Linux through `/proc` and
  `waitid(WNOWAIT)`; where it cannot be observed, the probe fails;
- its own `PROBE.json` is readable and says `ok`.

A success report never makes up for a process that failed or did not finish cleanly.

The simulator interpreter runs the staged runner in isolated mode (`python -I`), so nothing from
the current directory, `PYTHONPATH` or user site-packages is importable. Install SAPIEN, NumPy,
Pillow and six into that interpreter's own environment. The runner loads its sibling `checks.py` by
explicit path, and `runner.py --help` names that file and its sha256.

### 3. Claude Code login

Install Claude Code by following its official instructions, then log in once with your own account:

```bash
claude auth login
claude auth status
python3 -m neo_vla.real2sim auth      # runs `claude --version` and `claude auth status`
```

The pipeline never reads, copies or stores credentials and has no login of its own. It runs your
`claude` executable with the model and effort you pin in `project.json`, without any tools:

```
claude --print --input-format stream-json --output-format stream-json --verbose
       --model <model> --effort <effort> --tools "" --safe-mode
       --strict-mcp-config --mcp-config '{"mcpServers":{}}' --disable-slash-commands
       --no-session-persistence --permission-prompts none
```

`--tools ""` removes every built-in tool, `--safe-mode` switches off CLAUDE.md files, skills,
installed plugins, hooks, MCP servers and other customizations while keeping your normal login and
model selection (settings an administrator manages by policy still apply), and the empty strict MCP
configuration excludes any other configured server. (Bare mode is not used: it skips subscription
logins.) The prompt and the job's declared inputs are sent as a
single user message on stdin: text files are quoted, images (PNG, JPEG, GIF, WebP) are attached as
base64 blocks. Before anything is sent, each input must be a regular file (no symlinks), within the
size limits (1 MiB per text file and 4 MiB of text per job; 20 images of at most 3.75 MB each), and
unchanged since `check` recorded its hash. The job fails unless the CLI's own session report lists no
tools and no MCP servers, and it fails if the session reports any tool request. This bounds what the
model can see and do inside its session; it is not an operating-system sandbox, and the CLI process
runs with your user's permissions.

Each job also has a deadline (`agent.timeout_s`) and a cap on the output it may write. The serving
model the CLI reports is recorded and must equal the pinned model; any other model, a missing
report, an error result or a timeout fails the job. There is no fallback to another model or
account. A stage makes at most `agent.max_attempts` (1 to 3) attempts per job and never repairs its
own output beyond that. Set `agent.enabled` to `false` to write every specification yourself.

### 4. Project

```bash
python3 -m neo_vla.real2sim init my_task
```

This writes `my_task/project.json`. Put your files next to it (any layout; paths in the project are
relative to its directory) and fill in the inputs.

## Inputs

`project.json` (`neo-vla-real2sim-project/1`) holds the instruction, the inputs, where the reviewed
specifications live, the simulator, the collection bounds, the split policy and the agent settings.

**Provenance.** Every physical quantity in a specification is written as
`{"value": ..., "unit": ..., "provenance": ..., "source": ...}`:

| Provenance | Meaning | Rules |
|---|---|---|
| `measured` | measured directly or calibrated | cite `measurement:<id>` or `calibration:<id>#<json.path>`; machine-readable sources are compared exactly |
| `derived` | computed from measurements, CAD or the URDF | say how in `source` |
| `estimate` | an assumed physical value | reviewed |
| `chosen` | a design decision | reviewed |
| `unverified` | unknown origin | reviewed; blocks fields that need measurements |
| `inferred_from_image` | read off a photograph | reviewed; never counts as measured geometry; never allowed for mass, friction, restitution or drive gains |

Geometry and calibration (table size and pose, robot base pose, TCP offset, gripper stroke and
offset, object sizes, camera intrinsics and poses) must be measured or derived. A missing or
unusable value becomes a request such as:

```json
{"id": "request-3", "field": "scene.objects[0].shape.size_m", "problem": "images alone are not measured geometry",
 "blocks": ["build"], "how_to_supply": "measure it, or derive it from measurements, and cite the source"}
```

**Images** (`inputs.images`) need a `use` of `appearance_reference`, `layout_reference` or
`scene_context`. They help the agent describe the scene and help you review it; they are never a
measurement. A measurement taken from an image (photogrammetry with a known scale) must be declared
`derived` and name its `scale_reference`.

**Measurements** (`inputs.measurements`) carry an id, value, unit, method, who measured and when.
**Calibration files** (`inputs.calibration`) are hashed and can be cited field by field when they are
JSON. **The robot** (`inputs.robot`) names the URDF and records its source and license; mesh paths
inside the URDF must be relative to it (no `package://`). **Assets** (`inputs.assets`) record source
and license for every mesh or texture you add.

The specification formats are described in full by the guides the agent receives, which are also
available from Python (`neo_vla.real2sim.templates.SCENE_GUIDE`, `TASK_GUIDE`, `PLUGIN_GUIDE`).
In short: the scene (`neo-vla-real2sim-scene/1`) declares physics rates, table, robot (URDF, base
pose, arm and gripper joints, TCP, drives, optional first-order follower fitted from real joint logs),
objects, cameras, lighting and layout randomization; the task (`neo-vla-real2sim-task/1`) declares the
instruction, episode budget, controller, success contract and integrity tolerances.

## Walkthrough

```bash
P=my_task/project.json
python3 -m neo_vla.real2sim check   --project $P            # lists errors and requests
python3 -m neo_vla.real2sim plan    --project $P            # candidates under work/stages/plan/runs/<run>/candidates
python3 -m neo_vla.real2sim review  --project $P --promote scene task
#   edit my_task/specs/scene.json and task.json until `check` is clean
python3 -m neo_vla.real2sim review  --project $P            # prints the items that need a decision
python3 -m neo_vla.real2sim review  --project $P --reviewer "Your Name" --accept-all
python3 -m neo_vla.real2sim build   --project $P
python3 -m neo_vla.real2sim qualify --project $P            # inspect work/stages/qualify/runs/<run>/previews
python3 -m neo_vla.real2sim collect --project $P --indices 0-9 --workers 2
python3 -m neo_vla.real2sim collect --project $P            # the rest; finished episodes are skipped
python3 -m neo_vla.real2sim verify  --project $P
python3 -m neo_vla.real2sim export  --project $P
python3 -m neo_vla.real2sim status  --project $P
```

`--accept ID ...` and `--reject ID ...` decide items one by one. A decision file
(`reviews/review-<fingerprint>.json`) covers exactly the current specifications and items; change
either and you review again. Promotion never overwrites an existing specification.

## Collection bounds

`collection.episodes` layouts are sampled deterministically from `collection.seed`; qualification
uses a separate pool that is never exported. Each episode gets at most
`collection.max_attempts_per_episode` attempts (1 to 3) across all `collect` invocations, including
attempts that were interrupted (they are recorded as `INCOMPLETE` in place and fail verification).
Every attempt is a separate process in its own process group with the deadline
`collection.attempt_timeout_s` and a 64 MiB cap on each of its stdout and stderr; reaching either
bound terminates the group. The episode budget (`task.episode.max_duration_s`) bounds the number of
control steps and frames. A process exiting with status 0 is never taken as success.

Child processes are cleaned up as follows. On Linux the pipeline notices the child's exit without
reaping it, which keeps its process-group ID reserved, lists what is left of the group from `/proc`,
sends SIGTERM and, after a grace period, SIGKILL, and only then reaps the child; this also covers
descendants left behind by a child that exited normally. On macOS the group is signalled the same
way on a timeout or output limit, but descendants of a child that exited normally are not.
Processes that move themselves out of the group (`setsid`, `setpgid`) are outside this control on
any platform. Windows is not supported.

## Outputs

```
work/
  stages/<stage>/runs/r000012-<utc>-<suffix>/STARTED.json RECEIPT.json
  stages/build/runs/<run>/BUILD.json, layouts.json, runtime/ (staged runner, checks, plugin)
  stages/plan/runs/<run>/agent/<job>-attempt1/STDIN.jsonl PROMPT.txt stdout.jsonl JOB.json
  stages/collect/inputs-<fingerprint>/attempts/ep00007_a1/
      SPEC.json IDENTITY.json RESULT.json stdout.txt stderr.txt
      out/RUNNER_RESULT.json out/trace.npz out/trace.json out/episode/ (or out/failures/)
  stages/verify/runs/<run>/AUDIT.json
  stages/export/runs/<run>/EXPORT_MANIFEST.json CHECKS.json dataset/<group>/episode<N>/
```

A raw episode holds `arrays.npz` (`states` measured TCP in the robot base frame as xyz cm,
quaternion xyzw and aperture / stroke; `actions` next-minus-current commanded TCP as dxyz cm,
spatial rotation vector and next commanded aperture / stroke; `valid`; `times`; `intrinsics`),
`camera_geometry.npz`, `joint_states.npz`, `manifest.json`, per-row pose and gripper records,
`camera/color/<camera>/<time>.png` with `sync.txt` and `config.json`, and `calibration/` with the
reviewed specifications. A row uses the latest image captured at or before its time. The trace
(`trace.npz`) records every physics tick: object poses and velocities, the end-effector pose, joint
positions and drive targets, gripper aperture and command, and contact flags.

**What verify checks for each attempt:** the attempt records belong together and the rollout
process ended cleanly, by the same rule as the probe (status 0 on its own within its deadline and
output limit, output captured to the end, nothing left running in its group). The driver records
an unclean exit as `UNCLEAN_EXIT`, even when the runner wrote a success result. Verify also checks
that the spec, layout, runner, checks and plugin bytes are the ones the build staged, and that the
trace has an exact tick grid, finite values and proper rigid transforms; no object moves more than
its own reported velocity explains (`max_teleport_residual_*`) and nothing drifts while settling;
the success contract holds for the final `hold_s`; re-running the audit reproduces the runner's
result exactly; tick 0 matches the layout; the episode passes the schema, numeric, causal, label and
recording checks; the rendered camera intrinsics match the calibration. Episodes that share a trace
(the same file, or the same arrays saved at a different time) or a layout are all rejected.

**Export levels:**

| Status | Meaning |
|---|---|
| `HELD` | no exportable episode, a raw check failed, or the route is synthetic; reasons are listed |
| `RAW_SCHEMA_VALID` | every exported episode passed the schema and numeric checks, but a causal, label or recording check, the split policy or the target count did not |
| `CONVERSION_READY` | all checks passed, every episode belongs to exactly one declared split group, no layout is shared, and the full target count is present (or you passed `--allow-partial`) |
| `TRAINING_ADMITTED` | recorded only by `admit`, with the hash of the evidence your own corpus build produced |

A synthetic test project declares `"fixture": {"synthetic": true}` in `project.json`; `fixture`
blocks with any other value are refused at intake. The flag is copied into every episode manifest
(`simulation.synthetic_fixture`), verify fails an episode that lost it, and export and `admit`
refuse such a project.

`output.split_policy.groups` maps group names to `{"indices": "0-15"}` or `{"splits": [...]}`.
This repository does not include corpus, cache or normalizer builders (see the
[training guide](training.md)); convert the exported episodes with your own tooling, using the
conversion settings recorded in `EXPORT_MANIFEST.json`, then record the result with
`admit --export <run> --evidence <report> --by <name> --statement <text>`.

## Agent jobs

| Job | Inputs (sent on stdin) | Returns | Accepted when |
|---|---|---|---|
| scene | project.json, images, calibration files, URDF and its joint/link summary, scene guide | `{"scene", "observations", "open_questions"}` | the scene validates: measured values cite existing measurements or calibration fields and match them, image-derived values are tagged |
| task | project.json, scene, images, task guide | `{"task", "rationale", "open_questions"}` | the task validates against the scene |
| controller | scene, task, plugin guide and skeleton | `{"files": [{"name", "content"}], "notes"}` | one file with the declared name that parses and sets `API_VERSION = 1` |
| diagnose | an attempt's records, logs and specification | causes with evidence file names, suggested checks | every cited file was among its inputs |

The pipeline writes accepted artifacts to declared candidate paths only. Open questions, notes and
static findings about plugin code become review items.

## Controller plugins

A plugin is one Python file:

```python
API_VERSION = 1


class Controller:
    def __init__(self, context):
        # params, task, scene, layout, arm_joints, joint_limits, init_arm_q, init_gripper_m,
        # gripper_stroke_m, base_pose_W, control_hz, robot_max_joint_speed_rad_s, ik, fk
        ...

    def act(self, obs):
        # obs: step, t_s, arm_q, gripper_m, tcp_pose_B, base_pose_W, objects, command_arm_q, ...
        return {"arm_q": [...], "gripper_m": 0.04, "phase": "carry", "done": False}
```

It sees observations only and has no handle on the simulator. The runner rejects non-finite
commands, joint targets outside the URDF limits, steps faster than `max_joint_speed_rad_s / 30` and
apertures outside the stroke; any rejection fails the episode. `ik(target_tcp_in_base, seed)` and
`fk(arm_q)` use SAPIEN's Pinocchio model of your URDF. A plugin is ordinary Python and runs with your
user's permissions inside the rollout process, so review it like any code you run: the build only
stages the exact bytes you reviewed (a regular file of at most 1 MiB, never through a symlink), and
its episodes count only if the independent checks pass.

## Adding a route

A route is a subclass of `neo_vla.real2sim.pipeline.Route` that implements `intake`, `review`,
`build_inputs`, `build`, `qualify`, `collect`, `targets` and `verify`, plus `disclosures` and
`conversion_settings` for the export. Stages return `outcome(status, reasons, outputs,
not_established)`; collection writes one directory per attempt finished by a `RESULT.json`; verify
writes `AUDIT.json` through `write_audit`. A route defines its own acceptance contract. Results of one
route or scene never stand in for another.

## Can-in-tape reference route

This route reproduces the simulated "stand the can in the tape roll" task of the NEO-VLA real-robot
experiments: an AgileX PiPER arm with an AgileX Pika gripper, a 57 × 146 mm can and a
107/76 × 48 mm tape roll (measured), the calibrated rig of that work, 200 in-domain and 200
out-of-distribution layouts. It drives the original `task3_collect.py` command line unchanged and
verifies every episode with the original success contract (re-run on the persisted trace, which must
reproduce the stored audit exactly) and the original per-episode checks of geometry, camera
extrinsics and intrinsics, appearance application, the no-teleport guard and visual asset identity.
Build refuses a runtime whose acceptance-contract sources differ from the reviewed versions.

It needs, supplied by you: the can-in-tape runtime workspace (the sim_twin base with its RoboTwin
overlay, the task modules and a reviewed native controller), a RoboTwin checkout and SAPIEN
3.0.0b1 environment set up as that workspace's README describes, a lavapipe runtime, the actuator
profile fitted from the real recordings, the photo-derived visual assets (their source photos are
pinned by hash) and the episode-1 execution plan. These come from the authors under their own terms
and are not part of this repository. Use `init --route can_in_tape_reference`, keep the workdir
inside the runtime workspace (the runtime writes only there), and keep the default seed to reproduce
the reference layouts.

Disclosed with every export of this route: the can and tape masses, friction and restitution and the
appearance variations are estimates; the scripted expert was revised after observing failures on
layouts of the same manifest, so accepted episodes are success-conditioned and are not an unbiased
benchmark of one controller; CPU rendering through lavapipe is the validated path and GPU runs have
not produced accepted reference episodes; passing these checks is simulation acceptance, not training
admission.

## Recovering from failures

- **Requests from `check`:** supply the named input (`how_to_supply` says how) and run `check` again.
- **A failed plan job:** read `stages/plan/runs/<run>/agent/<job>-attempt<n>/JOB.json` (reasons,
  reported models, the session's reported tools, CLI exit code and stderr); `STDIN.jsonl` is exactly
  what was sent. Fix the input or the pinned model and run `plan` again, or write the specification
  yourself.
- **An interrupted command:** the next command records the unfinished run or attempt as
  `INCOMPLETE`; run the stage again.
- **Build failures:** the receipt names the URDF, mesh, limit, layout or probe problem.
  `probe.stderr.txt` holds the simulator output.
- **A failed rollout:** `out/RUNNER_RESULT.json` has the reason and traceback, `out/trace.npz` the
  physics up to the failure, `out/failures/raw_partial/` the partial recording, and verify's
  `AUDIT.json` the failing checks. `diagnose --attempt <attempt dir>` asks the agent for an advisory
  explanation. Change the reviewed inputs, review again and rebuild; the old attempts stay in their
  own collection namespace.
- **An episode out of attempts:** it stays failed for these inputs. Raising
  `max_attempts_per_episode` changes the inputs and starts a new collection namespace.
- **A stale lock:** check that no pipeline process is running, then delete `<workdir>/LOCK`.

## Native integration check

The configurable route has completed one qualification rollout and three episodes in SAPIEN
3.0.0b1 on the CPU renderer. An original two-joint primitive arm followed joint waypoints while
a cube stayed on a table. Pose readback, trace physics, two camera recordings, process cleanup,
provenance and duplicate checks passed. The episodes are marked as synthetic test fixtures and
are blocked from export and training admission. This checks the simulator and recording path;
it does not establish manipulation performance or agreement with a real scene.

That run used Mesa llvmpipe (LLVM 17) and a task-local SAPIEN renderer compatibility patch,
loaded through `simulator.preload_libraries`. Those binaries are not distributed here. An
unmodified pip-only CPU installation and GPU execution of this configurable route remain
unverified; provide a compatible Vulkan runtime and pass the probe on your machine.

## What is not measured

- Visual similarity between rendered and real camera images. Compare the qualification previews
  with your own frames.
- The accuracy of declared estimates (mass, friction, restitution, drive gains, follower lag).
- Sim-to-real transfer, policy performance and deployment reliability.
- Pixel content of exported frames beyond presence, PNG header and size.

## Third-party components

Nothing third-party is bundled with this pipeline. SAPIEN, NumPy, Pillow, six, OpenCV, Mesa and
Claude Code are installed by you under their own licenses. Your robot description, meshes, textures and
photos keep the terms you record for them in `project.json`. The can-in-tape reference runtime
relies on RoboTwin, SAPIEN and the AgileX Pika gripper model, each under its own terms; the Pika
model files carried no license when they were fetched for that work (October 2026), so they are not
redistributed here.
