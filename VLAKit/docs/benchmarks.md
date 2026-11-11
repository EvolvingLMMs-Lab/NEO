# Simulation benchmarks

`neo_vla.bench` scores a step on LIBERO, LIBERO-plus, LIBERO-PRO and RoboTwin
2.0. The policy runs as [neo-vla-serve](inference.md); each simulator runs in an
environment of its own (their pins conflict) and asks the server for actions
over HTTP. The client side needs numpy and the simulator, not torch.

Everything a benchmark must know about the step is in the step's `eval.json`:
the camera slots and their calibration, the frame each arm's actions come back
in, the state readings and action columns the step uses, the gripper's raw open
and closed readings, the centimetres and radians per unit of each benchmark's
own action, and the protocol: suites, step budgets, episodes, the executed
horizon and the seeds. The released
[`stage_libero`](https://huggingface.co/weepiess2383/NEO-VLA/tree/main/stage_libero)
carries the LIBERO family's and
[`stage_rtw`](https://huggingface.co/weepiess2383/NEO-VLA/tree/main/stage_rtw)
RoboTwin's.

| Benchmark | Tasks × episodes | Horizon | Notes |
|---|---|---|---|
| LIBERO | 4 suites × 10 tasks × 50 | 16 | 220 / 280 / 300 / 520 steps for spatial / object / goal / 10 |
| LIBERO-plus | 10,030 tasks × 1 | 20 | perturbation tag stripped from the instruction |
| LIBERO-PRO | 20 suites × 10 tasks × 50 | 16 | instruction from the task file name |
| RoboTwin 2.0 clean | 50 tasks × 100 | 30 | `demo_clean`, seen instructions |
| RoboTwin 2.0 randomized | 50 tasks × 100 | 30 | `demo_randomized`, seen instructions |

Every benchmark uses simulator seed 0 and seeded sampling noise (base 0), so a
rerun of the same step reproduces the same episodes. Each LIBERO episode first
holds still for 10 steps while the scene settles; those are not counted in the
budget.

## Serve the step

```bash
export NEO_VLA_BACKBONE=/path/to/NEO-VLA/stage2_backbone
python -m neo_vla.bench.spec --checkpoint /path/to/NEO-VLA/stage_libero > server.json
neo-vla-serve --config server.json --port 8000
```

The config maps camera slots, calibration, camera poses and arm states to the
names the bench clients send, and selects the weights the step was scored with.
The service answers one request at a time.

## LIBERO

Install into an environment of its own: `mujoco==3.2.3`, `robosuite==1.4.1`,
`numpy==2.2.6`, and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
checkout with `pip install --no-deps -e .`. Other mujoco and robosuite versions
change the score. Point `LIBERO_CONFIG_PATH` at the directory holding LIBERO's
`config.yaml`, and render headless with `MUJOCO_GL=egl` (or `osmesa` on CPU).

```bash
python -m neo_vla.bench.libero --benchmark libero \
    --checkpoint /path/to/NEO-VLA/stage_libero --server http://127.0.0.1:8000 \
    --libero-root /path/to/LIBERO --out results/libero
```

The arm is model slot 0; the scene camera fills camera slot 0 and the wrist
camera slot 1, both rendered at 256 × 256 and turned upright. Both camera
poses are read from the simulator at every step, and actions come back in the
wrist camera's frame; the client turns them into the base frame with that
camera's live pose and divides by the suite's unit factors. Only the
end-effector increments and the gripper are executed.

## LIBERO-plus

The same client in an environment with the
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) checkout:

```bash
python -m neo_vla.bench.libero --benchmark libero_plus \
    --checkpoint /path/to/NEO-VLA/stage_libero --server http://127.0.0.1:8000 \
    --libero-root /path/to/LIBERO-plus --out results/libero_plus
```

LIBERO-plus builds each task's language from its file name, so the variant's
identifier ends the sentence (`... in the basket view 0 0 100 2 4`). It is
stripped before the instruction is sent, and the run refuses to start unless
the strip changes the instructions and every stripped name is a task the suite
holds.

## LIBERO-PRO

The [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) fork (evaluated at
`eafdb809`) and its released task files:

```bash
python -m neo_vla.bench.libero --benchmark libero_pro \
    --checkpoint /path/to/NEO-VLA/stage_libero --server http://127.0.0.1:8000 \
    --libero-root /path/to/LIBERO-PRO --pro-data /path/to/LIBERO-PRO-data \
    --out results/libero_pro
```

`--pro-data` holds the fork's `bddl_files` and `init_files`; they are linked
into `results/libero_pro/staged`, with the config LIBERO reads.

## RoboTwin 2.0

The [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) checkout in its
own environment, with CuRobo and a Vulkan driver, as its install guide sets up.
RoboTwin's `script/eval_policy.py` builds the scenes, draws the seeds and
judges success; `robotwin_run` installs a plug-in for it as `policy/neo_policy`
and runs the tasks one at a time:

```bash
python -m neo_vla.bench.robotwin_run --protocol clean \
    --checkpoint /path/to/NEO-VLA/stage_rtw --server http://127.0.0.1:8000 \
    --robotwin-root /path/to/RoboTwin --robotwin-python /path/to/robotwin/bin/python \
    --results results/robotwin
```

`--protocol randomized` runs the randomized scenes. The left arm is model slot 0
and the right slot 1; the head camera fills camera slot 0 and the left and
right wrist cameras slots 1 and 2, at 320 × 240. Each arm's actions come back in
its own wrist camera's frame and are composed, row by row, onto that arm's
end-effector pose as RoboTwin `ee` targets. Rendering is ray-traced by default,
as the protocol specifies; `--renderer cpu --mesa-root DIR` uses a software
Vulkan driver.

## Results

```text
results/libero/
  settings.json                     the protocol the run used
  libero_spatial/task_0000.json     one task: every episode's outcome and length
  summary.json                      success rate per suite and their mean
results/robotwin/clean/
  run.json  <task>.json  tasks/<task>/  summary.json
```

A task with a result file is skipped, so rerunning the same command resumes.
`--suites`, `--tasks`, `--episodes` and `--horizon` (LIBERO family) or `--task`
(RoboTwin) run a subset; the settings file records what was run.
