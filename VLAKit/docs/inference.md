# Real-robot deployment

`neo_vla.serve` runs a post-trained policy next to a real robot. The policy
runs as an HTTP service on a GPU machine; the robot's control loop sends the
current camera frames, joint and end-effector readings and the task, and gets
back a chunk of actions in physical units, which it executes with its own
controller. The service knows nothing about the robot's drivers, so it can be
wired into any control stack that can make an HTTP call.

## The step to deploy

Deploy a step from [stage 3](training.md#stage-3-real-robot-adaptation):

```text
runs/posttrain_real/
  .hydra/config.yaml          the run config, found automatically
  checkpoints/step_000030000/
    state.pt                  model weights
    meta.json                 format, record stamps and prefix layout
    embodiments.json          the normalizer, datasets records and trained prompts
```

Keep the three step files together: actions are read back through the step's
own `embodiments.json`. The build reads the backbone geometry and tokenizer
from `NEO_VLA_BACKBONE`, as in training; its weights are replaced by the step's.

## Serve

```bash
export NEO_VLA_BACKBONE=/path/to/backbone
neo-vla-serve --config robot.json --host 0.0.0.0 --port 8000
```

The service answers `GET /health`, `GET /metadata` and `POST /predict` and runs
one prediction at a time. `robot.json` maps the names your client sends to the
model's inputs:

```json
{
  "checkpoint": "runs/posttrain_real/checkpoints/step_000030000",
  "device": "cuda",
  "embodiment": "my_robot_key",
  "directory": "my_robot_pick_place",
  "arm_slots": [0],
  "frame_ref": "camera:1",
  "camera_names": ["scene", "wrist", null],
  "intrinsics_fields": ["scene_intrinsics", "wrist_intrinsics", null],
  "extrinsics_fields": ["scene_pose", "wrist_pose", null],
  "state_fields": ["arm"],
  "output_fields": [
    {"name": "joint_targets", "width": 7, "unit": "rad", "mode": "absolute"},
    {"name": "position_delta", "width": 3, "unit": "cm", "mode": "delta", "frame": "camera:1"},
    {"name": "rotation_delta", "width": 3, "unit": "rad", "mode": "delta", "frame": "camera:1",
     "description": "spatial rotation vector"},
    {"name": "gripper", "width": 1, "unit": "ratio", "mode": "absolute",
     "description": "0 open, 1 closed"},
    {"name": "hand", "width": 12, "unit": "native", "mode": "absolute"}
  ]
}
```

| Key | Meaning |
|---|---|
| `checkpoint` | Step directory |
| `model_config` | Optional config to build the model with; default the run's |
| `device`, `weights` | Default `cpu` and `auto` (the moving average when the step has one) |
| `embodiment` | The robot's key in `config/embodiments.yaml` and the step's normalizer |
| `directory` | Corpus directory the robot's data came from; sets the rate and the usable columns |
| `arm_slots` | Model arm slots to drive, in order |
| `frame_ref` | Action frame, `robot`, `camera:N`, or one per arm; the frame used in post-training |
| `action_hz` | Control rate; only needed without a `directory` |
| `camera_names` | Image name per camera slot, `null` for an empty slot |
| `intrinsics_fields`, `extrinsics_fields` | State name per camera slot, or `null` |
| `state_fields` | State name holding each arm's 29-value record |
| `body_field` | Optional state name holding the 8-value body record |
| `output_fields` | Descriptions of the returned columns, in output order |
| `action_columns` | Optional model action columns to sample per arm, e.g. `[[7, 8, 9, 10, 11, 12, 13]]` for end-effector and gripper only |
| `prompt_template` | Optional prompt with `{instruction}`; by default it is rendered from `config/prompt_template/policy.yaml` exactly as in training |

Use the camera slots, frame and control rate of the post-training data. Camera
intrinsics are `fx, fy, cx, cy, width, height` in pixels. A camera pose is the
camera's forward and up axes in the robot base frame;
`neo_vla.data.conditioning.extrinsics_6` builds it from the base-from-camera
rotation (OpenCV axes). A `null` field is read as unknown, which is not the
same input as a measured value.

## Control loop

```python
import time

from neo_vla.serve import Client, Image, Observation, StateField

client = Client("http://gpu-host:8000")
previous = None
for step in range(max_steps):
    observation = Observation(
        request_id=f"episode-3-step-{step}",
        instruction="put the cup on the tray",
        timestamp_s=time.time(),
        images={"scene": Image(640, 480, scene_rgb_bytes()),
                "wrist": Image(640, 480, wrist_rgb_bytes())},
        state={"arm": StateField(arm_record(), "model_record"),
               "scene_intrinsics": StateField(SCENE_K, "px"),
               "wrist_intrinsics": StateField(WRIST_K, "px"),
               "scene_pose": StateField(scene_pose(), "unit"),
               "wrist_pose": StateField(wrist_pose(), "unit")},
        history=() if previous is None else (previous,),
    )
    chunk = client.predict(observation)
    for row in chunk.actions[:16]:
        robot.apply(row)                       # your controller
        time.sleep(1.0 / chunk.action_hz)
    previous = Observation(observation.request_id, observation.instruction,
                           observation.timestamp_s, observation.images, observation.state)
```

The client and the protocol types need neither torch nor numpy, so the loop
can run on the robot's own computer. Send the previous call's observation as
`history`, without its own history. The service turns the time between the
two into a count of executed rows, which must be at least the step's
`history.min_gap` (`meta.json`); execute at least that many rows of each chunk.

## Records

The arm state is 29 values per slot. Send `null` for anything the robot does
not measure; a measured zero stays zero.

| Columns | Meaning |
|---|---|
| 0–6 | Joint positions, rad |
| 7–9 | End-effector position in the base frame, m |
| 10–15 | First two columns of the end-effector rotation matrix |
| 16 | Gripper, 0 open and 1 closed |
| 17–28 | Hand |

Each arm action is 26 values:

| Columns | Meaning |
|---|---|
| 0–6 | Absolute joint targets, rad |
| 7–9 | Position increment per action step in the action frame, cm |
| 10–12 | Spatial rotation-vector increment per action step, rad |
| 13 | Gripper target, 0 open and 1 closed |
| 14–25 | Absolute hand targets |

The returned columns are the arm blocks in `arm_slots` order, then the body
block when the robot has one. A column the model was not trained to emit for
this robot comes back `null`; command only the valid ones. The service does no
unit, frame or gripper conversion: map the gripper back through the robot's
open and closed values, and turn camera-frame increments into the frame your
controller expects. The directory's record in `embodiments.json` also holds its
`unit_factors`, the centimetres and radians per unit of the original action
data.

Images are base64 RGB8, at most 12 MiB each; a request is at most 16 MiB.
Invalid requests get HTTP 400, oversized ones 413, failed predictions 500.

## Without HTTP

```python
from neo_vla.serve.loader import load_policy
from neo_vla.serve.predictor import Predictor

policy, report = load_policy("runs/posttrain_real/checkpoints/step_000030000", device="cuda")
predictor = Predictor(policy, arm_slots=[0], min_gap=report.meta["history"]["min_gap"],
                      device="cuda")
```

`predictor.observation(request)` checks one native request, `predictor([...])`
runs the policy and `predictor.decode(...)` returns physical actions;
`neo_vla.serve.neo_backend` shows how the HTTP backend builds the request.
