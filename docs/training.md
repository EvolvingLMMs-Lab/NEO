# Training NEO-VLA

The launchers implement three stages on the `skip28` architecture. Stages 2 and
3 use local COCO ShareGPT conversations in NEO's native chat format. The COCO
recipes use one unpacked chat row per robot row at loss weight 0.1. For
comparisons with reported results, use the checkpoint-specific experimental protocol.

Commands below run from the repository root. Install the package and obtain the
NEOpp-2B-SFT checkpoint and tokenizer as described in the [README](../README.md).

## COCO ShareGPT inputs

Obtain the annotations from [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V)
and COCO images under their respective terms. Use a COCO-only annotation file;
the loader reads the supplied list as-is and does not select COCO records from
a mixed-source annotation file.

```bash
export NEO_VLA_COCOGPTSHARE_RECORDS=/path/to/cocogptshare.json
export NEO_VLA_COCOGPTSHARE_IMAGES=/path/to/images
```

The annotation file must be a non-empty JSON list. Each record carries a
conversation and relative image reference. For example:

```json
[
  {
    "id": "coco-example",
    "image": "coco/train2017/000000000009.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the image."},
      {"from": "gpt", "value": "An image description."}
    ]
  }
]
```

The image root is joined directly to the reference: this example requires
`$NEO_VLA_COCOGPTSHARE_IMAGES/coco/train2017/000000000009.jpg`. Preserve the
relative paths in the selected annotations. `image` may also be a list, and an
`images` list is accepted. Unreadable or unrenderable rows are counted and
redrawn; persistent failures stop the loader. Prepare the file and image tree
consistently before a run.

Both COCO recipes set `min_pixels=65536`, `max_pixels=1003520` and
`max_tokens=4096`. A single image uses the stated pixel budget; multi-image
rows share NEO's image budget, with a per-image maximum of
`max(2 * max_pixels // image_count, min_pixels)`.
Assistant turns are supervised; image and user tokens are context. The reader
preserves NEO's native image convention and tag-mode chat rendering.

## Common environment

| variable | meaning |
|---|---|
| `NEO_VLA_BACKBONE` | NEOpp-2B-SFT checkpoint directory |
| `NEO_VLA_DATA_ROOT` | corpus root holding `manifests/<corpus>.json` and `prep_<corpus>/embodiments.json` |
| `NEO_VLA_OUTPUT_ROOT` | where run directories go: `<root>/<RUN_NAME>/checkpoints/step_<9 digits>` |
| `NEO_VLA_PYTHON` | interpreter to use (default `python`) |
| `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT` | distributed shape (defaults 1, 0, 8, -, 29500) |
| `RUN_NAME`, `WORKERS`, `PREFETCH` | run name and loader workers / batches read ahead per worker |

Set `MASTER_ADDR` to the rank-0 host and run the same command on every node
with that node's `NODE_RANK` (0 through `NNODES - 1`); the values
go to `torchrun` unchecked. Relaunching with the same `RUN_NAME` resumes from
the newest checkpoint. Anything after `--` is appended as a Hydra override.

Each recipe states its global batch; the accumulation is derived from the
world size (`NNODES x GPUS_PER_NODE`) and must divide exactly. The reference
shape of stages 1 and 2 is 16 nodes x 8 GPUs.

Keep the inductor cache (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`)
node-local; stages 1 and 2 compile layers with `torch.compile`.

Prompt caches record the exact checksum of `config/prompt_template/neo_v7.yaml`.
Build or re-render them with this file before training; caches made with a
different copy are rejected even when only its comments differ. Use the same
template throughout all three stages.

## Stage 1: pretrain

```bash
export NEO_VLA_BACKBONE=/path/to/NEOpp-2B-SFT
export NEO_VLA_DATA_ROOT=/path/to/corpus          # holds manifests/gd7_tokens_nobw.json
export NEO_VLA_OUTPUT_ROOT=/path/to/runs
NNODES=16 NODE_RANK=0 MASTER_ADDR=rank-0-host bash scripts/stage1.sh
```

Recipe (`mode=frozen_vlm data=gd7 schedule=pretrain`): 113,400 steps x 8,192
samples (8 per GPU x accumulation 8 x 128 GPUs), about one pass of the
corpus rows; AdamW lr 2e-4, betas (0.9, 0.95), weight decay 1e-10, 2,000
warmup steps, cosine to 2e-5, gradient clip 1.0; EMA 0.99; seed 0; layers
compiled (`trainer.compile_layers`), prefix trimmed to 64-column buckets,
no gradient checkpointing; a checkpoint every 1,000 steps, the last 2 kept.

**Input.** The prepared robot corpus `gd7_tokens_nobw` (2,704 directories,
2,231,583 episodes, 928,401,030 rows from AgiBot, OXE, DROID, RH20T,
InternData-A1, RoboCOIN, Galaxea, RoboMIND and BimanualYAM), with its
manifest, cache and normaliser under `NEO_VLA_DATA_ROOT`. The corpus is not
distributed with this code, and the per-source converters that produced it
are not included; `config/layouts/gd7.json` documents its layout and
`config/data/gd7.yaml` its mixture (InternData-A1 fixed at 15%, the other
families by row count^0.43) and row sampling.

**Preprocessing.** [preprocessing.md](preprocessing.md) gives the
commands that turn converted directories into the manifest, cache and
normaliser. It covers five quality stages:

1. implausible joint readings are stored as NaN (`tools/sidecars/`);
2. dead and wrapping joint and hand channels are masked (`tools/build_stats.py`);
3. an out-of-band action target masks only its own group at that step;
4. corrupt episodes are dropped with a named reason
   (`tools/validate_episodes.py`, `tools/filter_episodes.py`);
5. camera calibrations are checked, and a pose that fails the projection QC
   becomes unknown while its episode is kept (`tools/qc_extrinsics.py`,
   `tools/build_cache.py --pose-qc`).

A summary of the processed corpus is in
[dataset card](../data_release/DATASET_CARD.md).

## Stage 2: joint training

```bash
export STAGE1_CKPT=$NEO_VLA_OUTPUT_ROOT/stage1/checkpoints/step_000113400
# Set NEO_VLA_COCOGPTSHARE_RECORDS and NEO_VLA_COCOGPTSHARE_IMAGES above.
NNODES=16 NODE_RANK=0 MASTER_ADDR=rank-0-host bash scripts/stage2.sh
```

The `gd7_cotrain_cocogptshare` recipe initializes from the stage-1 EMA and trains
the full policy for 75,000 steps with a global robot batch of 8,192. The action
expert learning rate is 2e-4; the backbone, vision tower and chat head use 5e-5.
The schedule uses 2,000 warmup steps and cosine decay to 2e-5. EMA is 0.999,
kept on CPU and excluding the backbone; seed is 1. The robot path is compiled,
with gradient checkpointing across all backbone layers. A
checkpoint is saved every 5,000 steps and the latest two are retained.

## Stage 3: real-robot adaptation

```bash
export STAGE2_CKPT=$NEO_VLA_OUTPUT_ROOT/stage2/checkpoints/step_000075000
export NEO_VLA_DATA_ROOT=/path/to/real-corpus
export REAL_CORPUS=my_robot
export REAL_FRAME_SET='my_robot_key=[[camera:1]]'
# Set NEO_VLA_COCOGPTSHARE_RECORDS and NEO_VLA_COCOGPTSHARE_IMAGES above.
NNODES=2 NODE_RANK=0 MASTER_ADDR=rank-0-host bash scripts/posttrain_real.sh
```

The `real_cotrain_cocogptshare` recipe performs full fine-tuning from the stage-2
EMA. Defaults are 30,000 steps, a global robot batch of 128 and 4 rows per device.
Learning rates are 5e-5 for the expert and 2.5e-5 for the backbone, vision tower
and chat head, with 1,000 warmup steps, cosine decay to 1e-5 and weight decay
1e-8. EMA is disabled; seed is 42. Gradient checkpointing is enabled. A
checkpoint is saved every 1,000 steps and the latest is retained. Set
`MAX_STEPS`, `GLOBAL_BATCH` and `PER_DEVICE_BATCH` for your corpus and world size.

| Variable | Meaning |
|---|---|
| `REAL_CORPUS` | Manifest name under `NEO_VLA_DATA_ROOT/manifests/` |
| `REAL_NORMALISER` | `embodiments.json`; defaults to `prep_<REAL_CORPUS>/embodiments.json` |
| `STAGE2_CKPT` | Stage-2 checkpoint directory |
| `REAL_FRAME_MODE` | `wrist` (default), `random`, or `none` to retain each directory's manifest `frame_ref` |
| `REAL_FRAME_SET` | Space-separated `KEY=FRAMES` entries, e.g. `my_key=[[camera:1]]`; for random mode, `my_key=[[robot],[camera:1]]` |
| `REAL_FRAME_DRAW` | Draw a random frame per `episode` (default) or `sample` |

**Preparing the real-robot corpus.** The loader reads any schema-3 manifest
built by the preparation pipeline; it is not tied to one robot.

1. Convert raw recordings into source directories (per-episode parquet and
   video files with `meta/info.json` and `meta/episodes.jsonl`) with a
   converter for your robot; none is included here.
2. Describe the directories in a layout file (the schema `tools/layouts.py`
   reads; `config/layouts/gd7.json` is a full example) and give each new
   embodiment key a block in `config/embodiments.yaml` with `stage: finetune`
   (optionally `action_from: <pretrain key>`).
3. Build the manifest, cache and normaliser:

   ```bash
   python tools/prepare_data.py --root $NEO_VLA_DATA_ROOT --corpus $REAL_CORPUS \
       --prep-dir prep_$REAL_CORPUS --layouts config/layouts/my_robot.json \
       --dir <source dir> [--dir ...] --drop <rule> ... \
       --tokenizer $NEO_VLA_BACKBONE --tokenizer-id $NEO_VLA_BACKBONE \
       --stats-from $STAGE2_CKPT/embodiments.json --check-fit \
       [--inherit-action my_key=franka_panda] \
       [--stats-frame-policies camera:1]
   ```

   The prompt cache is stamped with `--tokenizer-id`, and the loader refuses
   a run whose `NEO_VLA_BACKBONE` string differs from it (or pass
   `-- data.train.tokenizer_id=<stamped id>`). `--stats-from` builds the
   normaliser against the stage-2 checkpoint's own keys; `--check-fit` and the
   run's start-up check hold every key to the normaliser's fit test. Use the
   ten `--drop` rules listed in [preprocessing.md](preprocessing.md)
   (stage 4). The layout file and `config/prompt_template/neo_v7.yaml` are
   read from under `--root`.
