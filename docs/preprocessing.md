# Data preprocessing

These tools turn converted robot datasets into the manifest, cache and
normaliser that training reads. Every stage is one program under `tools/` and
writes one file. That file starts with a provenance header: schema version,
tool and version, git commit, the root-relative path of each input, and the
creation time.
Paths inside outputs are relative to the corpus root (`--root`).

The pipeline starts from converted directories, one per dataset: per-episode
parquet, video or image columns, `meta/info.json` and `meta/episodes.jsonl`.
The converters from vendor archives are not part of this repository.

Install the preprocessing extras (`h5py`, `scipy`, OpenCV) with
`pip install -e '.[prep]'`, and run every tool from the repository root with
the package importable:

```bash
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
R=$NEO_VLA_DATA_ROOT          # data root
C=action_pretrain             # corpus name
P=metadata/$C                 # stage records and normaliser, as scripts/train.sh expects
```

Every corpus uses one layout under the root: the manifest at
`manifests/<corpus>.json`, stage records, normaliser and sidecars under
`metadata/<corpus>/`, and the numeric, frame, camera and prompt caches under
`processed/<corpus>/`. `tools/prepare_data.py` uses these by default.

Inputs are read from under the root. That includes the calibration and pose
trees passed with `--calib`, which may be links into the source trees. Copy
`config/layouts/<layout>.json` and `config/prompt_template/policy.yaml` there
at the same relative paths; prompt caches record the template's name, and a
cache rendered from an edited template must be rebuilt.
`config/embodiments.yaml` and `config/rates/oxe_published.json` are read from
the repository.

## The five quality stages

| # | rule | tools | what happens to the data |
|---|---|---|---|
| 1 | implausible joint readings | `tools/sidecars/` | a reading beyond ±2π, or more than π from the median of the 5 in-episode rows centred on it, is stored as NaN (not recorded); the episode is kept |
| 2 | dead and wrapping channels | `build_stats.py` | masked per directory (dead) or per embodiment key (wrap) in the normaliser |
| 3 | action outliers | `build_stats.py` (bounds), the loader (mask) | an out-of-band or non-finite target removes only its own group at that step |
| 4 | corrupt episodes | `validate_episodes.py`, `filter_episodes.py` | dropped, each with the named rule and the value that fired it |
| 5 | camera geometry | `build_intrinsics.py`, `validate_intrinsics.py`, `build_extrinsics.py`, `qc_extrinsics.py`, `qc_droid_extrinsics.py`, `build_cache.py --pose-qc` | a pose that fails its checks is written as unknown for that episode; the episode is kept |

### 1. Joint readings (stage-1 joint sidecars)

Directories whose converted parquet has no joint, hand or body columns (AgiBot,
DROID, RH20T and OXE group A) get one row-aligned sidecar parquet each. It is
read from the source's own files and merged into the cache later. Every
episode is checked against its directory: the row count, and, where the
source records one, the end-effector track row by row. A directory whose
source does not line up is refused and named in the report. Nothing is
written for it.

`tools/sidecars/common.py` `implausible_joints` applies rule 1: a reading
beyond `JOINT_CEILING_RAD = 2π`, or more than `SPIKE_RAD = π` from the median
of the `JOINT_WINDOW = 5` rows centred on it, is written as NaN. The RH20T
writer (and the post-train LIBERO writer) applies it, because RH20T's
`joint.npy` holds isolated impossible values such as 1.88e38. A sustained jump
moves the median with it, so a real wrap still reaches the wrap gate (stage 2).
`sidecars/verify.py` re-reads every written sidecar against its directory,
checks that no joint reading in any family lies beyond 2π, and counts the
in-episode steps past π.

```bash
# RH20T: stream the lowdim archives once (joint.npy lives only inside them)
python tools/sidecars/run.py rh20t-extract --root $R --manifest <manifest naming the directories> \
    --archive <RH20T_cfg1.tar.gz> --archive <RH20T_cfg2.tar.gz> ... --out $R/$P/sidecars
# write one sidecar per directory (resumable; --shard K/N splits by rows across machines)
python tools/sidecars/run.py build --root $R --manifest <manifest naming the directories> \
    --families agibot,droid,rh20t,oxe --agibot-raw <AgiBot World proprio_stats root> \
    --rh20t-source-root <RH20T v3 source root> [--droid-raw <DROID raw root>] \
    --out $R/$P/sidecars --workers 24
# re-read all of them against their directories; optionally compare N raw episodes
python tools/sidecars/verify.py --root $R --manifest <same manifest> --out $R/$P/sidecars \
    [--sample-episodes 200 --agibot-raw <...> --droid-raw <...>]
```

`--manifest` can be any manifest whose entries name the directories (each
entry named after its directory, with its row count). For a corpus that has no
manifest yet, `build` takes `--layouts <layout> --dirs D1,D2,...` together with
an explicit `--out`. `build_manifest.py --sidecars $P/sidecars` then checks each
sidecar's stamp and episodes and names it in the manifest, and `build_cache.py`
merges the sidecars into the arrow files.

### 2. Dead and wrapping channels

`build_stats.py` scans every directory in full (`--constant-scan full`) and
masks what cannot be a target. Each gate logs its count and the denominator:

- **dead reading** (`DEAD_SHARE = 0.9`): a **joint or hand** column whose
  reading is constant over a whole episode in at least 90% of a directory's
  read episodes is masked for that directory and listed in its `dead_dims`.
  The gate covers joint and hand columns only. A column constant over every
  valid row of a directory (any column) is masked by the separate constant
  scan, and the end-effector and gripper columns are judged by that scan, not
  per episode.
- **wrap** (`WRAP_STEP = π`): a joint with any step larger than π between two
  consecutive rows of an episode is masked for the whole embodiment key.

The masks are written per directory and per key into `embodiments.json`. The
loader trains under them, and serving rebuilds them from the same file.

### 3. Action outliers

`build_stats.py` measures weighted q01/q99 per (embodiment key, slot,
dimension) over the loader's own targets (`--quantiles 1 99`) and records the
band alpha (`--band-alpha 1.0`). At training time
`neo_vla/data/embodiment.py` (`OUT_OF_BAND = "loss_mask_group"`) maps each
target. A value that is non-finite, or outside
`[q01 - α(q99 - q01), q99 + α(q99 - q01)]`, removes only its own group at that
step: end-effector, joints or hand per arm (body base and posture for the
body). The other groups at that step still train. Nothing is dropped from the
stored data.

### 4. Corrupt episodes

```bash
python tools/scan_source.py --root $R --dirs-file dirs.txt --out $P/inventory.jsonl
python tools/validate_episodes.py --root $R --inventory $P/inventory.jsonl \
    --layouts config/layouts/action_pretrain.json --out $P/validation.jsonl
python tools/filter_episodes.py --root $R --validation $P/validation.jsonl --out $P/kept.jsonl \
    --drop data_file_missing --drop episode_unreadable --drop rows_mismatch \
    --drop frames_missing --drop episode_too_short --drop state_width_mismatch \
    --drop action_width_mismatch --drop not_finite --drop quaternion_not_unit \
    --drop action_equals_state
```

These ten drop rules are the reference corpus's drop list, verbatim. Every
dataset block of its layout lists no additional drops. `validate_episodes.py`
measures each video's size on disk, so an absent or empty stream is
`frames_missing`. `episode_too_short` uses `--min-rows`, default 2.
`filter_episodes.py` writes a `dropped` record naming the rule and its value
for every removal, and refuses to write unless kept + dropped = validated.
Other defects are reported and kept. `record_out_of_range` (a finite value
beyond `RECORD_LIMIT = 1e6`) is not a drop rule: the loader reads such a
value as not recorded. `verify_manifest.py` also checks that every frame file
the manifest names exists, is non-empty and starts as the container it claims.
For the reference corpus, 3 of 2,231,586 checked episodes were removed. That is
the figure's historical count, not a recomputation. It agrees with the
2,231,583 episodes of the published manifest.

### 5. Camera geometry

Intrinsics are copied from each source's own calibration together with the
frame size they were measured against:

```bash
python tools/build_intrinsics.py --root $R --source agibot --calib <AgiBot calibration tree> \
    --dirs-file agibot_dirs.txt --layouts config/layouts/action_pretrain.json --out $P/intrinsics_agibot.jsonl
python tools/validate_intrinsics.py --root $R --intrinsics $P/intrinsics_agibot.jsonl \
    --layouts config/layouts/action_pretrain.json --out $P/intrinsics_validation.jsonl
```

`validate_intrinsics.py` names, per record: non-positive focal lengths or
frame size, a principal point off-centre or outside the frame, an implausible
field of view, a calibration aspect that disagrees with the stored frames,
duplicates, and a provenance outside `neo_vla.data.rays.PROVENANCE_ALLOWED`.
`verify_manifest.py` repeats the aspect check against the cache.

Vendor poses go through `build_extrinsics.py` (`--source agibot` for the head,
`--source agibot_wrist` for the wrist mounts). A record with any defect
(`pose_not_rotation`, `pose_implausible`, `pose_varies_within_episode`, a
failed identity join, ...) carries no rotation.

The full-pose projection QC checks that a stored pose describes the camera its
frames show. The arm is the only thing moving in these scenes. Under a correct
pose, the projected arm points land on moving pixels. The score is the motion
energy at the projected points divided by the motion energy of the whole
frame. A score of at least 2.5 is `ok` and below 2.5 is `fail`. An episode
with no reading (fewer than 30 samples, or no calibration, pose or video) is
`inconclusive`. `--control` scores every episode a second time
under a deliberately wrong pose: another directory's pose at least 10° away
for AgiBot, another lab's pose for DROID. The header then reports how often
the episode's own pose beats its control (`win_rate`).

```bash
# AgiBot head camera: a full census (no --sample) with the wrong-pose control
python tools/qc_extrinsics.py --root $R --source agibot --control \
    --calib <AgiBot pose tree holding head_extrinsic_sidecar.jsonl> \
    --intrinsics $P/intrinsics_agibot.jsonl --dirs-file agibot_dirs.txt \
    --layouts config/layouts/action_pretrain.json --out $P/qc_extrinsics_agibot.jsonl --workers 32
# DROID exterior cameras: same score and threshold, lab-swap control
python tools/qc_droid_extrinsics.py --root $R --control --calib <DROID pose tree> \
    --dirs-file droid_dirs.txt --urdf <Franka Panda URDF> --out $P/qc_droid.jsonl --workers 32
```

`qc_droid_extrinsics.py` needs a Franka Panda URDF supplied by you. Its hand
links are what gets projected. The URDF is third-party and is not distributed
here; follow its own licence. The output header records the URDF's resolved
path.

`build_cache.py --pose-qc` applies the census. It accepts only a
`qc_extrinsics.py` output whose header states a census (`sample` 0), the 2.5
threshold, a control run (`win_rate`, `control_verdicts`) and verdict counts
that match its records. For each `fail` episode, that camera's pose is
written as the unknown marker. The loader's camera-pose channel then says
unknown, and the camera can no longer be the action frame. If no other camera
of the episode has a valid pose, the episode is drawn in the robot frame, and
its prompt clause, rendered from the same arrays, says so. The episode stays in
the kept list. `ok` and `inconclusive` keep
the pose that `build_extrinsics` audited, and the index counts kept episodes
by verdict. Only views whose extrinsics are `kind: episode` from a stage file
(the AgiBot head) can be gated this way. DROID's poses reach the cache through
episode metadata, so the DROID QC output is a report, and `build_cache.py`
does not consume it.

```bash
python tools/build_cache.py --root $R --manifest manifests/$C.json --kept $P/kept.jsonl \
    --inventory $P/inventory.jsonl --parts episode_starts,arrow,frames,extrinsics,intrinsics,prompt \
    --tokenizer $NEO_VLA_BACKBONE --out processed/$C/index.json \
    --pose-qc $P/qc_extrinsics_agibot.jsonl
```

## Running everything in order

`tools/prepare_data.py` chains the stages it knows, re-running only those whose
arguments or tool version changed or whose output, inputs or artifacts are
missing: scan, validate, filter, scale, frequency, weights, calibration,
embodiment, gripper, manifest, cache, normaliser, verify. File contents are not
compared, so after editing an input in place pass `--force`, or `--only` with
the stages that read it. It does not run the sidecars or the pose QC, and its
cache step takes no `--pose-qc`. For a build with both:

1. Write and verify the joint sidecars (stage 1).
2. Run `prepare_data.py` through the manifest, passing the drop list and the
   sidecars:

   ```bash
   python tools/prepare_data.py --root $R --corpus $C \
       --layouts config/layouts/action_pretrain.json --dirs-file dirs.txt --sidecars $P/sidecars \
       --history-blocked tools/lists/rh20t_history_blocked_20260930.json \
       --calib agibot=<calibration tree> --calib-dirs agibot=agibot_dirs.txt \
       --extrinsics agibot=<pose tree> --extrinsics-dirs agibot=agibot_dirs.txt \
       --drop data_file_missing --drop episode_unreadable --drop rows_mismatch \
       --drop frames_missing --drop episode_too_short --drop state_width_mismatch \
       --drop action_width_mismatch --drop not_finite --drop quaternion_not_unit \
       --drop action_equals_state --share <source>=<share> ... \
       --tokenizer-id <id> --only scan_source --only validate_episodes \
       --only filter_episodes --only measure_scale --only declare_frequency \
       --only build_weights --only build_intrinsics:agibot --only build_extrinsics:agibot \
       --only validate_intrinsics --only build_embodiment --only measure_gripper_polarity \
       --only build_manifest
   ```

3. Run the pose QC census, then `build_cache.py --pose-qc` as above.
4. Run the normaliser and the verification:
   `python tools/prepare_data.py <same arguments> --only build_stats --only verify_manifest`.

Training reads `manifests/$C.json` and `$P/embodiments.json`.

## Reproducing the reference corpus

Use the released manifest and normalization records with the processed reference
corpus. Keep these metadata with their corresponding robot rows and frame
references. Rebuilding the corpus recomputes quantile estimates from sampled
windows; source revisions and sampling can affect the resulting bounds.

## A new real-robot corpus (post-train)

1. Convert your recordings into per-episode directories with `meta/info.json`
   and `meta/episodes.jsonl`.
2. Describe them in a layout file (schema read by `tools/layouts.py`;
   `config/layouts/action_pretrain.json` is a complete example). Give each new embodiment
   key a block in `config/embodiments.yaml` with `stage: finetune`.
3. Run `tools/prepare_data.py` with `--stats-from <stage-2 checkpoint>/embodiments.json
   --check-fit` (and `--inherit-action KEY=SRC` where needed). Read the defect
   counts in `validation.jsonl` and the drops in `kept.jsonl` before training.
