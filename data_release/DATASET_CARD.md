---
pretty_name: NEO-VLA Processed Action Pretraining
license: other
license_name: source-specific
license_link: https://github.com/WeichenFan/NEO-VLA/blob/main/data_release/UPSTREAM_TERMS.md
task_categories:
- robotics
tags:
- robotics
- vision-language-action
- processed
---

# NEO-VLA stage-1 robot corpus `gd7_tokens_nobw` (processed)

Processed robot demonstrations used to pretrain NEO-VLA, an academic
vision-language-action research model. The corpus combines nine public robot
datasets, converted to one layout, quality-filtered and cached for training.

**Status: prepared for private release.** These files are processed
derivatives of third-party datasets. Each source keeps its own licence and
terms, and every source is **pending licence review**
(`sources_inventory.json`). No licence is granted here. A source is shared
only after its review is complete.

## Size

2,704 directories, 2,231,583 episodes, 928,401,030 rows (timesteps).

| family | upstream dataset | directories | episodes | rows |
|---|---|---:|---:|---:|
| agibot | AgiBot World (Beta) | 198 | 158,100 | 282,649,801 |
| interndata | InternData-A1 (five robot setups) | 1,473 | 585,953 | 393,804,466 |
| bimanual_yam | BimanualYAM | 1 | 32,184 | 75,974,251 |
| robocoin | RoboCOIN | 417 | 98,797 | 74,053,114 |
| oxe | Open X-Embodiment | 120 | 1,210,137 | 30,275,046 |
| galaxea | Galaxea Open-World Dataset | 226 | 20,911 | 26,730,307 |
| robomind | RoboMIND | 18 | 53,097 | 19,607,926 |
| rh20t | RH20T | 210 | 32,323 | 14,557,623 |
| droid3cam | DROID | 41 | 40,081 | 10,748,496 |

Per-source counts (the manifest's 15 source keys) are in `corpus_summary.json`.
The counts are the manifest's own. They were read from its metadata without
re-scanning the payload.

| record | identity |
|---|---|
| manifest `manifests/gd7_tokens_nobw.json` | sha256 `7367f9c862ca5512378ae093e66ed785c9650a15b41c7f49b94a2d532ec90320` |
| normaliser `prep_gd7_tokens_nobw/embodiments.json` | sha256 `612852446612cd2180fabd840a7f61665bd3114306495bf56b1481c9ab547b4a` |
| code the corpus was built with | commit `9beb654ef40dfa3907a4b011216b472034737950` |

## Contents

| item | path | what it is |
|---|---|---|
| manifest | `manifests/gd7_tokens_nobw.json` | every directory, its cameras, arms, action layout, embodiment key, control rate and weight (schema 3) |
| normaliser | `prep_gd7_tokens_nobw/embodiments.json` | per-embodiment q01/q99 bounds and the masks training uses |
| stage records | `prep_gd7_tokens_nobw/*.jsonl`, `*.json` | inventory, per-episode validation defects, kept and dropped episodes with reasons, calibration records and their validation, frequency, gripper and embodiment declarations, weights; each starts with a provenance header (tool, version, commit, input hashes) |
| joint sidecars | `prep_gd7_tokens_nobw/sidecars/` | joints, hand and body for directories whose converted parquet lacks them |
| cache | `cache_gd7_tokens_nobw/` | numeric arrow files, frame index, per-episode camera geometry, rendered prompts |
| converted directories | one per dataset | per-episode parquet and video, `meta/info.json`, `meta/episodes.jsonl` |
| layout and keys | `config/layouts/gd7.json`, `config/embodiments.yaml` | per-dataset layout with its evidence; embodiment key table |

Which of these are shared is set per source by the release plan
(`stage_upload.py`). Vendor raw archives are never redistributed.

## Release validation

The current cache index has no `pose_qc` entry. The projection-census masking
step must be verified or completed before this corpus is described as having
all five filtering stages applied. The code includes the original scoring and
masking tools; this card does not certify that step for the existing cache.

## Processing

The tools and commands are in the code repository under `docs/preprocessing.md`.

1. **Implausible joint readings.** A joint reading beyond ±2π, or more than π
   from the median of the 5 rows of its episode centred on it, is stored as
   NaN. The episode is kept.
2. **Dead and wrapping channels.** A joint or hand column constant over a
   whole episode in at least 90% of a directory's episodes is masked for that
   directory (`dead_dims`). A column constant over every row of a directory is
   masked for it too. A joint stepping more than π between consecutive rows is
   masked for its whole embodiment key.
3. **Action outliers.** At each step, a target that is non-finite or outside
   `[q01 - (q99 - q01), q99 + (q99 - q01)]` of its embodiment's bounds removes
   only its own group (end-effector, joints or hand) from the loss. The rest of
   the step trains.
4. **Corrupt episodes.** Episodes are dropped by ten rules: `data_file_missing`,
   `episode_unreadable`, `rows_mismatch`, `frames_missing`, `episode_too_short`,
   `state_width_mismatch`, `action_width_mismatch`, `not_finite`,
   `quaternion_not_unit`, `action_equals_state`. Each removal is recorded with
   its rule and value. The original build reported 3 removed of 2,231,586
   checked, which leaves the 2,231,583 episodes of the manifest.
5. **Camera geometry.** Intrinsics are the sources' own calibrations, checked
   for focal length, principal point, field of view and frame size against the
   stored frames. A view without calibration states none. Stored camera poses
   are audited, and a pose that fails the audit is unknown for that episode.
   A full-pose projection check (motion-ratio score ≥ 2.5, with a wrong-pose
   control) scores the AgiBot head and DROID exterior poses. When the cache is
   built with the AgiBot census, a failed head pose is written as unknown. In
   every case the episode is kept. The DROID scores are a report.

### Which masks are in the files and which are applied at load time

| effect | where it lives |
|---|---|
| dropped episodes | **materialised**: absent from the kept list, the manifest and the cache; listed with their reason in `kept.jsonl` |
| implausible joint readings | **materialised**: NaN in the joint sidecars and in the cache built from them |
| not-recorded values | **materialised**: NaN (never 0) in stored columns |
| dead, constant and wrap-gated channels | **recorded, applied by the loader**: listed per directory and key in `embodiments.json`; the stored values are unchanged |
| action outliers | **applied by the loader** per sample and step from the normaliser bounds; nothing is removed from the data |
| finite values beyond 1e6 | reported by validation; **applied by the loader**, which reads them as not recorded |
| failed camera poses | **materialised** in the cache's per-episode camera arrays as the unknown marker, when the cache is built with the pose-QC census; the cache index then records the census files and verdict counts under `pose_qc` |

Anyone training with other code must apply the loader-side masks from
`embodiments.json` and the manifest. They are not baked into the values.

## Limitations

- Camera calibrations come only from what a source publishes. Most OXE
  datasets publish none, and their views state none.
- Joint readings exist only where a source records them. Elsewhere the
  columns are NaN and masked.
- Control rates are each source's nominal rate. Where stored timestamps
  disagree by more than 10%, the rate is stated as unknown.
- The counts above describe this build. A rebuild with different inputs or
  tools records different provenance hashes.

## Licences and citation

Every source dataset is distributed by its authors under its own terms, and
all of them are pending review here (`sources_inventory.json`). Open
X-Embodiment is a collection whose datasets each carry their own terms. When
you use this data, cite and attribute each upstream dataset as its authors
request.
