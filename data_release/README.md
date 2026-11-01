# Data release

This directory describes the processed robot corpus NEO-VLA's action
pretraining reads, `action_pretrain`. The corpus is currently staged in a
private Hugging Face dataset repository and is not yet publicly downloadable;
each source keeps its own terms (below).

| file | contents |
|---|---|
| [`DATASET_CARD.md`](DATASET_CARD.md) | the dataset card: contents, size, processing, masks, limitations |
| [`corpus_summary.json`](corpus_summary.json) | counts per family and per manifest source, the manifest and normaliser schemas, the episode drop rules |
| [`sources_inventory.json`](sources_inventory.json) | the nine source families and their processed form; every licence is pending review |
| [`UPSTREAM_TERMS.md`](UPSTREAM_TERMS.md) | the terms identified per source and what remains to be resolved |
| [`upstream_license_metadata.json`](upstream_license_metadata.json) | the upstream repositories, revisions and licence metadata checked |

## Layout

Paths are relative to the data root (`NEO_VLA_DATA_ROOT`):

| path | contents |
|---|---|
| `manifests/action_pretrain.json` | every directory: cameras, arms, action layout, embodiment key, control rate, weight |
| `metadata/action_pretrain/` | the normaliser `embodiments.json` (per-key bounds and per-directory masks), the stage records of the five preprocessing stages, and the joint sidecars |
| `processed/action_pretrain/` | numeric arrow files, frame indexes, per-episode camera geometry and rendered prompts |
| `agibot/`, `oxe/`, `rh20t/`, ... | the converted source directories, one per dataset |

A corpus of your own, such as real-robot data for adaptation, uses the same
layout under its own name: `manifests/<corpus>.json`, `metadata/<corpus>/`,
`processed/<corpus>/`.

## Masks and source metadata

The loader applies masks that are recorded rather than baked into the values:
dead, constant and wrap-gated channels per directory and key in
`embodiments.json`, and action outliers per sample from its bounds. Anyone
training with other code must apply them. The manifest and the stage records
state where every value came from; the dataset card lists which effects are
materialised and which are applied at load time.

## Terms and status

The code is Apache-2.0; the data retain their original terms, which are
source-specific and pending review ([upstream terms](UPSTREAM_TERMS.md)).
The current processed cache records no projection census (`pose_qc`), so the
fifth filtering stage, the camera-pose projection check, is not certified for
it (see the dataset card).

The preprocessing pipeline that builds this layout is described in
[`docs/preprocessing.md`](../docs/preprocessing.md).
