# NEO-VLA Action Pretraining Data

Processed robot demonstrations used for NEO-VLA action pretraining. The corpus
contains 2,231,583 episodes and 928,401,030 timesteps from nine dataset families.

The data are available on [Hugging Face](https://huggingface.co/datasets/weepiess2383/NEO-VLA-Action-Pretrain).
The upload is ongoing; these counts refer to the full corpus, not the files
currently available for download.

| Document | Description |
|---|---|
| [Dataset card](DATASET_CARD.md) | Dataset composition, filtering, format and limitations |
| [Corpus summary](corpus_summary.json) | Episode and timestep counts, metadata schemas and exclusion rules |
| [Source inventory](sources_inventory.json) | Source datasets and their processed representations |
| [Source terms](UPSTREAM_TERMS.md) | Upstream licenses, attribution requirements and review status |
| [License metadata](upstream_license_metadata.json) | Upstream repositories and recorded license information |

## Data format

The training loader uses the following paths under `NEO_VLA_DATA_ROOT`:

| Path | Contents |
|---|---|
| `manifests/action_pretrain.json` | Dataset entries, camera views, action layouts and sampling weights |
| `metadata/action_pretrain/` | Normalization bounds, channel masks and filtering records |
| `processed/action_pretrain/` | Arrow trajectories, frame indexes, camera geometry and instruction caches |

Filtering code and commands are provided in the [filtering guide](../docs/preprocessing.md).
The Hugging Face dataset contains processed data and metadata; scripts are
maintained in this repository.

## Validity masks

The loader applies the channel masks recorded in `embodiments.json` and masks
outlier action groups using the normalization bounds. These masks must also be
applied when using a different loader. The [dataset card](DATASET_CARD.md)
distinguishes corrections stored in the files from masks applied during loading.

Full-corpus camera-pose projection validation has not been completed for the
current cache. See the dataset card for the scope of the calibration checks.

## Licenses

The source datasets retain their respective licenses. The repository's
Apache-2.0 license applies to code and does not relicense the data. Source license
review is ongoing; consult the [source terms](UPSTREAM_TERMS.md) for attribution
and redistribution requirements.
