# Data release

This directory describes the processed NEO-VLA pretraining corpus and holds
the tools for preparing a **private** upload of it. Nothing is uploaded unless
`--upload` is passed, and an upload always goes to a private repository.

| file | purpose |
|---|---|
| `DATASET_CARD.md` | the dataset card: contents, size, processing, masks, limitations |
| `corpus_summary.json` | counts per family and per manifest source, manifest and normaliser identity, the episode drop rules |
| `sources_inventory.json` | the nine source families; every licence is pending review |
| `sources_manifest.template.json` | per-source licence record to fill in before a source can be released |
| `allowlist.example.txt` | which files under the data root a release may contain |
| `inventory.py` | lists the allowlisted files and flags what must not be released |
| `stage_upload.py` | builds a release plan; with `--upload`, sends it to a private repository |
| `release_lib.py` | shared helpers for the two scripts |

## Requirements

Python 3.10 or later. Both scripts use the standard library only, except that
`--upload` needs the Hugging Face client:

```bash
pip install huggingface_hub        # or: pip install -e '.[data-release]' from the repository root
hf auth login                      # stores your token with the client; the scripts never read it
```

## Preparing a release

All paths below are placeholders. Keep the filled files and every output
outside the data root; both scripts refuse to write under it.

1. **Record each source's licence.** Copy `sources_manifest.template.json`
   and fill in one entry per source: the upstream licence and its URL,
   whether redistribution is permitted and under which conditions, the
   attribution text, dated evidence of who checked it, the root-relative
   paths of that source's files, and the path of a copy of the upstream
   licence to ship with them. Set `approved_for_upload` only after review.
   Give each Open X-Embodiment dataset its own entry, since their terms differ.
2. **Choose the files.** Copy `allowlist.example.txt` and list the corpus
   records (manifest, `prep_<corpus>/`, layout and key table) and the
   directories of approved sources.
3. **Check what would be released:**

   ```bash
   python3 data_release/inventory.py --root <DATA_ROOT> --allowlist <allowlist> \
       --check-text --out <OUTSIDE>/inventory.json
   python3 data_release/stage_upload.py --root <DATA_ROOT> --sources <sources_manifest> \
       --allowlist <allowlist> --plan-out <OUTSIDE>/release_plan.json
   ```

   `release_plan.json` lists the planned files, the excluded files with their
   reasons, sources not yet approved and what each lacks, and any blocker.
   Exit code 2 means the plan is blocked.
4. **Upload, once approved:**

   ```bash
   python3 data_release/stage_upload.py --root <DATA_ROOT> --sources <sources_manifest> \
       --allowlist <allowlist> --plan-out <OUTSIDE>/release_plan.json \
       --upload --card data_release/DATASET_CARD.md \
       --repo-id <org>/<name> --confirm-repo-id <org>/<name>
   ```

   The upload creates the dataset repository as private, refuses an existing
   repository that is not private, and also uploads `SOURCES.json` (the
   source manifest) and `RELEASE_PLAN.json`. Changing visibility is a separate
   decision outside this tool.

## What the plan enforces

- A file is planned only if the allowlist matches it and it belongs to exactly
  one approved source or to the corpus records (`common_files`).
- A source is approved only with a licence id, `redistribution` `permitted`
  or `permitted_with_conditions` (conditions stated), attribution, dated
  evidence, a shipped licence copy and `approved_for_upload: true`. Template
  markers such as `<FILL ...>` count as missing.
- The corpus records the data needs travel with it: the manifest,
  `validation.jsonl`, `kept.jsonl`, `embodiments.json` and the layout. Masks
  that the loader applies live in `embodiments.json` and the manifest, so a
  release without them would misrepresent the data.
- A manifest naming a directory that no approved source covers blocks the
  plan. Build a manifest restricted to the approved directories with the
  preprocessing tools instead of editing one by hand, because its provenance
  hashes would no longer match.
- Hidden files, logs, lock and pid files, `plan.json` (it records the absolute
  root), `outputs/`, `checkpoints/`, and small text files whose first 64 KiB
  hold an absolute lab-style path or a private IP address are excluded.
- Symbolic links are never followed or uploaded. A link that resolves outside
  the root blocks the plan.
- Nothing is copied or hashed. Only the allowlisted prefixes are walked, by
  `lstat`, up to `--max-entries`.

## Remaining release checks

The current cache index has no recorded projection census (`pose_qc`). Verify
or complete the projection-census masking step before publishing the data as
using all five filters. Resolve the source-specific terms listed in
[upstream terms](UPSTREAM_TERMS.md). The code release is Apache-2.0; the data
retain their original terms.
