# Third-party notices

The NEO backbone implementation in `neo_vla/model/backbone/neo_llm.py` is
adapted from the NEO checkpoint's modeling code and has Qwen3 lineage, as its
module documentation records. The backbone checkpoint, tokenizer and external
libraries are obtained separately and retain their upstream terms. Existing
source attribution is preserved.

The Apache-2.0 grant for NEO-VLA does not relicense upstream datasets. Their
terms and release checks are recorded in `data_release/UPSTREAM_TERMS.md`.

The default multimodal understanding example uses COCO ShareGPT, with
ShareGPT4V-format annotations and COCO images supplied by the user. The annotation and image licenses apply
separately; these datasets are not included in this repository. See the
[ShareGPT4V project](https://github.com/ShareGPT4Omni/ShareGPT4V) for the upstream
annotation source.
