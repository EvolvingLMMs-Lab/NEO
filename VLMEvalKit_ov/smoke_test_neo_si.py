#!/usr/bin/env python
"""Self-contained smoke test for the migrated NEOChatSI (spatial-intelligence) model.

Runs a few MindCube-tiny samples end-to-end through NEOChatSI.generate_inner and
reports rough letter-match accuracy. This exercises the real GPU inference path
that cannot be validated on a CPU-only node.

Usage (on a GPU node, easi env):
    CUDA_VISIBLE_DEVICES=0 \
    /mnt/afs/zhuyue1/miniconda3/envs/easi/bin/python smoke_test_neo_si.py [N]

    N = number of samples to test (default 20). Use 0 for the full dataset.

Reference (EASI logs, MindCubeBench_tiny_raw_qa, 2B):
    NEOov-2B  ~68-76% overall  |  Qwen3-VL-2B 35%  |  InternVL3.5-2B 43%
A passing smoke test should land well above ~35% and, above all, run without error.
"""
import os
import re
import sys

from vlmeval.dataset import build_dataset
from vlmeval.vlm.neo_si import NEOChatSI

MODEL_PATH = "/mnt/afs/zhuyue1/pretrained_model/NEO1_5-2B-SFT"
DATASET = "MindCubeBench_tiny_raw_qa"


def extract_letter(text):
    """Best-effort extraction of the chosen option letter from a response."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\b([A-H])\b", text.strip())
    return m.group(1).upper() if m else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    print(f"[smoke] loading dataset {DATASET} ...")
    ds = build_dataset(DATASET)
    total_rows = len(ds.data)
    n = total_rows if n <= 0 else min(n, total_rows)
    print(f"[smoke] dataset rows: {total_rows}, testing first {n}")

    print(f"[smoke] loading NEOChatSI from {MODEL_PATH} ...")
    # Mirror the committed NEOov-2B-si config settings.
    model = NEOChatSI(
        model_path=MODEL_PATH,
        patch_size=16,
        min_pixels=720 * 960,
        max_pixels=720 * 960,
        downsample_ratio=0.5,
    )
    print("[smoke] model loaded OK")

    # Wire up image dumping the way run.py's inference loop does.
    model.set_dump_image(ds.dump_image)

    correct = 0
    counted = 0
    for i in range(n):
        line = ds.data.iloc[i]
        message = model.build_prompt(line, DATASET)
        resp = model.generate_inner(message, DATASET)
        gt = str(line["answer"]).strip().upper()
        pred = extract_letter(resp)
        ok = pred is not None and pred == gt
        counted += 1
        correct += int(ok)
        print(f"[{i:03d}] gt={gt} pred={pred} ok={ok} | resp={str(resp)[:80]!r}")

    acc = 100.0 * correct / counted if counted else 0.0
    print(f"\n[smoke] rough letter-match accuracy: {correct}/{counted} = {acc:.1f}%")
    if acc >= 45.0:
        print("[smoke] PASS (well above chance / competitor baselines)")
    else:
        print("[smoke] WARN: accuracy below expected band — inspect outputs above")


if __name__ == "__main__":
    main()
