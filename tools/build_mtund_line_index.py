"""Build the MT_Und line index a neo_native co-train reads (neo_vla/data/line_index.py).

Reads each entry's annotation file once, up to the line positions NEO's
trainer reads (the first ceil(length * repeat_time) lines, or every line of an
entry it cycles), and writes offsets.npy and index.json under --out. Refuses to
write into a directory that already holds an index: build a new one and point
NEO_VLA_MTUND_LINE_INDEX at it.

Usage:
    python tools/build_mtund_line_index.py --registry MT_Und_d0804.json --out DIR --workers 16
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from neo_vla.data.line_index import INDEX_FILE, build_line_index

logger = logging.getLogger("build_mtund_line_index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, help="the MT_Und registry json")
    parser.add_argument("--out", required=True, help="a new directory for the index")
    parser.add_argument("--workers", type=int, default=8, help="files read at once")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    if (out / INDEX_FILE).exists():
        logger.error("refused: %s exists; build into a new directory", out / INDEX_FILE)
        return 1
    start = time.time()
    index = build_line_index(args.registry, out, workers=args.workers, log=logger.info)
    records = index["entries"].values()
    logger.info(
        "entries %d offsets %d short %d empty %d seconds %.0f", len(index["entries"]),
        index["offsets"], sum(1 for r in records if r["used"] < r["need"]),
        sum(1 for r in records if r["count"] == 0), time.time() - start,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
