"""Row-aligned v7 sidecars for the stage-1 directories.

The 569 stage-1 directories keep their parquet; the v7 columns their parquet
does not hold (arm joints, hand, the body block and DROID's joint command)
arrive as one sidecar per directory under prep_<corpus>/sidecars/<entry>.parquet.
common holds the format and the one reader; agibot, droid, rh20t and oxe each
read one family's source; run writes them, sharded and resumable, and verify
re-reads them against their directories.
"""
