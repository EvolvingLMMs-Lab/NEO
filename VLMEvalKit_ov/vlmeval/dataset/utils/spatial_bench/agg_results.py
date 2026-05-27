import ast
import re
import pandas as pd

from pathlib import Path


def safe_len_candidates(val):
    """
    Try to infer the number of options from the candidates/options field:

    - If it can be parsed as a Python/JSON list, use len(list)
    - Else, count lines starting with labels like A./B)/C:
    - Else, count non-empty lines (>=2 considered as multiple options)
    """
    LETTER_PATTERN = re.compile(r"(?m)^\s*([A-F])\s*[\.\)\:：、]\s+")

    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (list, tuple)):
        return len(val)

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return len(parsed)
        except Exception:
            pass

        letters = set(LETTER_PATTERN.findall(s))
        if letters:
            return len(letters)

        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return len(lines)
        return None

    return None


def ensure_options_count(row, default_choices: int = 4) -> int:
    """
    Infer the number of choices for a single row.

    Priority:
      0) row['forced_n'] if provided and valid
      1) candidates / options via safe_len_candidates
      2) default_choices
    """
    # 0) forced_n
    if "forced_n" in row:
        fn = row["forced_n"]
        try:
            if fn is not None and not (isinstance(fn, float) and pd.isna(fn)):
                fn = int(fn)
                if fn > 0:
                    return fn
        except Exception:
            pass

    # 1) candidates / options
    n = None
    if "candidates" in row:
        n = safe_len_candidates(row["candidates"])
    if "options" in row:
        n = safe_len_candidates(row["options"])

    # 2) fallback to default
    return n if (isinstance(n, int) and n > 0) else default_choices


def add_num_choices(df: pd.DataFrame, default_choices: int = 4) -> pd.DataFrame:
    """
    Add a `num_choices` column to the dataframe by applying ensure_options_count
    row by row.
    """
    df = df.copy()
    df["num_choices"] = df.apply(
        lambda r: ensure_options_count(r, default_choices=default_choices),
        axis=1,
    )
    return df


def compute_metrics(df: pd.DataFrame):
    df = df.copy()
    df["hit"] = (
        pd.to_numeric(df["hit"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    )

    N = len(df)
    X = df["hit"].to_numpy(dtype=int)
    # n_list = df["num_choices"].to_numpy(dtype=float)

    n_list = df.apply(lambda r: ensure_options_count(r, default_choices=4), axis=1)
    df["num_choices"] = n_list

    sum_X = X.sum()
    inv_n = 1.0 / n_list
    sum_inv = inv_n.sum()
    denom = N - sum_inv

    acc = float(sum_X / N) if N > 0 else 0.0
    rand = float(inv_n.mean()) if N > 0 else 0.0
    caa = float((sum_X - sum_inv) / denom) if (N > 0 and denom != 0) else 0.0

    overall = dict(n=int(N), acc=acc, caa=caa, rand=rand)

    # Per-category metrics (same formula applied to each group)
    if "category" in df.columns:
        rows = []
        for cat, g in df.groupby("category", dropna=False):
            N_c = len(g)
            X_c = g["hit"].to_numpy(dtype=int)
            n_c = g["num_choices"].to_numpy(dtype=float)

            sum_X_c = X_c.sum()
            inv_n_c = 1.0 / n_c
            sum_inv_c = inv_n_c.sum()
            denom_c = N_c - sum_inv_c

            acc_c = float(sum_X_c / N_c) if N_c > 0 else 0.0
            rand_c = float(inv_n_c.mean()) if N_c > 0 else 0.0
            caa_c = (
                float((sum_X_c - sum_inv_c) / denom_c)
                if (N_c > 0 and denom_c != 0)
                else 0.0
            )

            rows.append(
                dict(
                    category=str(cat),
                    n=int(N_c),
                    acc=acc_c,
                    caa=caa_c,
                    rand=rand_c,
                )
            )

        percat = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    else:
        percat = pd.DataFrame(columns=["category", "n", "acc", "caa", "rand"])

    return overall, percat


def print_summary(overall: dict, percat: pd.DataFrame):
    """
    Pretty-print overall and per-category metrics to stdout.
    """
    print("\n=== Overall ===")
    print(
        f"n={overall['n']}, "
        f"acc={overall['acc']:.4f}, "
        f"caa={overall['caa']:.4f}, "
        f"rand={overall['rand']:.4f}"
    )

    if not percat.empty:
        df = percat.copy()
        for c in ["acc", "caa", "rand"]:
            df[c] = df[c].astype(float).map(lambda x: f"{x:.4f}")
        print("\n=== By category ===")
        print(df[["category", "n", "acc", "caa", "rand"]].to_string(index=False))
    else:
        print("\n(No per-category rows)")


def summarize_sitebench(dir_path: str, default_choices: int = 4):
    """
    Scan SiteBench result .xlsx files under `dir_path`, merge them,
    and compute overall and per-category metrics.

    Returns:
      {
        "overall":   dict(n, acc, caa, rand),
        "per_category": DataFrame,
        "merged":       merged raw DataFrame with `num_choices`
      }
    """
    root = Path(dir_path)
    patterns = [
        "*_SiteBenchVideo_32frame*.xlsx",
        "*_SiteBenchImage_llm*.xlsx",
    ]

    files = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))

    print(files)

    if not files:
        print("No SiteBench result files found.")
        return

    frames = []
    for p in files:
        try:
            df = pd.read_excel(p)
            if "hit" not in df.columns:
                print(f"[Skip] {p.name} missing column: ['hit']")
                continue

            # df = add_num_choices(df, default_choices=default_choices)
            df["__file__"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"[Read failed] {p.name}: {e}")

    if not frames:
        print("No valid data.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    overall, percat = compute_metrics(all_df)
    print_summary(overall, percat)

    return dict(overall=overall, per_category=percat, merged=all_df)


if __name__ == "__main__":
    result = summarize_sitebench(
        "",
        default_choices=4,
    )
