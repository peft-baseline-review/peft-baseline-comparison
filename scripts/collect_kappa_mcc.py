#!/usr/bin/env python
"""
scripts/collect_kappa_mcc.py

Collect Cohen's kappa and MCC from saved experiment_results.json files
and aggregate them into the layout used by Supplementary Table S7.

Reads (walks recursively):
    <experiments-root>/**/experiment_results.json

Extracts from each file:
    experiment_metadata.name   -> e.g., "Afro-XLM-R SAP-PEFT"
    experiment_metadata.seed
    results.model_metrics.cohen_kappa
    results.model_metrics.mcc

Infers the experiment group from the name + task folder and the
method from the name suffix, then aggregates mean and sample std per
(experiment, method) across the five seeds.

Usage:
    python scripts/collect_kappa_mcc.py \
        --experiments-root experiments \
        --output kappa_mcc.csv \
        --tex kappa_mcc.tex

Windows PowerShell:
    python scripts\collect_kappa_mcc.py `
        --experiments-root experiments `
        --output kappa_mcc.csv `
        --tex kappa_mcc.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


# ---------------- experiment group and method inference ----------------

# Order of experiments as they should appear in the table columns
EXPERIMENT_ORDER = [
    "tiroberta_tidg",
    "tiroberta_sentiment",
    "afro_xlmr_tidg",
    "afro_xlmr_sentiment",
    "gpt2_tidg",
    "gpt2_sentiment",
]

# Order of methods as they should appear in the table rows
METHOD_ORDER = [
    "full_finetuning",
    "vanilla",
    "lora",
    "dora",
    "adalora",
    "progressive_adaptive",
    "progressive_linear",
    "progressive_fixed",
    "sap_peft",
]

DISPLAY = {
    "full_finetuning":      "Full Fine-Tuning",
    "vanilla":              "Vanilla",
    "lora":                 "LoRA",
    "dora":                 "DoRA",
    "adalora":              "AdaLoRA",
    "progressive_adaptive": "Progressive-Adaptive",
    "progressive_linear":   "Progressive-Linear",
    "progressive_fixed":    "Progressive-Fixed",
    "sap_peft":             "SAP-PEFT",
}

COLUMN_HEADER = {
    "tiroberta_tidg":      "TiroB. news",
    "tiroberta_sentiment": "TiroB. sent.",
    "afro_xlmr_tidg":      "Afro news",
    "afro_xlmr_sentiment": "Afro sent.",
    "gpt2_tidg":           "GPT-2 news",
    "gpt2_sentiment":      "GPT-2 sent.",
}


# The "name" field in experiment_metadata is like:
#   "Afro-XLM-R SAP-PEFT", "TiroBERTa TiDG Full Fine-tuning", "GPT-2 LoRA"
# We normalise the model and method tokens separately.

MODEL_PATTERNS = [
    ("tiroberta",  "tiroberta"),
    ("afro",       "afro_xlmr"),
    ("gpt 2",      "gpt2"),   # matches normalised "GPT-2" (U+2011 → space)
    ("gpt-2",      "gpt2"),   # safety net for ASCII hyphen
    ("gpt2",       "gpt2"),   # safety net for unspaced variant
]

TASK_PATTERNS = [
    ("tidg",       "tidg"),
    ("news",       "tidg"),
    ("sentiment",  "sentiment"),
]

# Method detection: order matters, longest/most specific first
METHOD_PATTERNS = [
    ("full fine-tuning",      "full_finetuning"),
    ("full fine tuning",      "full_finetuning"),
    ("full",                  "full_finetuning"),
    ("vanilla",               "vanilla"),
    ("progressive adaptive",  "progressive_adaptive"),
    ("progressive linear",    "progressive_linear"),
    ("progressive fixed",     "progressive_fixed"),
    ("progressive-adaptive",  "progressive_adaptive"),
    ("progressive-linear",    "progressive_linear"),
    ("progressive-fixed",     "progressive_fixed"),
    ("sap-peft",              "sap_peft"),
    ("sap peft",              "sap_peft"),
    ("sappeft",               "sap_peft"),
    ("sap peft",              "sap_peft"),
    ("adalora",               "adalora"),
    ("dora",                  "dora"),
    # Must come before "lora" so "dylora"/"salora" are not captured by
    # the substring "lora". EXCLUDE means: skip silently.
    ("dylora",                "EXCLUDE"),
    ("dy lora",               "EXCLUDE"),
    ("salora",                "EXCLUDE"),
    ("sa lora",               "EXCLUDE"),

    ("lora",                  "lora"),
]


def normalise(s: str) -> str:
    """Lowercase, strip, collapse whitespace and dashes for pattern matching."""
    s = s.lower()
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_experiment_key(name: str, task_dir: str | None) -> str | None:
    """Return one of the EXPERIMENT_ORDER keys, or None if not recognised."""
    n = normalise(name)
    task_hint = normalise(task_dir) if task_dir else ""

    model = None
    for pat, key in MODEL_PATTERNS:
        if pat in n:
            model = key
            break
    if model is None:
        return None

    # Prefer task hint from the folder; fall back to name
    task = None
    for pat, key in TASK_PATTERNS:
        if pat in n:
            task = key
            break
    if task is None:
        for pat, key in TASK_PATTERNS:
            if pat in task_hint:
                task = key
                break
    if task is None:
        return None

    return f"{model}_{task}"


def infer_method_key(name: str) -> str | None:
    n = normalise(name)
    for pat, key in METHOD_PATTERNS:
        if pat in n:
            return key
    return None


# ---------------- discovery and extraction ----------------

@dataclass
class Record:
    experiment: str
    method: str
    seed: int
    kappa: float
    mcc: float
    source: Path


def parse_one(path: Path, experiments_root: Path) -> Record | None:
    """Parse one experiment_results.json. Return None if unusable."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  skip (json parse): {path}: {e}", file=sys.stderr)
        return None

    try:
        meta = data.get("experiment_metadata") or {}
        name = meta.get("name") or ""

        # Seed can live in several places depending on when the run was
        # produced. Try each in order of reliability.
        seed = None
        seed_sources = [
            meta.get("seed"),
            (meta.get("reproducibility_info") or {}).get("seed"),
            (data.get("configuration") or {}).get("seed"),
        ]
        for candidate in seed_sources:
            if candidate is not None:
                seed = candidate
                break

        metrics = (data.get("results") or {}).get("model_metrics") or {}
        kappa = metrics.get("cohen_kappa")
        mcc = metrics.get("mcc")
    except Exception as e:
        print(f"  skip (field access): {path}: {e}", file=sys.stderr)
        return None

    if seed is None or kappa is None or mcc is None:
        print(f"  skip (missing field): {path}  "
              f"seed={seed}, kappa={kappa}, mcc={mcc}", file=sys.stderr)
        return None

    # Task folder hint: the directory immediately under the experiment name
    rel = path.relative_to(experiments_root)
    parts = rel.parts
    task_dir = parts[1] if len(parts) > 1 else None

    exp_key = infer_experiment_key(name, task_dir)
    if exp_key is None:
        print(f"  skip (unknown experiment): {path}  name='{name}'  "
              f"task_dir='{task_dir}'", file=sys.stderr)
        return None

    method_key = infer_method_key(name)
    if method_key == "EXCLUDE":
        # DyLoRA and SaLoRA are not baselines in the paper.
        return None
    if method_key is None:
        print(f"  skip (unknown method): {path}  name='{name}'", file=sys.stderr)
        return None

    return Record(
        experiment=exp_key,
        method=method_key,
        seed=int(seed),
        kappa=float(kappa),
        mcc=float(mcc),
        source=path,
    )


def discover(experiments_root: Path) -> list[Record]:
    # Collect all candidates first.
    candidates: list[Record] = []
    for path in sorted(experiments_root.rglob("experiment_results.json")):
        r = parse_one(path, experiments_root)
        if r is not None:
            candidates.append(r)

    # Deduplicate: for each (experiment, method, seed), keep the record
    # whose path contains the lexicographically latest timestamp folder.
    # The runner names timestamp folders YYYYMMDD_HHMMSS, so lexical order
    # equals chronological order.
    def timestamp_key(rec: Record) -> str:
        # The timestamp is the immediate parent of experiment_results.json
        return rec.source.parent.name

    latest: dict[tuple[str, str, int], Record] = {}
    for r in candidates:
        key = (r.experiment, r.method, r.seed)
        prev = latest.get(key)
        if prev is None or timestamp_key(r) > timestamp_key(prev):
            latest[key] = r

    return list(latest.values())


# ---------------- aggregation ----------------

@dataclass
class Cell:
    experiment: str
    method: str
    kappa_mean: float
    kappa_std: float
    mcc_mean: float
    mcc_std: float
    n_seeds: int
    seeds: list[int]


def aggregate(records: list[Record]) -> list[Cell]:
    buckets: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        buckets.setdefault((r.experiment, r.method), []).append(r)

    cells: list[Cell] = []
    for (exp, method), items in buckets.items():
        ks = [x.kappa for x in items]
        ms = [x.mcc for x in items]
        seeds = sorted(x.seed for x in items)
        cells.append(Cell(
            experiment=exp,
            method=method,
            kappa_mean=mean(ks),
            kappa_std=stdev(ks) if len(ks) > 1 else 0.0,
            mcc_mean=mean(ms),
            mcc_std=stdev(ms) if len(ms) > 1 else 0.0,
            n_seeds=len(items),
            seeds=seeds,
        ))
    return cells


# ---------------- output ----------------

def write_csv(cells: list[Cell], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "method", "n_seeds", "seeds",
                    "kappa_mean", "kappa_std",
                    "mcc_mean", "mcc_std"])
        for c in cells:
            w.writerow([
                c.experiment, c.method, c.n_seeds,
                "|".join(str(s) for s in c.seeds),
                f"{c.kappa_mean:.6f}", f"{c.kappa_std:.6f}",
                f"{c.mcc_mean:.6f}", f"{c.mcc_std:.6f}",
            ])
    print(f"wrote {out}")


def print_console(cells: list[Cell]) -> None:
    idx = {(c.experiment, c.method): c for c in cells}
    exps = [e for e in EXPERIMENT_ORDER if any(c.experiment == e for c in cells)]
    methods = [m for m in METHOD_ORDER if any(c.method == m for c in cells)]

    for label, km, ks in [
        ("Cohen's kappa", "kappa_mean", "kappa_std"),
        ("MCC", "mcc_mean", "mcc_std"),
    ]:
        print()
        print("=" * 110)
        print(label)
        print("=" * 110)
        header = ["Method".ljust(24)] + [
            COLUMN_HEADER.get(e, e).ljust(16) for e in exps
        ]
        print("".join(header))
        print("-" * sum(len(h) for h in header))
        for m in methods:
            row = [DISPLAY[m].ljust(24)]
            for e in exps:
                c = idx.get((e, m))
                if c is None:
                    row.append("n/a".ljust(16))
                else:
                    row.append(
                        f"{getattr(c, km):.4f}±{getattr(c, ks):.4f}".ljust(16)
                    )
            print("".join(row))


def write_tex(cells: list[Cell], out: Path) -> None:
    idx = {(c.experiment, c.method): c for c in cells}
    exps = [e for e in EXPERIMENT_ORDER if any(c.experiment == e for c in cells)]
    methods = [m for m in METHOD_ORDER if any(c.method == m for c in cells)]

    def block(metric_mean_key: str, metric_std_key: str, comment: str) -> list[str]:
        lines = []
        lines.append(f"% --- {comment} ---")
        lines.append(r"\begin{tabular}{@{}l " +
                     " ".join(["c"] * len(exps)) + r"@{}}")
        lines.append(r"\toprule")
        lines.append("Method & " +
                     " & ".join(COLUMN_HEADER.get(e, e) for e in exps) +
                     r" \\")
        lines.append(r"\midrule")
        for m in methods:
            row = []
            for e in exps:
                c = idx.get((e, m))
                if c is None:
                    row.append("{\\% TODO}")
                else:
                    mv = getattr(c, metric_mean_key)
                    sv = getattr(c, metric_std_key)
                    row.append(f"${mv:.3f}\\pm{sv:.3f}$")
            label = DISPLAY[m]
            if m == "sap_peft":
                label = f"\\textbf{{{label}}}"
            lines.append(label + " & " + " & ".join(row) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")
        return lines

    out_lines = []
    out_lines.extend(block("kappa_mean", "kappa_std", "kappa"))
    out_lines.extend(block("mcc_mean", "mcc_std", "MCC"))
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {out}")


def sanity_check(cells: list[Cell]) -> None:
    """Print warnings for missing methods or wrong seed counts."""
    idx = {(c.experiment, c.method): c for c in cells}
    problems = []
    for e in EXPERIMENT_ORDER:
        for m in METHOD_ORDER:
            c = idx.get((e, m))
            if c is None:
                problems.append(f"  MISSING: {e} / {m}")
            elif c.n_seeds != 5:
                problems.append(
                    f"  {e} / {m}: only {c.n_seeds} seeds {c.seeds}"
                )
    if problems:
        print()
        print("=" * 80)
        print("SANITY CHECK: missing or incomplete cells")
        print("=" * 80)
        for p in problems:
            print(p)
        print()
    else:
        print()
        print("sanity check: all 54 cells present with 5 seeds each")


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments-root", type=Path,
        default=Path("experiments"),
        help="Root of the experiments/ tree (default: experiments)",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("kappa_mcc.csv"),
        help="CSV output path",
    )
    ap.add_argument(
        "--tex", type=Path, default=Path("kappa_mcc.tex"),
        help="LaTeX snippet output path",
    )
    args = ap.parse_args()

    root = args.experiments_root
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        return 1

    print(f"scanning {root} ...")
    records = discover(root)
    print(f"found {len(records)} usable experiment_results.json files")

    if not records:
        print("nothing to do.", file=sys.stderr)
        return 1

    cells = aggregate(records)
    write_csv(cells, args.output)
    write_tex(cells, args.tex)
    print_console(cells)
    sanity_check(cells)

    return 0


if __name__ == "__main__":
    sys.exit(main())