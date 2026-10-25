#!/usr/bin/env python
"""
Significance-count summary for the six baseline experiments.

Reads statistical_analysis.json for each experiment and reports McNemar
significance counts with Holm-Bonferroni correction. Splits results by
in-distribution vs out-of-distribution backbones, matching the
three-regime framing of the manuscript (Section 5.11).

Also flags suspicious p-values: exact 0.0 values indicate that the
experiment's statistical_analysis.json was written before the McNemar
b+c==0 fix in baseline_config.py. Such files should be regenerated.

Usage:
    python scripts/analysis/final_sig_counts.py
    python scripts/analysis/final_sig_counts.py --verbose
    python scripts/analysis/final_sig_counts.py --expect-indist-peft 0
    python scripts/analysis/final_sig_counts.py --expect-ood-peft 21
    python scripts/analysis/final_sig_counts.py --expect-no-suspicious

Exit code:
    0   diagnosis succeeded (all requested assertions passed)
    1   diagnosis succeeded but an assertion failed
    2   no stats files were found
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment catalog
# ---------------------------------------------------------------------------
# (experiment_dir_name, regime)
#   regime = "in"  for in-distribution backbones (TiroBERTa, Afro-XLM-R)
#   regime = "ood" for out-of-distribution backbone (GPT-2)
EXPERIMENTS = [
    ("tiroberta_tidg",      "in"),
    ("tiroberta_sentiment", "in"),
    ("afro_xlmr_tidg",      "in"),
    ("afro_xlmr_sentiment", "in"),
    ("gpt2_tidg",           "ood"),
    ("gpt2_sentiment",      "ood"),
]

# Suffix patterns for each PEFT method. We match on ``_{suffix}`` at the
# end of the full method name, which works regardless of whether the
# prefix is ``tiroberta_tidg_``, ``afro_xlmr_``, or ``gpt2_``. This is
# the fix for the earlier diagnostic that tried to strip the full
# experiment name and silently missed 4 of the 6 experiments.
PEFT_SUFFIXES = {
    "lora":                 "LoRA",
    "dora":                 "DoRA",
    "adalora":              "AdaLoRA",
    "progressive_adaptive": "Prog-Adaptive",
    "progressive_linear":   "Prog-Linear",
    "progressive_fixed":    "Prog-Fixed",
    "sap_peft":             "SAP-PEFT",
}

# Timestamp directory format used by the baseline runner.
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def is_peft(name: str) -> str | None:
    """Return the short display name if ``name`` ends with a known PEFT
    suffix, else None."""
    for suffix, display in PEFT_SUFFIXES.items():
        if name.endswith(f"_{suffix}"):
            return display
    return None


def is_vanilla(name: str) -> bool:
    return name.endswith("_vanilla")


def is_full_finetuning(name: str) -> bool:
    return name.endswith("_full_finetuning")


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------

def find_stats_file(exp_dir: Path) -> Path | None:
    """Return the oldest timestamped run's statistical_analysis.json."""
    if not exp_dir.is_dir():
        return None
    candidates = sorted(
        p for p in exp_dir.rglob("statistical_analysis.json")
        if TIMESTAMP_RE.match(p.parent.name)
    )
    if not candidates:
        return None
    return candidates[0]


def count_significance(stats: dict) -> dict:
    """Categorize McNemar pairs into three buckets and count SIG."""
    out = {
        "peft_vs_vanilla": {"sig": 0, "total": 0, "sig_pairs": []},
        "peft_vs_peft":    {"sig": 0, "total": 0, "sig_pairs": []},
        "fullft_vs_peft":  {"sig": 0, "total": 0, "sig_pairs": []},
        "suspicious":      [],  # pairs with p_median == 0.0
    }

    for t in stats.get("pairwise_tests", {}).get("mcnemar", []):
        m1 = t["method1"]
        m2 = t["method2"]
        p1 = is_peft(m1)
        p2 = is_peft(m2)

        if p1 and p2:
            cat = "peft_vs_peft"
        elif (p1 and is_vanilla(m2)) or (p2 and is_vanilla(m1)):
            cat = "peft_vs_vanilla"
        elif (p1 and is_full_finetuning(m2)) or (p2 and is_full_finetuning(m1)):
            cat = "fullft_vs_peft"
        else:
            continue

        out[cat]["total"] += 1
        if t.get("significant"):
            out[cat]["sig"] += 1
            out[cat]["sig_pairs"].append((m1, m2, t["p_value_corrected"]))

        # Suspicious: exact 0.0 anywhere in the per-seed or median
        # p-values. This is the fingerprint of the pre-fix McNemar
        # b+c==0 bug.
        is_susp = (
            t.get("p_value_median") == 0.0
            or any(p == 0.0 for p in t.get("p_values_per_seed", []))
        )
        if is_susp:
            out["suspicious"].append((m1, m2, t.get("p_value_median")))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="List individual significant pairs")
    ap.add_argument("--base-dir", type=Path, default=Path("results/baselines"),
                    help="Root containing per-experiment subdirs "
                         "(default: results/baselines)")
    ap.add_argument("--expect-indist-peft", type=int, default=None,
                    help="Assert in-distribution PEFT-vs-PEFT SIG count")
    ap.add_argument("--expect-ood-peft", type=int, default=None,
                    help="Assert OOD PEFT-vs-PEFT SIG count")
    ap.add_argument("--expect-no-suspicious", action="store_true",
                    help="Exit nonzero if any p==0.0 is found")
    args = ap.parse_args()

    # Load all stats files
    results = {}
    missing = []
    for exp, regime in EXPERIMENTS:
        stats_file = find_stats_file(args.base_dir / exp)
        if stats_file is None:
            missing.append(exp)
            continue
        stats = json.load(stats_file.open())
        results[exp] = {
            "regime": regime,
            "path": stats_file,
            "counts": count_significance(stats),
        }

    if missing:
        print(f"WARNING: no statistical_analysis.json found for: "
              f"{', '.join(missing)}", file=sys.stderr)
    if not results:
        print("ERROR: nothing to report.", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    print()
    print("=" * 82)
    print("McNemar significance counts (Holm-corrected, alpha=0.05)")
    print("=" * 82)
    header = (f"{'Experiment':28s}  "
              f"{'PEFT-vs-vanilla':>16s}  "
              f"{'PEFT-vs-PEFT':>12s}  "
              f"{'FullFT-vs-PEFT':>14s}")
    print(header)
    print("-" * 82)

    aggregate = {
        "in":  {"peft_vs_vanilla": [0, 0], "peft_vs_peft": [0, 0],
                "fullft_vs_peft":  [0, 0]},
        "ood": {"peft_vs_vanilla": [0, 0], "peft_vs_peft": [0, 0],
                "fullft_vs_peft":  [0, 0]},
    }

    for exp, regime in EXPERIMENTS:
        if exp not in results:
            continue
        c = results[exp]["counts"]
        row = (f"{exp:28s}  "
               f"{c['peft_vs_vanilla']['sig']:>3d}/{c['peft_vs_vanilla']['total']:<3d}         "
               f"{c['peft_vs_peft']['sig']:>3d}/{c['peft_vs_peft']['total']:<3d}      "
               f"{c['fullft_vs_peft']['sig']:>3d}/{c['fullft_vs_peft']['total']:<3d}")
        print(row)

        for cat in ("peft_vs_vanilla", "peft_vs_peft", "fullft_vs_peft"):
            aggregate[regime][cat][0] += c[cat]["sig"]
            aggregate[regime][cat][1] += c[cat]["total"]

    print("-" * 82)
    for label, regime in [("In-distribution subtotal", "in"),
                          ("OOD subtotal", "ood")]:
        a = aggregate[regime]
        row = (f"{label:28s}  "
               f"{a['peft_vs_vanilla'][0]:>3d}/{a['peft_vs_vanilla'][1]:<3d}         "
               f"{a['peft_vs_peft'][0]:>3d}/{a['peft_vs_peft'][1]:<3d}      "
               f"{a['fullft_vs_peft'][0]:>3d}/{a['fullft_vs_peft'][1]:<3d}")
        print(row)

    tot = {"peft_vs_vanilla": [0, 0], "peft_vs_peft": [0, 0],
           "fullft_vs_peft":  [0, 0]}
    for regime in ("in", "ood"):
        for cat in tot:
            tot[cat][0] += aggregate[regime][cat][0]
            tot[cat][1] += aggregate[regime][cat][1]
    row = (f"{'All experiments':28s}  "
           f"{tot['peft_vs_vanilla'][0]:>3d}/{tot['peft_vs_vanilla'][1]:<3d}         "
           f"{tot['peft_vs_peft'][0]:>3d}/{tot['peft_vs_peft'][1]:<3d}      "
           f"{tot['fullft_vs_peft'][0]:>3d}/{tot['fullft_vs_peft'][1]:<3d}")
    print(row)
    print("=" * 82)
    print()

    # ------------------------------------------------------------------
    # Verbose: SIG pairs
    # ------------------------------------------------------------------
    if args.verbose:
        print("Significant pairs by experiment:")
        print("-" * 82)
        any_sig = False
        for exp, _ in EXPERIMENTS:
            if exp not in results:
                continue
            c = results[exp]["counts"]
            all_sig = (c["peft_vs_vanilla"]["sig_pairs"]
                       + c["peft_vs_peft"]["sig_pairs"]
                       + c["fullft_vs_peft"]["sig_pairs"])
            if not all_sig:
                print(f"  {exp}: (none)")
                continue
            any_sig = True
            print(f"  {exp}:")
            for m1, m2, p in all_sig:
                print(f"    {m1:36s} vs {m2:36s}  p_adj={p:.2e}")
        if not any_sig:
            print("  (no significant pairs on any experiment)")
        print()

    # ------------------------------------------------------------------
    # Suspicious p-values
    # ------------------------------------------------------------------
    suspicious = []
    for exp, _ in EXPERIMENTS:
        if exp not in results:
            continue
        for m1, m2, p_med in results[exp]["counts"]["suspicious"]:
            suspicious.append((exp, m1, m2, p_med))

    print("Suspicious p-values (exact 0.0 from pre-fix McNemar bug):")
    print("-" * 82)
    if not suspicious:
        print("  (none -- stats files are current)")
    else:
        for exp, m1, m2, p_med in suspicious:
            print(f"  {exp:20s}  {m1:36s} vs {m2:36s}  p_median={p_med}")
        print()
        print("  Regenerate the affected files with regen_stats_from_raw.py")
        print("  after applying the b+c==0 fix in baseline_config.py.")
    print()

    # ------------------------------------------------------------------
    # Paper-facing summary
    # ------------------------------------------------------------------
    print("=" * 82)
    print("Paper-facing numbers")
    print("=" * 82)
    in_dist_peft = aggregate["in"]["peft_vs_peft"][0]
    in_dist_peft_total = aggregate["in"]["peft_vs_peft"][1]
    ood_peft = aggregate["ood"]["peft_vs_peft"][0]
    ood_peft_total = aggregate["ood"]["peft_vs_peft"][1]

    print(f"  In-distribution PEFT-vs-PEFT:  {in_dist_peft}/{in_dist_peft_total} SIG")
    print(f"    ->  text: \"{('none' if in_dist_peft == 0 else str(in_dist_peft))}"
          f" of the {in_dist_peft_total}\"")
    print(f"  OOD PEFT-vs-PEFT:              {ood_peft}/{ood_peft_total} SIG")

    # Per-experiment PEFT-vs-vanilla pattern
    print()
    print("  PEFT-vs-vanilla per experiment:")
    for exp, _ in EXPERIMENTS:
        if exp not in results:
            continue
        c = results[exp]["counts"]["peft_vs_vanilla"]
        print(f"    {exp:22s}  {c['sig']}/{c['total']} SIG")
    print()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    exit_code = 0

    if args.expect_indist_peft is not None:
        ok = in_dist_peft == args.expect_indist_peft
        status = "PASS" if ok else "FAIL"
        print(f"Assertion: in-distribution PEFT-vs-PEFT == "
              f"{args.expect_indist_peft}  [{status}; actual={in_dist_peft}]")
        if not ok:
            exit_code = 1

    if args.expect_ood_peft is not None:
        ok = ood_peft == args.expect_ood_peft
        status = "PASS" if ok else "FAIL"
        print(f"Assertion: OOD PEFT-vs-PEFT == "
              f"{args.expect_ood_peft}  [{status}; actual={ood_peft}]")
        if not ok:
            exit_code = 1

    if args.expect_no_suspicious:
        ok = len(suspicious) == 0
        status = "PASS" if ok else "FAIL"
        print(f"Assertion: no suspicious p-values  "
              f"[{status}; found={len(suspicious)}]")
        if not ok:
            exit_code = 1

    if exit_code:
        print()
        print("Exit code: 1 (one or more assertions failed)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())