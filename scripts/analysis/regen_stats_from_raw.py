#!/usr/bin/env python
"""
Regenerate statistical_analysis.json (plus summary CSV and markdown
report) for a baseline run, using an existing raw_results.json.

Use this when a run's raw_results.json was patched (e.g., vanilla
merged from a separate rerun) but statistical_analysis.json was
never regenerated.

Usage:
    python scripts/analysis/regen_stats_from_raw.py \
        --config configs/baselines/gpt2_sentiment_baseline.yaml \
        --run-dir results/baselines/gpt2_sentiment/2026-03-05_07-51-48 \
        --old-root "Absolute path prefix in stored predictions.json entries;"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config.baseline_config import (
    BaselineRunnerConfig, ScientificBaselineRunner, ExperimentResult,
)

EXPECTED_SEEDS = {42, 43, 44, 45, 46}


def resolve_predictions(p: str | None, old_root: str) -> str | None:
    """Return a path that exists on disk, or None."""
    if not p:
        return None
    candidates = [
        p,
        p.replace(old_root, str(_REPO_ROOT)),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Baseline config YAML for this run")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Directory containing raw_results.json")
    ap.add_argument("--old-root", default="",
                help="Absolute path prefix in stored predictions.json "
                     "entries; leave empty if paths are already valid")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found", file=sys.stderr)
        return 1

    with raw_path.open() as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} methods from {raw_path}")
    print()

    # Convert + resolve predictions paths
    results: dict[str, list[ExperimentResult]] = {}
    unresolved = 0
    for method_name in sorted(raw):
        items = []
        for s in raw[method_name]:
            s = dict(s)
            resolved = resolve_predictions(
                s.get("predictions_path"), args.old_root
            )
            if resolved is None and s.get("predictions_path"):
                unresolved += 1
            s["predictions_path"] = resolved
            # Leave config_path / experiment_dir untouched —
            # nothing in _perform_statistical_analysis reads them.
            items.append(ExperimentResult(**s))
        results[method_name] = items

        seeds = sorted({r.seed for r in items})
        missing = EXPECTED_SEEDS - set(seeds)
        status = "OK" if not missing else f"missing seeds {sorted(missing)}"
        print(f"  {method_name:30s}  {len(items)} seeds  [{status}]")

    if unresolved:
        print(f"\nWARNING: {unresolved} predictions.json files could not "
              f"be resolved. McNemar tests for those rows will be skipped.")
        print(f"  Pass --old-root to override (current: {args.old_root!r})")
        print(f"  or copy results/ into {_REPO_ROOT}")

    # Build runner, override output + results
    print(f"\nLoading config: {args.config}")
    config = BaselineRunnerConfig.from_yaml(args.config)

    runner = ScientificBaselineRunner(
        baseline_config=config,
        output_dir=str(run_dir.parent),   # we override below
        enable_mlflow=False,
        enable_monitoring=False,
    )
    runner.output_dir = run_dir
    runner.results = results
    runner.random_seeds = sorted({r.seed for items in results.values()
                                  for r in items})

    print(f"Seeds from data: {runner.random_seeds}")
    print(f"Output: {run_dir / 'statistical_analysis.json'}")
    print()

    # Run stats
    runner._perform_statistical_analysis()

    # Regenerate summary CSV, report, and plots
    print("Regenerating summary CSV, markdown report, and figures...")
    try:
        runner._generate_reports()
    except Exception as e:
        print(f"WARNING: _generate_reports failed: {e}")
        print("  statistical_analysis.json was written successfully; "
              "only the summary/report step was skipped.")

    # Verify output
    out = run_dir / "statistical_analysis.json"
    with out.open() as f:
        d = json.load(f)
    summary = d.get("performance_summary", {})
    wilcoxon = d.get("pairwise_tests", {}).get("wilcoxon", [])
    mcnemar_ = d.get("pairwise_tests", {}).get("mcnemar", [])
    frontier = d.get("pareto_frontier", {})

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"  performance_summary: {len(summary)} methods")
    has_vanilla = any('vanilla' in k for k in summary)
    print(f"    has vanilla: {has_vanilla}")
    print(f"  wilcoxon tests:  {len(wilcoxon)}")
    print(f"    involve vanilla: {sum(1 for t in wilcoxon if 'vanilla' in (t.get('method1','')+t.get('method2','')))}")
    print(f"  mcnemar tests:   {len(mcnemar_)}")
    print(f"    involve vanilla: {sum(1 for t in mcnemar_ if 'vanilla' in (t.get('method1','')+t.get('method2','')))}")
    if "hypervolume" in frontier:
        print(f"  hypervolume:     {frontier['hypervolume']:.5f}")
        print(f"  frontier members: {frontier['frontier_methods']}")

    # Key paper-facing numbers to report
    if "gpt2_vanilla" in summary:
        v = summary["gpt2_vanilla"]
        print()
        print("gpt2_vanilla in new stats:")
        print(f"  accuracy_mean      = {v['accuracy_mean']:.4f}")
        print(f"  accuracy_ci_width  = {v['accuracy_ci_width']:.4f}")
        print(f"  params_millions    = {v['params_millions']:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())