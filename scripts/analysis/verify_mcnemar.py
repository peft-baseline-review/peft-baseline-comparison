"""
Manually recompute McNemar p-values for DoRA vs SAP-PEFT on gpt2_tidg
directly from the predictions.json files, and compare against both
the pre-regen and post-regen statistical_analysis.json values.
"""
import json
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

RUN = Path("results/baselines/gpt2_tidg/2026-03-14_13-57-52")
SEEDS = [42, 43, 44, 45, 46]


def load_predictions(method, seed):
    p = RUN / method / f"seed_{seed}" / "predictions.json"
    with p.open() as f:
        d = json.load(f)
    return np.array(d["labels"]), np.array(d["predictions"])


print(f"{'seed':>5}  {'both_correct':>12}  {'dora_only':>10}  "
      f"{'sap_only':>10}  {'both_wrong':>10}  {'p_raw':>12}")
print("-" * 80)

p_values = []
for seed in SEEDS:
    yt, yp_dora = load_predictions("gpt2_dora", seed)
    _, yp_sap = load_predictions("gpt2_sap_peft", seed)

    if len(yt) != len(yp_dora) or len(yt) != len(yp_sap):
        print(f"  seed {seed}: LENGTH MISMATCH "
              f"({len(yt)}, {len(yp_dora)}, {len(yp_sap)})")
        continue

    both_correct = int(np.sum((yp_dora == yt) & (yp_sap == yt)))
    dora_only    = int(np.sum((yp_dora == yt) & (yp_sap != yt)))
    sap_only     = int(np.sum((yp_dora != yt) & (yp_sap == yt)))
    both_wrong   = int(np.sum((yp_dora != yt) & (yp_sap != yt)))

    table = [[both_correct, dora_only], [sap_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    p = float(result.pvalue)
    p_values.append(p)

    print(f"{seed:>5}  {both_correct:>12}  {dora_only:>10}  "
          f"{sap_only:>10}  {both_wrong:>10}  {p:>12.2e}")

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
if p_values:
    median = float(np.median(p_values))
    print(f"  Manual computation (this script):  {median:.2e}")
else:
    print("  Manual computation: no valid seeds — see errors above")

# Also load the stored values for comparison
print()
print("  Pre-regen file:  2.14e-01 (0.214, n.s.)")
print("  Post-regen file: 3.56e-06 (SIG)")

# Show per-seed values from the CURRENT (post-regen) file
current_file = RUN / "statistical_analysis.json"
with current_file.open() as f:
    d = json.load(f)

for t in d["pairwise_tests"]["mcnemar"]:
    names = {t["method1"], t["method2"]}
    if names == {"gpt2_dora", "gpt2_sap_peft"}:
        print()
        print("  Current statistical_analysis.json per-seed p-values:")
        for i, seed in enumerate(SEEDS):
            if i < len(t["p_values_per_seed"]):
                print(f"    seed {seed}: {t['p_values_per_seed'][i]:.2e}")
        print(f"    p_value_median: {t['p_value_median']:.2e}")
        print(f"    p_value_corrected: {t['p_value_corrected']:.2e}")
        print(f"    significant: {t['significant']}")
        break