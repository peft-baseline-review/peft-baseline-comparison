"""
Recompute / annotate trainable-parameter counts for progressive PEFT runs.

Two distinct counts per seed:

    nominal_params    physical tensor footprint
                      = `param_counts.trainable` (or `model_statistics.trainable`)
                      Corresponds to `initial_rank × n_modules × module_cost
                      + classifier`. This is the peak training-memory cost.

    effective_params  post-allocation active compute
                      = sum over `rank_history[-1].allocation` of
                        rank × module_cost + classifier.
                      This is the inference-time compute.

Writes to `results.experiment_report.efficiency_metrics`:
    trainable_params_nominal
    trainable_params_effective
    trainable_params_initial_allocation     (from rank_history[0])
    trainable_params_peak_allocation        (max over rank_history)
    trainable_percentage_nominal / _effective
    pruning_ratio                           (effective / nominal)
    nominal_units, effective_units          (rank units, not params)
    dynamic_method = "progressive"
    storage_mode                            ("masked" | "shrunk" | "unknown")

Existing fields (`trainable_params`, `param_counts.trainable`) are NOT
overwritten — backward compatible with the nominal-headline convention
applied by the evaluator refactor.

Usage:
    python scripts/recompute_progressive_params.py \
        --manifest scripts/manifest_progressive.txt \
        --root . --dry-run --verify-checkpoints

    python scripts/recompute_progressive_params.py \
        --manifest scripts/manifest_progressive.txt \
        --root . --apply --verify-checkpoints --strict
"""

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    print("ERROR: torch is required.", file=sys.stderr)
    sys.exit(1)


# =====================================================================
# 1. Config path resolution (handles both JSON layouts)
# =====================================================================

def get_config_block(d: dict) -> Optional[dict]:
    """Try both known JSON locations for the experiment config."""
    for path in (
        ("results", "experiment_report", "configuration"),   # current canonical
        ("results", "configuration"),                        # legacy fallback
        ("configuration",),                                  # oldest fallback
    ):
        node = d
        try:
            for key in path:
                node = node[key]
            if isinstance(node, dict):
                return node
        except (KeyError, TypeError):
            continue
    return None


def get_rank_history(d: dict) -> Optional[list]:
    """Try both known JSON locations for rank_history."""
    for path in (
        ("results", "experiment_report", "additional_info", "rank_history"),
        ("results", "experiment_report", "rank_history"),
        ("results", "training_history", "rank_history"),
    ):
        node = d
        try:
            for key in path:
                node = node[key]
            if isinstance(node, list) and node:
                return node
        except (KeyError, TypeError):
            continue
    return None


def normalize_method(raw) -> str:
    """Lowercase, enum-prefix-stripped method name."""
    if isinstance(raw, dict):
        raw = raw.get("value") or raw.get("name") or ""
    s = str(raw).lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.strip()


# =====================================================================
# 2. Module cost map (architecture-aware)
# =====================================================================

def build_module_costs(hidden: int, ffn: int, architecture: str) -> dict[str, int]:
    """
    Longest-substring match wins. Each value is (in_features + out_features)
    for the LoRA A/B pair.
    """
    arch = (architecture or "").lower()

    if "gpt2" in arch or "gpt-2" in arch:
        return {
            "c_attn":       hidden + 3 * hidden,   # fused QKV
            "attn.c_proj":  2 * hidden,
            "mlp.c_fc":     hidden + ffn,
            "mlp.c_proj":   ffn + hidden,
        }

    # bert / roberta / xlm-roberta share these
    return {
        "query":                  2 * hidden,
        "key":                    2 * hidden,
        "value":                  2 * hidden,
        "attention.output.dense": 2 * hidden,
        "out_proj":               2 * hidden,
        "intermediate.dense":     hidden + ffn,
        "output.dense":           ffn + hidden,
    }


def module_cost(module_name: str, costs: dict[str, int]) -> Optional[int]:
    """Longest-substring match wins."""
    for key in sorted(costs.keys(), key=len, reverse=True):
        if key in module_name:
            return costs[key]
    return None


# =====================================================================
# 3. Classifier head param count (architecture-aware)
# =====================================================================

def classifier_param_count(hidden: int, num_labels: int, architecture: str) -> int:
    arch = (architecture or "").lower()

    # RoBERTa / XLM-RoBERTa: RobertaClassificationHead
    #   dense:    (hidden, hidden) + bias(hidden)
    #   out_proj: (num_labels, hidden) + bias(num_labels)
    if "roberta" in arch:
        return hidden * hidden + hidden + hidden * num_labels + num_labels

    # DistilBERT: pre_classifier + classifier
    if "distilbert" in arch:
        return hidden * hidden + hidden + hidden * num_labels + num_labels

    # GPT-2: score layer, no bias
    if "gpt2" in arch or "gpt-2" in arch:
        return hidden * num_labels

    # BERT / similar: single Linear with bias
    return hidden * num_labels + num_labels


# =====================================================================
# 4. Physical checkpoint inspection (verification only)
# =====================================================================

def load_state(ckpt_path: Path) -> Optional[dict]:
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"    [WARN] could not load {ckpt_path.name}: {e}", file=sys.stderr)
        return None
    return ckpt.get("model_state_dict", {}) or None


def lora_physical_params(ckpt_path: Path) -> Optional[int]:
    """Sum of unique LoRA tensors — dedup by canonical path so PEFT's
    modules_to_save copies (which prefix with `base_model.model.`) do not
    get counted twice."""
    state = load_state(ckpt_path)
    if not state:
        return None
    tags = ("lora_A", "lora_B", "lora_E", "lora_magnitude",
            "lora_embedding_A", "lora_embedding_B")
    seen = set()
    total = 0
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if not any(tag in k for tag in tags):
            continue
        canonical = k.replace("base_model.model.", "").replace(".modules_to_save.default", "")
        if canonical in seen:
            continue
        seen.add(canonical)
        total += v.numel()
    return total


def classifier_from_state(ckpt_path: Path) -> Optional[int]:
    state = load_state(ckpt_path)
    if not state:
        return None
    total = 0
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        if ("classifier" in k or "pre_classifier" in k
                or k.endswith("score.weight")
                or ".score.modules_to_save" in k):
            total += v.numel()
    return total if total else None


# =====================================================================
# 5. Recomputation for one seed
# =====================================================================

def sum_allocation(alloc: dict[str, int], costs: dict[str, int]):
    """Return (sum_rank_cost, unknown_modules)."""
    total = 0
    unknown = []
    for mod_name, rank in alloc.items():
        cost = module_cost(mod_name, costs)
        if cost is None:
            unknown.append(mod_name)
            continue
        total += int(rank) * cost
    return total, unknown


def recompute_seed(
    json_path: Path,
    dry_run: bool,
    verify_checkpoints: bool,
) -> Optional[dict]:
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [WARN] cannot load {json_path.name}: {e}", file=sys.stderr)
        return None

    # --- Config + method gate ----------------------------------------
    config = get_config_block(d)
    if config is None:
        return {"path": str(json_path), "skip_reason": "config block not found"}

    raw_method = (config.get("peft") or {}).get("method")
    method = normalize_method(raw_method)
    if method not in ("progressive", "adalora", "lora", "dora"):
        return {"path": str(json_path),
                "skip_reason": f"unsupported method: {raw_method!r}"}

    # --- Model + task dims -------------------------------------------
    try:
        model_cfg  = config["model"]
        hidden     = int(model_cfg["hidden_size"])
        ffn        = int(model_cfg["intermediate_size"])
        arch       = model_cfg["architecture"]
        task_cfg   = config["task"]
        num_labels = int(task_cfg["num_labels"])
    except KeyError as e:
        return {"path": str(json_path), "skip_reason": f"missing config key {e}"}

    costs      = build_module_costs(hidden, ffn, arch)
    classifier = classifier_param_count(hidden, num_labels, arch)

    results  = d.get("results", {}) or {}
    eff_dict = results.get("efficiency_metrics", {}) or {}

    # --- Nominal totals from param_counts (universal) ----------------
    nominal_params_from_pc = None
    nominal_source = None
    for source_path, src_label in (
        (("param_counts",     "trainable"), "param_counts"),
        (("model_statistics", "trainable"), "model_statistics"),
    ):
        node = results
        try:
            for k in source_path:
                node = node[k]
            nominal_params_from_pc = int(node)
            nominal_source = src_label
            break
        except (KeyError, TypeError, ValueError):
            continue

    nominal_total_params = None
    for source_path, _ in (
        (("param_counts",     "total"), None),
        (("model_statistics", "total"), None),
    ):
        node = results
        try:
            for k in source_path:
                node = node[k]
            nominal_total_params = int(node)
            break
        except (KeyError, TypeError, ValueError):
            continue

    old_trainable = int(eff_dict.get("trainable_params", 0) or 0)

    # --- Method-specific extraction ----------------------------------
    rh: Optional[list] = None
    unknown_modules: set[str] = set()
    adalora_unit_pruning_ratio = None
    storage_mode_default = "unknown"

    if method == "progressive":
        rh = get_rank_history(d)
        if not rh:
            return {"path": str(json_path), "skip_reason": "rank_history missing/empty"}

        per_epoch, per_epoch_units = [], []
        for entry in rh:
            alloc = entry.get("allocation") or {}
            if not alloc:
                continue
            lora_sum, unknown = sum_allocation(alloc, costs)
            unknown_modules.update(unknown)
            per_epoch.append(lora_sum + classifier)
            per_epoch_units.append(sum(int(r) for r in alloc.values()))

        if not per_epoch:
            return {"path": str(json_path), "skip_reason": "no usable allocation entries"}

        initial_params   = per_epoch[0]
        peak_allocated   = max(per_epoch)
        effective_params = per_epoch[-1]
        initial_units    = per_epoch_units[0]
        peak_units       = max(per_epoch_units)
        effective_units  = per_epoch_units[-1]

        nominal_params = nominal_params_from_pc
        if nominal_params is None:
            # fall back to initial_rank × n_modules
            try:
                init_rank = int(config["peft"].get("initial_rank")
                                or config["peft"].get("rank") or 8)
                total_lora = sum(init_rank * (module_cost(m, costs) or 0)
                                 for m in rh[0]["allocation"].keys())
                if total_lora > 0:
                    nominal_params = total_lora + classifier
                    nominal_source = f"inferred(initial_rank={init_rank})"
            except Exception:
                pass

    elif method in ("adalora", "adalora_fixed"):
        # AdaLoRA has no rank_history. Read from efficiency_metrics.
        #
        # Two cases:
        #   (a) Post-refactor JSON  → has `trainable_params_nominal` +
        #                              `trainable_params_effective` explicitly
        #   (b) Pre-refactor JSON   → only has `trainable_params` and
        #                              `param_counts.trainable`
        #
        # AdaLoRA-Fixed (init_r == target_r) was run before the refactor,
        # so its JSONs are case (b). Because no pruning happens in that
        # configuration, nominal == effective, and the pre-refactor
        # `param_counts.trainable` is the exact nominal value.
        nominal_eff   = eff_dict.get("trainable_params_nominal")
        effective_eff = (eff_dict.get("trainable_params_effective")
                         or eff_dict.get("trainable_params"))

        nominal_source_local = "efficiency_metrics"

        # --- Pre-refactor fallback -----------------------------------
        if nominal_eff is None and nominal_params_from_pc is not None:
            nominal_eff = nominal_params_from_pc
            nominal_source_local = "param_counts(pre-refactor)"
            if effective_eff is None:
                # For fixed rank this is correct; for dynamic it would be
                # wrong. The sanity check below warns if that ever happens.
                effective_eff = nominal_params_from_pc

            # Safety net: if the config claims pruning but we are falling
            # back to pre-refactor data, warn loudly.
            peft_cfg = config.get("peft", {}) or {}
            init_r = peft_cfg.get("adalora_init_rank") or peft_cfg.get("initial_rank")
            tgt_r  = peft_cfg.get("adalora_target_rank") or peft_cfg.get("rank")
            if (init_r is not None and tgt_r is not None
                    and int(init_r) != int(tgt_r)):
                print(f"    [WARN] {json_path.name}: AdaLoRA config has "
                      f"init_r={init_r} != target_r={tgt_r}, but JSON lacks "
                      f"nominal/effective fields. The fallback will report "
                      f"nominal=effective which is probably WRONG. "
                      f"Consider re-running the evaluator on this JSON.",
                      file=sys.stderr)

        if nominal_eff is None or effective_eff is None:
            return {"path": str(json_path),
                    "skip_reason": "adalora: nominal/effective missing in JSON "
                                   "(run the evaluator refactor first)"}

        nominal_params   = int(nominal_eff)
        effective_params = int(effective_eff)
        nominal_source   = nominal_source_local

        # AdaLoRA only shrinks (init_r → target_r): initial = peak = nominal
        initial_params = nominal_params
        peak_allocated = nominal_params

        initial_units   = eff_dict.get("adalora_nominal_units")
        peak_units      = eff_dict.get("adalora_nominal_units")
        effective_units = eff_dict.get("adalora_active_units")

        adalora_unit_pruning_ratio = eff_dict.get("adalora_pruning_ratio")

        # --- Fixed vs Dynamic detection ------------------------------
        # AdaLoRA-Fixed: init_r == target_r, no pruning happens.
        # Treat as "static" for storage purposes (checkpoint_ratio = 1.0,
        # no masking tax) even though the code path uses mask machinery
        # internally with rank_pattern = all-1s.
        if nominal_params == effective_params:
            storage_mode_default = "static"
            if adalora_unit_pruning_ratio is None:
                adalora_unit_pruning_ratio = 1.0
        else:
            storage_mode_default = "masked"

    elif method in ("lora", "dora"):
        # Fixed-rank adapter — no rank_history, no pruning, no growth.
        # Adapter is fixed at configured rank from start to end of training.
        tp = eff_dict.get("trainable_params")
        if tp is None:
            return {"path": str(json_path),
                    "skip_reason": f"{method}: trainable_params missing in JSON"}
        tp = int(tp)

        # Static: nominal == effective == initial == peak
        nominal_params   = tp
        effective_params = tp
        initial_params   = tp
        peak_allocated   = tp

        # Rank units: sum(rank × n_modules), but we lack rank_history.
        # Leave as None; downstream handles None gracefully.
        initial_units = peak_units = effective_units = None

        adalora_unit_pruning_ratio = None
        storage_mode_default = "static"     # NEW: distinct from shrunk/masked
        rh = None
        unknown_modules = set()

    if nominal_params is None:
        return {"path": str(json_path),
                "skip_reason": "cannot determine nominal params"}

    # --- Shipped total = base + effective (works for masked and shrunk) -
    # For masked AdaLoRA, efficiency_metrics.total_params is base + nominal;
    # for shrunk progressive it is base + effective. Normalise to base + effective
    # so the percentage denominator is the same "inference-time model size".
    if nominal_total_params and nominal_params and effective_params is not None:
        base_params          = nominal_total_params - nominal_params
        shipped_total_params = base_params + effective_params
    else:
        shipped_total_params = int(eff_dict.get("total_params", 0) or 0)

    def pct_effective(x):
        return (x / shipped_total_params * 100.0) if shipped_total_params else None

    def pct_nominal(x):
        return (x / nominal_total_params * 100.0) if nominal_total_params else None

    def comp_effective(x):
        return (shipped_total_params / x) if x else None

    def comp_nominal(x):
        return (nominal_total_params / x) if (x and nominal_total_params) else None

    pruning_ratio    = (effective_params / nominal_params) if nominal_params else None
    allocation_ratio = (effective_params / peak_allocated) if peak_allocated else None

    # --- Checkpoint verification (guarded for AdaLoRA) ----------------
    storage_mode = storage_mode_default
    verification = ""
    if verify_checkpoints:
        ckpt_path = None
        for candidate in (
            json_path.parent / "checkpoints" / "best_model.pt",
            json_path.parent.parent / "checkpoints" / "best_model.pt",
        ):
            if candidate.exists():
                ckpt_path = candidate
                break

        if ckpt_path:
            ckpt_epoch = None
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ckpt_epoch = ckpt.get("epoch", None)
            except Exception:
                ckpt = None

            phys_lora = lora_physical_params(ckpt_path)
            if phys_lora is not None:
                physical_total = phys_lora + classifier

                # --- Method-type dispatch (takes precedence over physical checks) ---
                # LoRA/DoRA never change rank → static
                # AdaLoRA always masks (even when init_r == target_r) → masked
                # Progressive changes rank dynamically → shrunk (checked below)
                tol_static = max(64, int(0.001 * max(effective_params, 1)))

                if method in ("lora", "dora"):
                    if abs(physical_total - effective_params) <= tol_static:
                        storage_mode = "static"
                        verification = f"physical~=effective ({physical_total:,})"
                    else:
                        storage_mode = "ambiguous"
                        verification = (
                            f"physical={physical_total:,} != "
                            f"effective={effective_params:,}"
                        )
                    # Skip rest of the physical checks for static methods
                    per_epoch_exp = []
                    physical_total = None

                elif method == "adalora":
                    # AdaLoRA is always masked — even at zero pruning range
                    if abs(physical_total - nominal_params) <= tol_static:
                        storage_mode = "masked"
                        verification = f"physical~=nominal ({physical_total:,})"
                    else:
                        storage_mode = "ambiguous"
                        verification = (
                            f"physical={physical_total:,} != "
                            f"nominal={nominal_params:,}"
                        )
                    per_epoch_exp = []
                    physical_total = None

                # --- Progressive methods fall through to the existing dispatch ---
                per_epoch_exp = []

                if physical_total is not None:              # NEW guard
                    if rh:
                        for entry in rh:
                            alloc = entry.get("allocation") or {}
                            if not alloc:
                                continue
                            lora_sum, _ = sum_allocation(alloc, costs)
                            per_epoch_exp.append({
                                "epoch": entry.get("epoch"),
                                "params": lora_sum + classifier,
                            })

                    matched_epoch = None
                    if ckpt_epoch is not None and per_epoch_exp:
                        for e in per_epoch_exp:
                            if e["epoch"] in (ckpt_epoch, ckpt_epoch + 1):
                                if abs(physical_total - e["params"]) <= 64:
                                    matched_epoch = e["epoch"]
                                    break

                    tol = max(64, int(0.0005 * (nominal_params - classifier)))

                    if matched_epoch is not None:
                        storage_mode = "shrunk"
                        verification = f"physical~=rank_history[epoch={matched_epoch}] ({physical_total:,})"
                    elif abs(physical_total - effective_params) <= tol:
                        storage_mode = "shrunk"
                        verification = f"physical~=effective ({physical_total:,})"
                    elif abs(physical_total - nominal_params) <= tol:
                        storage_mode = "masked"
                        verification = f"physical~=nominal ({physical_total:,})"
                    elif per_epoch_exp:
                        params_observed = [e["params"] for e in per_epoch_exp]
                        pmin = min(params_observed)
                        pmax = max(params_observed)
                        if pmin - 64 <= physical_total <= pmax + 64:
                            storage_mode = "shrunk"
                            verification = (f"physical={physical_total:,} in rank-history range "
                                            f"[{pmin:,}, {pmax:,}] (best ckpt @ intermediate epoch)")
                        else:
                            storage_mode = "ambiguous"
                            verification = f"physical={physical_total:,} outside [{pmin:,}, {pmax:,}]"
                    else:
                        storage_mode = "ambiguous"
                        verification = (f"physical={physical_total:,} matches neither "
                                        f"nominal nor effective")
            else:
                verification = "checkpoint unreadable"
        else:
            verification = "checkpoint missing"

    # Post-hoc: AdaLoRA-Fixed is functionally static for storage purposes —
    # no pruning happened, so physical == nominal == effective. Report it
    # as static to make checkpoint_ratio = 1.0 unambiguous in Table 4.
    if (method in ("adalora", "adalora_fixed")
            and nominal_params == effective_params
            and storage_mode == "masked"):
        storage_mode = "static"
        if verification:
            verification = verification.replace("~=nominal", "~=effective")

    # --- Info dict ---------------------------------------------------
    info = {
        "path":            _canonical_name(str(json_path)),
        "run_id":          json_path.parent.name,
        "seed": (
            d.get("experiment_metadata", {}).get("seed")
            or d.get("experiment_metadata", {}).get("reproducibility_info", {}).get("seed")
            or config.get("seed") or "?"
        ),
        "method":          method,
        "dynamic_method":  method,
        "architecture":    arch,
        "hidden":          hidden,
        "ffn":             ffn,
        "num_labels":      num_labels,
        "classifier":      classifier,

        "nominal_source":            nominal_source,
        "nominal_params":            nominal_params,
        "first_logged_alloc_params": initial_params,
        "initial_alloc_params":      initial_params,
        "peak_alloc_params":         peak_allocated,
        "effective_params":          effective_params,

        "nominal_units":    None,
        "initial_units":    int(initial_units)   if initial_units   is not None else None,
        "peak_units":       int(peak_units)      if peak_units      is not None else None,
        "effective_units":  int(effective_units) if effective_units is not None else None,

        "pruning_ratio":    round(pruning_ratio, 4)    if pruning_ratio    else None,
        "allocation_ratio": round(allocation_ratio, 4) if allocation_ratio else None,
        "adalora_unit_pruning_ratio": adalora_unit_pruning_ratio,

        "old_trainable": old_trainable,
        "delta_effective_vs_old": (
            round(100.0 * (effective_params - old_trainable) / old_trainable, 2)
            if old_trainable else None
        ),

        "nominal_pct":   round(pct_nominal(nominal_params),     4) if pct_nominal(nominal_params)     else None,
        "effective_pct": round(pct_effective(effective_params), 4) if pct_effective(effective_params) else None,

        "nominal_total_params": nominal_total_params,
        "shipped_total_params": shipped_total_params,

        "storage_mode":  storage_mode,
        "verification":  verification,
        "unknown_modules": sorted(unknown_modules),
    }

    info["params_ratio"] = info["pruning_ratio"]
    info["is_growth"]    = (pruning_ratio is not None and pruning_ratio > 1.0)
    info["is_pruned"]    = (pruning_ratio is not None and pruning_ratio < 1.0)

        # Adapter-only pruning ratio (excludes classifier) — cross-method-comparable
    classifier_used = classifier
    nominal_adapter   = nominal_params   - classifier_used
    effective_adapter = effective_params - classifier_used
    info["adapter_pruning_ratio"] = (
        round(effective_adapter / nominal_adapter, 4)
        if nominal_adapter > 0 else None
    )
    info["nominal_adapter_params"]   = nominal_adapter
    info["effective_adapter_params"] = effective_adapter

    # nominal_units
    if method == "progressive" and rh:
        try:
            n_modules_phys = len(rh[0]["allocation"])
            init_rank_cfg  = int(config["peft"].get("initial_rank")
                                 or config["peft"].get("rank") or 0)
            if init_rank_cfg and n_modules_phys:
                info["nominal_units"] = init_rank_cfg * n_modules_phys
        except Exception:
            pass
    elif method == "adalora":
        info["nominal_units"] = initial_units   # = adalora_nominal_units

    # checkpoint on disk
    if storage_mode == "shrunk":
        info["checkpoint_size_params"] = int(effective_params)
    elif storage_mode == "masked":
        info["checkpoint_size_params"] = int(nominal_params)
    elif storage_mode == "static":
        info["checkpoint_size_params"] = int(effective_params)   # = nominal
    else:
        info["checkpoint_size_params"] = None

    # In recompute_seed, replace the checkpoint_size_pct calculation
    if info["checkpoint_size_params"]:
        on_disk_total = (
            nominal_total_params - nominal_params + info["checkpoint_size_params"]
        )
        info["checkpoint_size_pct"] = (
            round(info["checkpoint_size_params"] / on_disk_total * 100.0, 4)
            if on_disk_total else None
        )
    else:
        info["checkpoint_size_pct"] = None
        
    info["checkpoint_ratio"] = (
        round(info["checkpoint_size_params"] / nominal_params, 4)
        if (info["checkpoint_size_params"] is not None and nominal_params)
        else None
    )

    # --- Write-back --------------------------------------------------
    if not dry_run and not unknown_modules:
        if shipped_total_params <= 0:
            print(f"    [WARN] {json_path.name}: shipped_total_params="
                  f"{shipped_total_params}; skipping write", file=sys.stderr)
        else:
            eff_dict["trainable_params_nominal"]       = int(nominal_params)
            eff_dict["trainable_percentage_nominal"]   = pct_nominal(nominal_params)
            eff_dict["trainable_params_effective"]     = int(effective_params)
            eff_dict["trainable_percentage_effective"] = pct_effective(effective_params)

            if method == "progressive":
                eff_dict["trainable_params_initial_allocation"] = int(initial_params)
                eff_dict["trainable_params_peak_allocation"]    = int(peak_allocated)

            eff_dict["compression_ratio_nominal"]   = comp_nominal(nominal_params)
            eff_dict["compression_ratio_effective"] = comp_effective(effective_params)

            eff_dict["nominal_total_params"] = nominal_total_params
            eff_dict["shipped_total_params"] = shipped_total_params

            eff_dict["pruning_ratio"]    = pruning_ratio
            eff_dict["allocation_ratio"] = allocation_ratio

            eff_dict["dynamic_method"] = method           # was hardcoded "progressive"
            eff_dict["storage_mode"]   = storage_mode

            eff_dict["adapter_pruning_ratio"]     = info["adapter_pruning_ratio"]
            eff_dict["nominal_adapter_params"]    = nominal_adapter
            eff_dict["effective_adapter_params"]  = effective_adapter

            if info["checkpoint_size_params"] is not None:
                eff_dict["checkpoint_size_params"] = int(info["checkpoint_size_params"])
            if info["checkpoint_size_pct"] is not None:
                eff_dict["checkpoint_size_pct"] = pct_nominal(info["checkpoint_size_params"])

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, default=str)

    return info

import unicodedata

_HYPHEN_VARIANTS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

def _canonical_name(name: str) -> str:
    """NFC-normalize and unify all hyphen variants to ASCII '-'."""
    n = unicodedata.normalize("NFC", name)
    for h in _HYPHEN_VARIANTS:
        n = n.replace(h, "-")
    return n


def resolve_manifest_path(entry: Path, root: Path) -> tuple[Optional[Path], Optional[str]]:
    """
    Resolve a manifest entry against the filesystem, tolerating Unicode
    normalization differences and hyphen-variant confusion.

    Returns (resolved_path, warning_message_or_None).
    Returns (None, None) if the path genuinely does not exist.
    """
    abs_path = entry if entry.is_absolute() else (root / entry)

    # Fast path: exact match
    if abs_path.exists():
        return abs_path, None

    # Slow path: walk components, matching by canonical form
    anchor    = Path(abs_path.anchor) if abs_path.is_absolute() else Path(".")
    parts     = abs_path.parts[1:] if abs_path.is_absolute() else abs_path.parts
    resolved  = anchor
    diverged  = False

    for part in parts:
        parent = resolved if resolved.is_absolute() or resolved.parts else Path(".")
        try:
            children = list(parent.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return None, None

        target   = _canonical_name(part)
        matches  = [c for c in children if _canonical_name(c.name) == target]

        if not matches:
            return None, None

        chosen = matches[0]
        if chosen.name != part:
            diverged = True
        resolved = chosen

    if resolved.exists():
        warn = (
            f"resolved via canonicalization: manifest had {entry!r}, "
            f"matched {resolved!r}"
        ) if diverged else None
        return resolved, warn

    return None, None
# =====================================================================
# 6. Manifest reader
# =====================================================================

def read_manifest(manifest_path: Path) -> list[Path]:
    """
    Read manifest file, auto-detecting UTF-8 / UTF-8-BOM / UTF-16 LE/BE
    from the leading bytes.

    Inline comments are supported: anything after a whitespace-preceded
    '#' is stripped from the path. This preserves any legitimate '#'
    that might appear mid-path (rare but not impossible) while correctly
    removing the `  # seed=N` annotations written by `generate_manifest.py`.
    """
    raw = manifest_path.read_bytes()

    # BOM detection (unchanged)
    if raw.startswith(b"\xff\xfe\x00\x00"):
        text = raw.decode("utf-32-le")
    elif raw.startswith(b"\x00\x00\xfe\xff"):
        text = raw.decode("utf-32-be")
    elif raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16-le")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16-be")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw[3:].decode("utf-8")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            print(f"[WARN] {manifest_path.name}: not UTF-8, "
                  f"falling back to latin-1", file=sys.stderr)
            text = raw.decode("latin-1")

    if text.startswith("\ufeff"):
        text = text[1:]

    paths = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Strip inline comment: ' # ...' or '\t# ...'
        for marker in (" #", "\t#"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx].rstrip()
                break

        if not line:
            continue

        paths.append(Path(line))
    return paths
# =====================================================================
# 7. CLI
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute progressive-PEFT trainable parameters."
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--report-csv", type=Path,
                    default=Path("scripts/recompute_report.csv"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="Print diffs, do not modify any JSON")
    mode.add_argument("--apply", action="store_true",
                      help="Write corrected fields back to the JSONs")
    ap.add_argument("--verify-checkpoints", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 2 if any seed has unknown modules or mismatch")
    args = ap.parse_args()

    dry_run = bool(args.dry_run)

    if not args.root.exists():
        print(f"ERROR: --root does not exist: {args.root}", file=sys.stderr)
        return 1
    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        return 1

    manifest = read_manifest(args.manifest)
    if not manifest:
        print("ERROR: manifest is empty.", file=sys.stderr)
        return 1

    # --- Resolve manifest entries (Unicode-tolerant) -------------------
    resolved_entries: list[tuple[Path, Path]] = []   # (original, resolved)
    missing: list[str] = []
    for p in manifest:
        resolved, warn = resolve_manifest_path(p, args.root)
        if resolved is None:
            missing.append(str(p))
            continue
        if warn:
            print(f"[WARN] {warn}", file=sys.stderr)
        resolved_entries.append((p, resolved))

    if missing:
        print("\nERROR: the following manifest entries do not exist:",
              file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
            # Diagnostic: show the deepest existing ancestor + available children
            abs_path = Path(m) if Path(m).is_absolute() else (args.root / m)
            current  = abs_path
            while not current.exists() and current != current.parent:
                current = current.parent
            if current != abs_path:
                rel_parts = abs_path.relative_to(current).parts
                print(f"    deepest existing: {current}", file=sys.stderr)
                print(f"    missing parts: {list(rel_parts)}", file=sys.stderr)
                try:
                    children = sorted(c.name for c in current.iterdir() if c.is_dir())
                    print(f"    available under {current}:", file=sys.stderr)
                    for c in children[:5]:
                        print(f"      {c!r}", file=sys.stderr)
                except Exception:
                    pass
        print("\nFix the manifest before continuing.", file=sys.stderr)
        return 1

    print(f"Manifest: {len(manifest)} run dirs")
    print(f"Root:     {args.root.resolve()}")
    print(f"Mode:     {'DRY RUN' if dry_run else 'APPLY'}"
          f"{' + checkpoint verification' if args.verify_checkpoints else ''}")
    print()

    rows: list[dict] = []
    n_ok = n_skipped = n_failed = 0

    for run_dir_rel, run_dir in resolved_entries:
        # Layout A: manifest entry IS a seed run dir (your layout)
        direct = run_dir / "experiment_results.json"
        seed_jsons = [direct] if direct.exists() else []
        # Layout B (legacy): parent of seed_*/ subfolders
        if not seed_jsons:
            seed_jsons = sorted(run_dir.glob("*/experiment_results.json"))
        if not seed_jsons:
            print(f"[WARN] no experiment_results.json under {run_dir_rel}",
                  file=sys.stderr)
            n_skipped += 1
            continue

        print(f"{run_dir_rel}")
        for seed_json in seed_jsons:
            info = recompute_seed(
                seed_json,
                dry_run=dry_run,
                verify_checkpoints=args.verify_checkpoints,
            )
            if info is None:
                print(f"  [SKIP] {seed_json.parent.name} (unreadable JSON)")
                n_skipped += 1
                continue
            if "skip_reason" in info:
                print(f"  [SKIP] {seed_json.parent.name} ({info['skip_reason']})")
                n_skipped += 1
                continue

            delta = info["delta_effective_vs_old"]
            delta_str = f"{delta:+.1f}%" if delta is not None else "?"
            ratio_str = (f"{info['pruning_ratio']:.3f}"
                         if info["pruning_ratio"] is not None else "?")

            ckpt_str = (
                f", ckpt={info['checkpoint_size_params']:,}"
                if info.get("checkpoint_size_params") else ""
            )

            print(f"  seed {info['seed']}: "
                  f"nominal={info['nominal_params']:,}  "
                  f"peak={info['peak_alloc_params']:,}  "
                  f"eff={info['effective_params']:,}{ckpt_str}  "
                  f"(ratio={ratio_str}, Δeff_vs_old={delta_str})"
                  + (f"  [{info['verification']}]" if info["verification"] else ""))

            if info["unknown_modules"]:
                print(f"    [ERROR] {len(info['unknown_modules'])} unmatched "
                      f"module(s) — recompute is an UNDERESTIMATE; not written:")
                for m in info["unknown_modules"][:3]:
                    print(f"      {m}")
                if len(info["unknown_modules"]) > 3:
                    print(f"      … and {len(info['unknown_modules']) - 3} more")
                n_failed += 1
            else:
                n_ok += 1

            if info["storage_mode"] == "ambiguous":
                print(f"  [WARN] physical checkpoint outside rank-history range")

            rows.append(info)

    # CSV report
    if rows:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id", "path", "seed", "method", "architecture", "hidden", "ffn",
            "num_labels", "classifier",
            "nominal_source", "nominal_params",
            "first_logged_alloc_params", "initial_alloc_params",
            "peak_alloc_params", "effective_params",
            "initial_units", "peak_units", "effective_units", "nominal_units",
            "pruning_ratio", "params_ratio", "is_growth", "is_pruned",
            "nominal_adapter_params", "effective_adapter_params",
            "allocation_ratio", "adalora_unit_pruning_ratio",
            "old_trainable", "delta_effective_vs_old",
            "nominal_pct", "effective_pct",
            "nominal_total_params", "shipped_total_params",
            "checkpoint_size_params", "checkpoint_size_pct", "checkpoint_ratio",
            "storage_mode", "verification", "unknown_modules",
        ]
        # was: encoding="utf-8"
        with open(args.report_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                r = dict(row)
                r["unknown_modules"] = ";".join(r.get("unknown_modules") or [])
                w.writerow(r)
        print(f"\nReport: {args.report_csv}")

    print(f"\nSummary: processed={n_ok}, skipped={n_skipped}, failed={n_failed}")
    if dry_run:
        print("(dry run — no files were written)")

    if args.strict and (n_failed > 0 or n_skipped > 0):
        return 2
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())