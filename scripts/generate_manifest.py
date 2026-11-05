"""
Enumerate all seed directories under each method path and emit a manifest
with one line per seed run.

Tolerates Unicode hyphen variants (U+2010..U+2015, U+2212) in directory
names - some project directories were created with non-breaking hyphens
(`-`, U+2011) that look identical to ASCII `-` but are different bytes.
"""
import json
import sys
import unicodedata
from pathlib import Path


# --- Unicode-tolerant path resolution (same logic as recompute script) ---
_HYPHEN_VARIANTS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


def _canonical_name(name: str) -> str:
    """NFC-normalize and unify all hyphen variants to ASCII '-'."""
    n = unicodedata.normalize("NFC", name)
    for h in _HYPHEN_VARIANTS:
        n = n.replace(h, "-")
    return n


def resolve_under_root(rel_path: str, root: Path) -> Path | None:
    abs_path = root / rel_path
    if abs_path.exists():
        return abs_path

    parts = Path(rel_path).parts
    resolved = root                          # ← FIX: start at root, not CWD
    for part in parts:
        try:
            children = list(resolved.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return None
        target = _canonical_name(part)
        matches = [c for c in children if _canonical_name(c.name) == target]
        if not matches:
            return None
        resolved = matches[0]

    return resolved if resolved.exists() else None


# --- Method paths (ASCII form; resolved to on-disk form at runtime) ------
BASE_DIRS = [
    # --- TiroBERTa / TiDG ---
    "experiments/TiroBERTa TiDG SAP-PEFT/news/roberta/progressive",
    "experiments/TiroBERTa TiDG Progressive Fixed/news/roberta/progressive",
    "experiments/TiroBERTa TiDG Progressive Linear/news/roberta/progressive",
    "experiments/TiroBERTa TiDG Progressive Adaptive/news/roberta/progressive",
    # --- TiroBERTa / Sentiment ---
    "experiments/TiroBERTa Sentiment SAP-PEFT/sentiment/roberta/progressive",
    "experiments/TiroBERTa Sentiment Progressive Fixed/sentiment/roberta/progressive",
    "experiments/TiroBERTa Sentiment Progressive Linear/sentiment/roberta/progressive",
    "experiments/TiroBERTa Sentiment Progressive Adaptive/sentiment/roberta/progressive",
    # --- Afro-XLM-R / TiDG ---
    "experiments/Afro-XLM-R SAP-PEFT/news/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Fixed/news/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Linear/news/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Adaptive/news/xlm-roberta/progressive",
    # --- Afro-XLM-R / Sentiment ---
    "experiments/Afro-XLM-R SAP-PEFT/sentiment/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Fixed/sentiment/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Linear/sentiment/xlm-roberta/progressive",
    "experiments/Afro-XLM-R Progressive Adaptive/sentiment/xlm-roberta/progressive",
    # --- GPT-2 / TiDG ---
    "experiments/GPT-2 SAP-PEFT/news/gpt2/progressive",
    "experiments/GPT-2 Progressive Fixed/news/gpt2/progressive",
    "experiments/GPT-2 Progressive Linear/news/gpt2/progressive",
    "experiments/GPT-2 Progressive Adaptive/news/gpt2/progressive",
    # --- GPT-2 / Sentiment ---
    "experiments/GPT-2 SAP-PEFT/sentiment/gpt2/progressive",
    "experiments/GPT-2 Progressive Fixed/sentiment/gpt2/progressive",
    "experiments/GPT-2 Progressive Linear/sentiment/gpt2/progressive",
    "experiments/GPT-2 Progressive Adaptive/sentiment/gpt2/progressive",
    # --- AdaLoRA / TiDG ---
    "experiments/TiroBERTa TiDG AdaLoRA Dynamic (Pilot_final)/news/roberta/adalora",
    "experiments/Afro-XLM-R TiDG AdaLoRA Dynamic (Pilot_Final)/news/xlm-roberta/adalora",
    "experiments/GPT-2 TiDG AdaLoRA Dynamic (Pilot-Final)/news/gpt2/adalora",
    # --- LoRA r=8 /TiDg ---
    "experiments/TiroBERTa TiDG LoRA/news/roberta/lora",
    "experiments/Afro-XLM-R LoRA/news/xlm-roberta/lora",
    "experiments/GPT-2 LoRA/news/gpt2/lora",
    # --- LoRA r=8 /sentiment ---
    "experiments/TiroBERTa Sentiment LoRA/sentiment/roberta/lora",
    "experiments/Afro-XLM-R LoRA/sentiment/xlm-roberta/lora",
    "experiments/GPT-2 LoRA/sentiment/gpt2/lora",
    # --- DoRA r=8 /TiDg ---
    "experiments/TiroBERTa TiDG DoRA/news/roberta/dora",
    "experiments/Afro-XLM-R DoRA/news/xlm-roberta/dora",
    "experiments/GPT-2 DoRA/news/gpt2/dora",
     # --- DoRA r=8 /sentiment ---
    "experiments/TiroBERTa Sentiment DoRA/sentiment/roberta/dora",
    "experiments/Afro-XLM-R DoRA/sentiment/xlm-roberta/dora",
    "experiments/GPT-2 DoRA/sentiment/gpt2/dora",

    # --- AdaLoRA fixed r=8/TiDg (NOT Dynamic) ---
    "experiments/TiroBERTa TiDG AdaLoRA/news/roberta/adalora",
    "experiments/Afro-XLM-R AdaLoRA/news/xlm-roberta/adalora",
    "experiments/GPT-2 AdaLoRA/news/gpt2/adalora",

    # --- AdaLoRA fixed r=8 /sentiment(NOT Dynamic) ---
    "experiments/TiroBERTa Sentiment AdaLoRA/sentiment/roberta/adalora",
    "experiments/Afro-XLM-R AdaLoRA/sentiment/xlm-roberta/adalora",
    "experiments/GPT-2 AdaLoRA/sentiment/gpt2/adalora",
]


def read_seed(json_path: Path):
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return (
            d.get("experiment_metadata", {}).get("seed")
            or d.get("experiment_metadata", {}).get("reproducibility_info", {}).get("seed")
            or "?"
        )
    except Exception:
        return "?"

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate a manifest of seed runs across all methods."
    )
    ap.add_argument("--output", type=Path, default=None,
                    help="Write manifest directly to this file (UTF-8). "
                         "If omitted, prints to stdout.")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="Project root (default: current directory)")
    args = ap.parse_args()

    root = args.root
    lines = [
        "# Auto-generated manifest - one line per seed run",
        "# Format: relative path from project root",
        "",
    ]
    total = 0
    missing_methods: list[str] = []
    resolved_with_canonicalization: list[str] = []

    for base in BASE_DIRS:
        base_path = resolve_under_root(base, root)
        if base_path is None:
            missing_methods.append(base)
            continue

        # Note if the disk name differs from the ASCII form in BASE_DIRS
        if base_path.as_posix() != base:
            resolved_with_canonicalization.append(
                f"{base}  →  {base_path.as_posix()}"
            )

        seed_dirs = []
        for ts_dir in sorted(base_path.iterdir()):
            if not ts_dir.is_dir():
                continue
            json_path = ts_dir / "experiment_results.json"
            if not json_path.exists():
                continue
            seed = read_seed(json_path)
            seed_dirs.append((seed, ts_dir))

        if not seed_dirs:
            missing_methods.append(f"{base}  (resolved, but no seeds found)")
            continue

        lines.append(f"\n# --- {base_path.as_posix()} ({len(seed_dirs)} seeds) ---")
        # Sort by timestamp folder name (YYYYMMDD_HHMMSS = ISO = lexicographic
        # == chronological). Older runs first.
        for seed, ts_path in sorted(seed_dirs, key=lambda x: x[1].name):
            lines.append(f"{ts_path.as_posix()}  # seed={seed}")
            total += 1

    manifest_text = "\n".join(lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Explicit UTF-8, no BOM - readable by every Python open()
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(manifest_text + "\n")
        print(f"Wrote: {args.output}", file=sys.stderr)
    else:
        print(manifest_text)


    # Manifest goes to stdout
     
    print(f"\n# Total runs: {total}", file=sys.stderr)

    if resolved_with_canonicalization:
        print(f"\n# Paths resolved via Unicode canonicalization "
              f"({len(resolved_with_canonicalization)}):", file=sys.stderr)
        for r in resolved_with_canonicalization:
            print(f"#   {r}", file=sys.stderr)

    if missing_methods:
        print(f"\n# Missing method paths ({len(missing_methods)}):", file=sys.stderr)
        for m in missing_methods:
            print(f"#   {m}", file=sys.stderr)

    

    return 0


if __name__ == "__main__":
    sys.exit(main())