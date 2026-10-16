#!/usr/bin/env python
"""
Run systematic baseline comparison - PRODUCTION VERSION
Usage: python run_baseline_comparison.py --config configs/baselines/scientific_baseline.yaml
"""

import argparse
from pathlib import Path
import sys
import logging
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # already set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we import from the correct location
from config.baseline_config import BaselineRunnerConfig, ScientificBaselineRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run systematic baseline comparison")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to baseline configuration YAML")
    parser.add_argument("--parallel", action="store_true",
                       help="Run experiments in parallel")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--seeds", type=str, default=None,
                       help="Comma-separated random seeds")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory")
    parser.add_argument("--sequential", action="store_true",
                       help="Force sequential execution (no parallel)")
    
    args = parser.parse_args()
    
    # Parse random seeds
    random_seeds=None
    if args.seeds is not None:
        random_seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info("🔬 Loading baseline configuration...")
    
    try:
        # Load baseline configuration using Pydantic
        baseline_config = BaselineRunnerConfig.from_yaml(str(config_path))
        logger.info(f"Loaded configuration with {len(baseline_config.baseline_methods)} methods")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Try alternative loading method
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            baseline_config = BaselineRunnerConfig(**config_data)
            logger.info("Loaded via direct parsing")
        except Exception as e2:
            logger.error(f" Failed to load configuration: {e2}")
            sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir:
        baseline_config.output_dir = args.output_dir
    
    # Create and run baseline comparison
    runner = ScientificBaselineRunner(
        baseline_config=baseline_config,
        random_seeds=random_seeds,
        output_dir=baseline_config.output_dir,
        enable_mlflow=True,
        enable_monitoring=True
    )
    
    # Run comparison
    logger.info("Starting baseline comparison...")
    
    # Determine parallel execution
    if args.sequential:
        parallel = False
    elif args.parallel:
        parallel = True
    else:
        parallel = False

    if parallel:
        logger.info(f" Running in parallel with {args.workers} workers")
    else:
        logger.info(" Running sequentially")
    
    try:
        results = runner.run_comparison(
            parallel=parallel,
            adaptive_scheduling=True
        )
    except Exception as e:
        logger.error(f" Baseline comparison failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Print summary
    summary = runner._generate_summary()
    
    print(f"\n{'='*80}")
    print(f"BASELINE COMPARISON COMPLETED!")
    print(f"{'='*80}")
    print(f" Success rate: {summary['success_rate']*100:.1f}%")
    print(f" Results saved to: {runner.output_dir}")
    print(f" Methods compared: {len(baseline_config.baseline_methods)}")
    print(f" Replications per method: {baseline_config.replication_count}")
    print(f"⏱ Average training time: {summary['average_training_time_hours']:.2f} hours")
    print(f"\n Reports generated:")
    print(f"  - {runner.output_dir}/baseline_report.md")
    print(f"  - {runner.output_dir}/baseline_summary.csv")
    print(f"  - {runner.output_dir}/statistical_analysis.json")
    print(f"  - {runner.output_dir}/baseline_comparison.png")
    
    if runner.mlflow_enabled:
        print(f" MLflow tracking available at: http://localhost:5000")
    
    return results

if __name__ == "__main__":
    main()