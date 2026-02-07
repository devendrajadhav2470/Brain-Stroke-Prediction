"""CLI entry point for running the full training pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import stroke_risk
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from stroke_risk.models.train import run_training_pipeline  # noqa: E402


def main():
    """Run the stroke risk training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train stroke risk prediction models with Optuna + MLflow."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to the configuration directory (default: configs)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("Stroke Risk ML -- Training Pipeline")
    logger.info("=" * 70)

    results = run_training_pipeline(config_dir=args.config_dir)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("Best model: %s", results["best_model_name"])
    logger.info("Best threshold: %.4f", results["best_threshold"])
    logger.info("=" * 70)
    logger.info("\nModel Comparison:")
    logger.info("\n%s", results["comparison_table"].to_string())


if __name__ == "__main__":
    main()

