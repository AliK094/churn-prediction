#!/usr/bin/env python
"""
CLI entrypoint for batch churn inference.

Example
-------
python -m scripts.run_inference \
    --input-csv data/raw/new_customers.csv \
    --model-dir artifacts/churn_model_v1 \
    --output-csv data/processed/new_customers_scored.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from churn_mlops.data.load_data import ChurnDataLoader
from churn_mlops.models.predict import ChurnPredictor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch churn prediction on a CSV file."
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV with customer data.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model artifacts "
             "(e.g., model.joblib, preprocessor.joblib).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save the scored CSV with churn probabilities.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Optional ID column name to carry through to the output.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probabilities to labels.",
    )

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def main():
    setup_logging()
    args = parse_args()

    input_path = Path(args.input_csv)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_csv)

    logger.info("Loading input data from %s", input_path)
    loader = ChurnDataLoader()
    # You can expose a more specific method like `load_inference_data`
    # if you want different behavior vs training/validation.
    df_raw: pd.DataFrame = loader.load_from_csv(input_path)

    logger.info("Loading model from %s", model_dir)
    predictor = ChurnPredictor(model_dir=model_dir, threshold=args.threshold)

    logger.info("Running batch inference on %d rows", len(df_raw))
    predictions_df = predictor.predict_dataframe(df_raw, id_col=args.id_col)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving scored data to %s", output_path)
    predictions_df.to_csv(output_path, index=False)

    logger.info("Inference complete. Sample of scored data:")
    logger.info("\n%s", predictions_df.head())


if __name__ == "__main__":
    main()
