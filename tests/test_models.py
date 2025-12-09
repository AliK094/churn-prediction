import numpy as np
import pandas as pd

from churn_prediction.features.build_features import FeatureBuilder, FeatureConfig
from churn_prediction.models.train import ModelConfig, train_xgb, ChurnModelTrainer
from churn_prediction.models.predict import PredictionConfig, ChurnPredictor


def _make_synthetic_telco_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)

    df = pd.DataFrame(
        {
            "customerID": [f"{i}" for i in range(n)],
            "tenure": rng.integers(0, 72, size=n),
            "MonthlyCharges": rng.uniform(20, 100, size=n),
            "TotalCharges": rng.uniform(0, 5000, size=n),
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No"], size=n),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"], size=n
            ),
            "OnlineSecurity": rng.choice(["Yes", "No"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No"], size=n),
            "TechSupport": rng.choice(["Yes", "No"], size=n),
            "StreamingTV": rng.choice(["Yes", "No"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No"], size=n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card",
                ],
                size=n,
            ),
        }
    )

    # Simple synthetic target with some signal
    df["Churn"] = (
        (df["Contract"] == "Month-to-month").astype(int)
        + (df["PaperlessBilling"] == "Yes").astype(int)
        + (df["SeniorCitizen"] == 1).astype(int)
    ) > 1
    df["Churn"] = df["Churn"].astype(int)
    return df


def test_trainer_and_predictor_end_to_end(tmp_path):
    df = _make_synthetic_telco_df(100)

    feature_cfg = FeatureConfig(
        id_cols=["customerID"],
        numeric_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=[
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
    )

    model_cfg = ModelConfig(
        learning_rate=0.1,
        n_estimators=10,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
    )

    trainer = ChurnModelTrainer(feature_config=feature_cfg, model_config=model_cfg)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    trainer.fit(X, y)

    model_path = tmp_path / "xgb_model.joblib"
    pipeline_path = tmp_path / "feature_pipeline.joblib"
    trainer.save(model_path, pipeline_path)

    pred_cfg = PredictionConfig(
        model_path=model_path,
        feature_pipeline_path=pipeline_path,
        threshold=0.5,
    )

    predictor = ChurnPredictor.from_config(pred_cfg)
    preds_df = predictor.predict_with_proba(X)

    assert len(preds_df) == len(X)
    assert "churn_score" in preds_df.columns
    assert "churn_pred" in preds_df.columns

