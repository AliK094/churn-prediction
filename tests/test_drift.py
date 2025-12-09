import numpy as np
import pandas as pd

from churn_prediction.drift.data_drift import (
    DataDriftConfig,
    DataDriftDetector,
)


def test_data_drift_detector_flags_shifted_numeric():
    rng = np.random.default_rng(0)
    n = 1000

    df_ref = pd.DataFrame(
        {
            "tenure": rng.normal(10, 2, size=n),
            "MonthlyCharges": rng.normal(50, 5, size=n),
            "TotalCharges": rng.normal(500, 50, size=n),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
        }
    )

    df_cur = df_ref.copy()
    df_cur["tenure"] = rng.normal(30, 2, size=n)

    cfg = DataDriftConfig(
        numeric_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=["Contract"],
        psi_threshold=0.1,
        cat_l1_threshold=0.3,
    )

    detector = DataDriftDetector(cfg)
    results = detector.detect_drift(df_ref, df_cur)

    assert "tenure" in results
    assert results["tenure"].drifted

