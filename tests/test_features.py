import pandas as pd

from churn_prediction.features.build_features import (
    FeatureBuilder,
    FeatureConfig,
)


def test_feature_builder_fit_transform_shapes():
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3"],
            "tenure": [1, 5, 10],
            "MonthlyCharges": [10.0, 20.0, 30.0],
            "TotalCharges": [10.0, 100.0, 300.0],
            "gender": ["Male", "Female", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer"],
        }
    )

    cfg = FeatureConfig(
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

    builder = FeatureBuilder(cfg)
    X = builder.fit_transform(df)

    assert X.shape[0] == len(df)
    assert X.shape[1] > 0


