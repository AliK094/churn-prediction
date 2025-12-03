churn-prediction/
├── README.md
├── pyproject.toml           # or requirements.txt + setup.cfg
├── .gitignore
├── .pre-commit-config.yaml  # for formatting/linting (phase 3)
├── Dockerfile               # optional, for deployment (phase 3)
├── .github/
│   └── workflows/
│       └── ci.yml           # unit tests, lint, basic checks (phase 3)
│
├── data/                    # NEVER commit real data
│   ├── raw/                 # original datasets (ignored in git)
│   ├── interim/             # after cleaning / preprocessing
│   └── processed/           # model-ready features
│
├── notebooks/               # exploration only
│   ├── 01_eda_churn.ipynb
│   └── 02_baseline_model.ipynb
│
├── src/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── training_config.yaml   # model, features, paths, etc.
│       ├── data/
│       │   ├── __init__.py
│       │   └── load_data.py           # raw → pandas DataFrame
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py      # preprocessing + feature pipeline
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py               # training pipeline
│       │   └── predict.py             # batch inference function
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # ROC AUC, PR AUC, F1, etc.
│       │   └── evaluate.py            # offline eval pipeline
│       ├── drift/                     # phase 2
│       │   ├── __init__.py
│       │   ├── data_drift.py          # feature dist, PSI, etc.
│       │   └── model_drift.py         # performance drift over time
│       └── pipelines/
│           ├── __init__.py
│           ├── train_pipeline.py      # orchestrate: load → features → train → eval
│           └── batch_inference.py     # simulate production batch scoring
│
├── scripts/                 # CLI entrypoints
│   ├── train_model.py       # `python -m scripts.train_model`
│   ├── run_inference.py
│   └── run_drift_checks.py  # phase 2
│
└── tests/                   # unit + integration tests (phase 3)
    ├── __init__.py
    ├── test_features.py
    ├── test_models.py
    └── test_drift.py        # phase 2
