# MLOps-2-Packaging-Loan-Approval-Model
This repository is presenting a strategy to package the model, presented in [Loan-Approval-Prediction-Model](https://github.com/asadhanif3188/MLOps-1-Loan-Approval-Prediction-Model).

## Directory structure

```
packaging-loan-model

├── MANIFEST.in
├── README.md
├── requirements.txt
├── setup.py
├── prediction_model
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── training_pipeline.py
│   ├── VERSION
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   └── __init__.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   └── trained_models
│       └── __init__.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


### Crate venv in Windows 

```
pip install virtualenv 
python -m venv mlpipeline
mlpipeline\Scripts\activate
deactivate
```