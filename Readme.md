## Goal
Learn ML learning sytem development on Industry following the tutorial from [madewithml](https://madewithml.com) and [Designing ML System by Chip](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969).
Both the book and tutorial proved to be invaluable resources for bridging academic knowledge with practical applications in the industry.
<br>
The project is centered around classical machine learning and aligns closely with the objectives of my current work. As a result, it selectively excludes certain aspects from the original tutorial to better tailor it to the specific requirements of my ongoing project.

### Virtual environment

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
python3 -m venv venv  # recommend using Python 3.10
source venv/bin/activate  # on Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
pre-commit install
pre-commit autoupdate
```

  > Highly recommend using Python `3.10` and using [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows).


## Scripts


```bash
madewithml
├── config.py
├── data.py
├── evaluate.py
├── models.py
├── predict.py
├── serve.py
├── train.py
├── tune.py
└── utils.py
```

### Training
```bash
```
### Tuning
```bash
```

### Experiment tracking

We'll use [MLflow](https://mlflow.org/) to track our experiments and store our models and the [MLflow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments.
```bash
export MODEL_REGISTRY=$(python -c "from attrition_pred import config; print(config.MODEL_REGISTRY)")
mlflow ui --backend-store-uri $MODEL_REGISTRY
mlflow ui --backend-store-uri efs/mlflow
```

If you're running this notebook on your local laptop then head on over to <a href="http://localhost:8080/" target="_blank">http://localhost:8080/</a> to view your MLflow dashboard.


### Evaluation
```bash
export EXPERIMENT_NAME="attrition_pred"
python attrition_pred/predict.py get-best-run-id $EXPERIMENT_NAME f1_score DESC
# Copy the RunID (did it this way case it also prints path for some reason)
export RUN_ID="5d2575f94915464db85f82f33eb06efd"
export HOLDOUT_LOC="data/processed/holdout_dataset.csv"

python attrition_pred/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --results-fp results/evaluation_results.json
```
```json
{
  "timestamp": "December 19, 2023 08:26:33 PM",
  "run_id": "5d2575f94915464db85f82f33eb06efd",
  "overall": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "num_samples": 3.0
  },
  "per_class": {
    "Yes": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "num_samples": 3.0
    }
  },
  "slices": {}
}
...
```

### Inference
```bash
python attrition_pred/predict.py \
    predict \
    data/processed/unit_sample.json \
    --run-id 5d2575f94915464db85f82f33eb06efd
```
```json
[
  {
    "0": 0.24176165854349077,
    "1": 0.75823834145651
  }
]
```

### Serving

  ```bash
    python attrition_pred/serve.py --run_id $RUN_ID
  ```

  Once the application is running, we can use it via cURL, Python, etc.:

  ```python
  # via Python
  import json
  import requests
  title = "Transfer learning with transformers"
  description = "Using transformers for transfer learning on text classification tasks."
  json_data = json.dumps({"title": title, "description": description})
  requests.post("http://127.0.0.1:8000/predict", data=json_data).json()
  ```

### Testing
```bash
# Code
python3 -m pytest tests/code --verbose --disable-warnings

# Data
export DATASET_LOC="data/processed/HR_Employee_data.csv"
pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings

# Model


# Coverage
python3 -m pytest tests/code --cov attrition_pred --cov-report html --disable-warnings  # html report
python3 -m pytest tests/code --cov attrition_pred --cov-report term --disable-warnings  # terminal report
```

Source :
@article{madewithml,
    author       = {Goku Mohandas},
    title        = { Home - Made With ML },
    howpublished = {\url{https://madewithml.com/}},
    year         = {2023}
}
