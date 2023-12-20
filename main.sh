#!/bin/bash

echo "Choose a command to run:"
echo "1. Setting Up the ENV"
echo "2. Spinning Up MLFLOW"
echo "3. Running Prediction On holdout set"
echo "4. Serving FastAPI"
echo "5. Testing Code"
echo "6. Testing Data"
echo "7. Coverage"
echo "8. Exit"

read -p "Enter the number corresponding to the command: " choice

case $choice in
    1)
        # Setting Up the ENV
        export PYTHONPATH=$PYTHONPATH:$PWD
        python3 -m venv venv
        source venv/bin/activate
        python3 -m pip install --upgrade pip setuptools wheel
        python3 -m pip install -r requirements.txt
        pre-commit install
        pre-commit autoupdate
        ;;
    2)
        # Spinning Up MLFLOW
        export MODEL_REGISTRY=$(python -c "from attrition_pred import config; print(config.MODEL_REGISTRY)")
        mlflow ui --backend-store-uri $MODEL_REGISTRY
        # Alternatively:
        # mlflow ui --backend-store-uri efs/mlflow
        ;;
    3)
        # Running Prediction On holdout set
        export EXPERIMENT_NAME="attrition_pred"
        RUN_ID=$(python attrition_pred/predict.py get-best-run-id $EXPERIMENT_NAME f1_score DESC | tail -n 1)
        export HOLDOUT_LOC="data/processed/holdout_dataset.csv"

        # Evaluate
        python attrition_pred/evaluate.py \
            --run-id $RUN_ID \
            --dataset-loc $HOLDOUT_LOC \
            --results-fp results/evaluation_results.json
        ;;
    4)
        # Serving FastAPI
        python attrition_pred/serve.py --run_id $RUN_ID
        ;;
    5)
        # Testing Code
        python3 -m pytest tests/code --verbose --disable-warnings
        ;;
    6)
        # Testing Data
        export DATASET_LOC="data/processed/HR_Employee_data.csv"
        pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
        ;;
    7)
        # Coverage
        python3 -m pytest tests/code --cov attrition_pred --cov-report html --disable-warnings
        python3 -m pytest tests/code --cov attrition_pred --cov-report term --disable-warnings
        ;;
    8)
        # Exit
        echo "Exiting the script."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
