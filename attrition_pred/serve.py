import argparse
from http import HTTPStatus
from typing import Dict

import uvicorn
from fastapi import FastAPI
from starlette.requests import Request

from attrition_pred import evaluate, predict
from attrition_pred.config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(title="Made With ML Tutorial", description="Employee Attrition Prediction", version="0.1")

model_deployment = None


class ModelDeployment:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.artifact_uri = predict.get_artifact_path(run_id)

    def get_run_id(self) -> Dict:
        return {"run_id": self.run_id}

    def evaluate_model(self, request: Request) -> Dict:
        data = request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    def predict_model(self, request: Request) -> Dict:
        data = request.json()
        data = dict(data)
        results = predict.predict(unit_data=data, run_id=self.run_id)
        return {"results": results}


@app.get("/")
def _index() -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/run_id/", response_model=dict)
def get_run_id():
    return model_deployment.get_run_id()


@app.post("/evaluate/", response_model=dict)
async def evaluate(request: Request):
    return await model_deployment.evaluate(request)


@app.post("/predict/", response_model=dict)
async def predict_online(request: Request):
    return await model_deployment.predict(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    args = parser.parse_args()

    model_deployment = ModelDeployment(run_id=args.run_id)

    # Run the FastAPI application with autoreload
    uvicorn.run(app, host="127.0.0.1", port=8000)


# import argparse
# import os
# from http import HTTPStatus
# from typing import Dict

# from fastapi import FastAPI
# from starlette.requests import Request

# from attrition_pred import evaluate, predict
# from attrition_pred.config import MLFLOW_TRACKING_URI, mlflow

# # Define application
# app = FastAPI(
#     title="Made With ML",
#     description="Classify machine learning projects.",
#     version="0.1",
# )

# class ModelDeployment:
#     def __init__(self, run_id: str):#, threshold: int = 0.9):
#         """Initialize the model."""
#         self.run_id = run_id
#         #self.threshold = threshold
#         self.artifact_uri = predict.get_artifact_path(run_id)
#         # model_weight_path = os.path.join(self.artifact_path, "model_weight", "random_forest_model.joblib")
#         # self.rf_model = CustomRandomForestClassifier()
#         # self.rf_model = self.rf_model.load_model(model_weight_path, self.rf_model.rf_params)


#     @app.get("/")
#     def _index() -> Dict:
#         """Health check."""
#         response = {
#             "message": HTTPStatus.OK.phrase,
#             "status-code": HTTPStatus.OK,
#             "data": {},
#         }
#         return response

#     @app.get("/run_id/")
#     def _run_id(self) -> Dict:
#         """Get the run ID."""
#         return {"run_id": self.run_id}

#     @app.post("/evaluate/")
#     async def _evaluate(self, request: Request) -> Dict:
#         data = await request.json()
#         results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
#         return {"results": results}

#     @app.post("/predict/")
#     async def _predict(self, request: Request):
#         data = await request.json()
#         data = dict(data)
#         #sample_ds = [{"title": data.get("title", ""), "description": data.get("description", ""), "tag": ""}]
#         results = predict.predict(unit_data=data, run_id=self.run_id)

#         # # Apply custom logic
#         # for i, result in enumerate(results):
#         #     pred = result["prediction"]
#         #     prob = result["probabilities"]
#         #     if prob[pred] < self.threshold:
#         #         results[i]["prediction"] = "other"

#         return {"results": results}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run_id", help="run ID to use for serving.")
#     #parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
#     args = parser.parse_args()

#     # Initialize the model deployment outside of Ray Serve
#     model_deployment = ModelDeployment(run_id=args.run_id)#, threshold=args.threshold)

#     # Run the FastAPI application
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
