"""Optional FastAPI inference endpoint for the LPBF PINN surrogate.

Run with::

    uvicorn api:app --reload

FastAPI/uvicorn are optional dependencies.
"""


import yaml

from infer import predict


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="LPBF-Optimizer Inference API")

    class PredictRequest(BaseModel):
        config_path: str = 'data/params.yaml'
        model_path: str
        params: dict
        mc_dropout: bool = False
        num_samples: int = 100

    @app.post("/predict")
    def predict_endpoint(request: PredictRequest):
        config = _load_config(request.config_path)
        param_names = list(config['optimizer']['param_bounds'].keys())
        param_values = [float(request.params[p]) for p in param_names]
        return predict(
            config,
            request.model_path,
            param_values,
            mc_dropout=request.mc_dropout,
            num_samples=request.num_samples,
        )

except ImportError as e:  # pragma: no cover - FastAPI is optional
    app = None
    _import_error = e
