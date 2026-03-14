import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and encoder
model = joblib.load("xgbmodel.pkl")
ohe = joblib.load("ohe.pkl")

app = FastAPI(title="API Bot Detection Model")

# Input schema
class RequestData(BaseModel):
    http_method: str
    endpoint: str
    status_code: int
    response_size: int
    request_rate: float
    session_duration: int
    requests_per_session: int
    time_between_requests: float
    failed_requests: int
    url_length: int
    query_param_count: int
    payload_size: int
    is_proxy: int
    distinct_endpoints_accessed: int
    login_attempts: int
    request_pattern_entropy: float

@app.get("/")
def home():
    return {"message": "API Bot Detection Model Running"}

@app.post("/predict")
def predict(data: RequestData):

    # convert input to dataframe
    row = pd.DataFrame([data.dict()])

    # categorical encoding
    cat_cols = ["http_method", "endpoint"]

    encoded = ohe.transform(row[cat_cols])

    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

    row = row.drop(cat_cols, axis=1)
    row_final = pd.concat([row.reset_index(drop=True), encoded_df], axis=1)

    # prediction
    pred = model.predict(row_final)[0]
    prob = model.predict_proba(row_final)[0]

    label = "BOT" if pred == 1 else "HUMAN"

    return {
        "prediction": label,
        "confidence": float(prob[pred])
    }
