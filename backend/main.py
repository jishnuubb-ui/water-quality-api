"""
main.py - the FastAPI web service.

Run locally (from the project root):
    python -m uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs  (automatic, interactive documentation)

Endpoints
    GET  /            what this service is
    GET  /health      is it alive? (use this to wake the service on Render)
    GET  /model-info  which model, threshold, CV metrics, known limitations
    GET  /limits      the EPA/WHO limits used to define a confirmed violation
    GET  /example     ready-made JSON bodies you can paste into Postman
    POST /predict     classifier only: safe / unsafe + probability
    POST /screen      full screening: classifier + anomaly + violations + category
    POST /screen/batch  the same for up to 200 samples at once
"""
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend import service
from src.screening import Screener

EXAMPLE_SAFE = {"aluminium": 1.65, "ammonia": 9.08, "arsenic": 0.04, "barium": 2.85, "cadmium": 0.007,
                "chloramine": 0.35, "chromium": 0.83, "copper": 0.17, "fluoride": 0.05, "bacteria": 0.2,
                "viruses": 0.0, "lead": 0.054, "nitrates": 16.08, "nitrites": 1.13, "mercury": 0.007,
                "perchlorate": 37.75, "radium": 6.78, "selenium": 0.08, "silver": 0.34, "uranium": 0.02}
EXAMPLE_UNSAFE = {"aluminium": 1.01, "ammonia": 14.02, "arsenic": 0.04, "barium": 0.58, "cadmium": 0.008,
                  "chloramine": 4.24, "chromium": 0.53, "copper": 0.02, "fluoride": 0.99, "bacteria": 0.05,
                  "viruses": 0.003, "lead": 0.078, "nitrates": 14.16, "nitrites": 1.11, "mercury": 0.006,
                  "perchlorate": 50.28, "radium": 7.07, "selenium": 0.07, "silver": 0.44, "uranium": 0.01}

state = {}   # holds the loaded models (loaded ONCE at startup, not on every request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["screener"] = Screener()
    yield


app = FastAPI(
    title="Water Quality Screening API",
    description="Classifies water samples as safe/unsafe, flags statistical anomalies, and checks "
                "EPA/WHO limit violations. Trained on a (probably synthetic) 20-chemical dataset.",
    version="1.0",
    lifespan=lifespan,
)

# Allow a browser front-end (built next week) to call this API from another address.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Sample(BaseModel):
    """One water sample: 20 chemical measurements. All must be numbers >= 0."""
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_SAFE})

    aluminium: float = Field(ge=0)
    ammonia: float = Field(ge=0)
    arsenic: float = Field(ge=0)
    barium: float = Field(ge=0)
    cadmium: float = Field(ge=0)
    chloramine: float = Field(ge=0)
    chromium: float = Field(ge=0)
    copper: float = Field(ge=0)
    fluoride: float = Field(ge=0)
    bacteria: float = Field(ge=0)
    viruses: float = Field(ge=0)
    lead: float = Field(ge=0)
    nitrates: float = Field(ge=0)
    nitrites: float = Field(ge=0)
    mercury: float = Field(ge=0)
    perchlorate: float = Field(ge=0)
    radium: float = Field(ge=0)
    selenium: float = Field(ge=0)
    silver: float = Field(ge=0)
    uranium: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def accept_original_typo(cls, data):
        # The original CSV header is misspelled "flouride"; accept it too.
        if isinstance(data, dict) and "flouride" in data and "fluoride" not in data:
            data = {**data, "fluoride": data["flouride"]}
        return data


class Batch(BaseModel):
    samples: List[Sample] = Field(min_length=1, max_length=200)


def screener() -> Screener:
    if "screener" not in state:
        raise HTTPException(status_code=503, detail="Models are not loaded yet; try again in a moment.")
    return state["screener"]


@app.get("/")
def root():
    return {"service": "Water Quality Screening API", "status": "running", "docs": "/docs",
            "endpoints": ["/health", "/model-info", "/limits", "/example",
                          "POST /predict", "POST /screen", "POST /screen/batch"]}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "screener" in state}


@app.get("/model-info")
def model_info():
    m = dict(screener().meta)
    m.pop("feature_stats", None)
    m["known_limitations"] = [
        "The dataset looks synthetic; chemical relationships (e.g. aluminium vs safety) do not follow real chemistry.",
        "Almost every sample exceeds at least one EPA/WHO limit, so limit violations do not separate safe from unsafe samples here.",
        "Anomaly flags mean 'statistically unusual', not 'unsafe'.",
        "Performance figures are cross-validated estimates on this dataset only.",
    ]
    return m


@app.get("/limits")
def limits():
    return screener().limits_cfg


@app.get("/example")
def example():
    return {"safe_like_sample": EXAMPLE_SAFE, "unsafe_like_sample": EXAMPLE_UNSAFE}


@app.post("/predict")
def predict(sample: Sample):
    s = screener()
    res = service.screen_many(s, [sample.model_dump()])[0]
    return res["classification"]


@app.post("/screen")
def screen(sample: Sample):
    return service.screen_many(screener(), [sample.model_dump()])[0]


@app.post("/screen/batch")
def screen_batch(batch: Batch):
    results = service.screen_many(screener(), [s.model_dump() for s in batch.samples])
    return {"count": len(results), "results": results}
