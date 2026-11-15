"""FastAPI service.  Run:  uvicorn app.api:app --reload"""
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import Forecaster

app = FastAPI(title="Cinema Ticket Demand Forecasting", version="1.0")


@lru_cache(maxsize=1)
def get_forecaster():
    return Forecaster()


class ShowRequest(BaseModel):
    movie_id: str = Field(examples=["M110"])
    genre: str
    language: str
    runtime_min: int = Field(ge=30, le=300)
    is_franchise: int = Field(ge=0, le=1)
    pre_release_buzz: float = Field(ge=0, le=100)
    release_date: str = Field(examples=["2024-06-07"])
    screen_capacity: int = Field(gt=0, le=1000)
    show_datetime: str = Field(examples=["2024-06-29T19:00:00"])
    ticket_price: float = Field(gt=0)


@app.get("/health")
def health():
    f = get_forecaster()
    return {"status": "ok", "model": f.art["best_name"], "history_ends": str(f.last_history)}


@app.post("/predict")
def predict(req: ShowRequest):
    try:
        return get_forecaster().predict(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
