"""FastAPI service for retrospective test-fleet SOH predictions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.inference.service import TestFleetPredictor


class PredictionRequest(BaseModel):
    """Selection sent by the Streamlit vehicle dialog."""

    vehicle_ids: list[int] | None = Field(
        default=None,
        description="One or more IDs from the configured Data 4 test split.",
    )
    all_vehicles: bool = Field(
        default=False,
        description="Forecast every vehicle in the configured test split.",
    )


def create_app(
    predictor_factory: Callable[[], Any] = TestFleetPredictor,
) -> FastAPI:
    """Create the API, allowing a lightweight predictor in endpoint tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor_factory()
        yield

    application = FastAPI(
        title="Battery SOH Inference API",
        description=(
            "Fixed-snapshot forecasts for held-out Data 4 test vehicles. "
            "The model uses the 100 charging events preceding each configured "
            "snapshot and predicts the next 10 events."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def service(request: Request):
        return request.app.state.predictor

    @application.get("/health")
    def health(request: Request):
        predictor = service(request)
        return {
            "status": "ok",
            "model_loaded": True,
            "device": predictor.device.type,
            "test_vehicles": len(predictor.test_vehicle_ids),
        }

    @application.get("/vehicles")
    def vehicles(request: Request):
        return service(request).list_vehicles()

    @application.get("/vehicles/{vehicle_id}/forecast")
    def vehicle_forecast(vehicle_id: int, request: Request):
        try:
            return service(request).forecast(vehicle_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle {vehicle_id} is not in the test split",
            ) from None

    @application.post("/predict")
    def predict(selection: PredictionRequest, request: Request):
        if selection.all_vehicles and selection.vehicle_ids:
            raise HTTPException(
                status_code=422,
                detail="Use all_vehicles or vehicle_ids, not both",
            )
        if not selection.all_vehicles and not selection.vehicle_ids:
            raise HTTPException(
                status_code=422,
                detail="Select at least one vehicle or set all_vehicles=true",
            )
        vehicle_ids = None if selection.all_vehicles else selection.vehicle_ids
        try:
            return service(request).predict_many(vehicle_ids)
        except KeyError as error:
            unknown = error.args[0]
            raise HTTPException(
                status_code=404,
                detail=f"Vehicles outside the test split: {unknown}",
            ) from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    return application


app = create_app()


__all__ = ["PredictionRequest", "app", "create_app"]
