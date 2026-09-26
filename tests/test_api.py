from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakePredictor:
    device = SimpleNamespace(type="cpu")
    test_vehicle_ids = (16, 67)

    def list_vehicles(self):
        return [
            {
                "vehicle_id": vehicle_id,
                "display_name": f"Véhicule {vehicle_id}",
                "available_cycles": 120,
            }
            for vehicle_id in self.test_vehicle_ids
        ]

    def forecast(self, vehicle_id):
        if vehicle_id not in self.test_vehicle_ids:
            raise KeyError(vehicle_id)
        return {"vehicle_id": vehicle_id, "forecast": [0.9] * 10}

    def predict_many(self, vehicle_ids=None):
        selected = self.test_vehicle_ids if vehicle_ids is None else vehicle_ids
        unknown = set(selected) - set(self.test_vehicle_ids)
        if unknown:
            raise KeyError(tuple(sorted(unknown)))
        return {
            "selected_vehicle_ids": list(selected),
            "total_vehicles": len(selected),
            "vehicles": [self.forecast(vehicle_id) for vehicle_id in selected],
        }


def client():
    return TestClient(create_app(predictor_factory=FakePredictor))


def test_api_lists_only_test_vehicles():
    with client() as test_client:
        response = test_client.get("/vehicles")

    assert response.status_code == 200
    assert [item["vehicle_id"] for item in response.json()] == [16, 67]


def test_api_predicts_multiple_selected_vehicles():
    with client() as test_client:
        response = test_client.post(
            "/predict", json={"vehicle_ids": [67, 16]}
        )

    assert response.status_code == 200
    assert response.json()["selected_vehicle_ids"] == [67, 16]


def test_api_select_all_uses_complete_test_split():
    with client() as test_client:
        response = test_client.post("/predict", json={"all_vehicles": True})

    assert response.status_code == 200
    assert response.json()["selected_vehicle_ids"] == [16, 67]


def test_api_rejects_an_empty_selection():
    with client() as test_client:
        response = test_client.post("/predict", json={})

    assert response.status_code == 422


def test_api_rejects_a_vehicle_outside_the_test_split():
    with client() as test_client:
        response = test_client.post(
            "/predict", json={"vehicle_ids": [999]}
        )

    assert response.status_code == 404
