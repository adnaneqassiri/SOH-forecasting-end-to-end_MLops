# SOH inference API and dashboard

The dashboard reproduces the original retrospective evaluation for the
held-out Data 4 vehicles: each vehicle has a fixed test snapshot, the model
receives the 100 preceding charging events, and it forecasts the following 10
real charge segments. The API loads the fine-tuned checkpoint and scaler once
at startup. Nominal capacity is not a user input: it is estimated per vehicle
as the 0.99 capacity quantile, matching the Data 4 processing pipeline.

Start the API from the repository root:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Start the Streamlit interface in another terminal:

```bash
API_URL=http://localhost:8000 streamlit run src/frontend/app.py
```

Then open <http://localhost:8501>. The first screen is a selection dialog for
one, several, or all configured test vehicles.

Useful API endpoints:

```text
GET  /health
GET  /vehicles
GET  /vehicles/{vehicle_id}/forecast
POST /predict
```

Multi-vehicle request examples:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"vehicle_ids":[16,67]}'

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"all_vehicles":true}'
```

The raw-CSV inference CLI remains the real-life/latest-window path for a single
vehicle. It expects at least 100 charge events and estimates nominal capacity
automatically:

```bash
python -m src.inference.predict vehicle.csv --output predictions.csv
```

---

sudo apt update
sudo apt install -y python3-pip python3-venv

mkdir -p ~/mlflow-server
cd ~/mlflow-server

python3 -m venv .venv
source .venv/bin/activate


pip install --upgrade pip
pip install mlflow boto3
mlflow --version

sudo apt install -y awscli
aws sts get-caller-identity
aws s3 ls s3://soh-project

cd ~/mlflow-server
source .venv/bin/activate


mlflow server \
    --backend-store-uri sqlite:////home/ubuntu/mlflow-server/mlflow.db \
    --artifacts-destination s3://soh-project/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"
