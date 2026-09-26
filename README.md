# Battery State-of-Health Forecasting — End to End

An end-to-end MLOps project for forecasting electric-vehicle battery state of
health (SOH). It covers raw-data processing, two-stage model training,
evaluation, MLflow experiment tracking, real-life inference, a FastAPI service,
a Streamlit dashboard, Docker packaging, and automatic deployment to AWS EC2.

Live dashboard: <http://ec2-13-62-230-243.eu-north-1.compute.amazonaws.com:8501/>

## What the project does

The model consumes 100 charging events and predicts SOH for the next 10 charging
events. Training is performed in two stages:

1. Pretraining on laboratory battery-cell data.
2. Fine-tuning on real EV fleet data (Data 4).

The architecture combines a numerical BiLSTM branch with an image-based
encoder-decoder. Historical SOH is encoded as an image, fused with engineered
charging features, and reconstructed into a trajectory containing the forecast.

The project exposes two complementary inference paths:

- **Test-fleet dashboard:** retrospective evaluation at fixed snapshots for the
  held-out vehicles `16`, `67`, `69`, `152`, `175`, and `176`. This reproduces
  the original evaluation interval and plots the observed history, filtered
  reference SOH, and the following 10 real charging segments.
- **Raw-CSV inference:** real-life inference for one vehicle using only its most
  recent 100 observed charging events. It does not calculate `SOH_ref`, because
  that offline EMD target uses future information.

Nominal capacity is never requested from the user. It is estimated for each
vehicle as the 0.99 quantile of its observed capacity, matching preprocessing.

## Architecture

```text
Raw lab MAT files ──> lab_process.py ──> lab.csv ──> pretraining
                                                        │
Raw Data 4 archive ─> data4_process.py ─> real_ev.csv ─> fine-tuning
                                                        │
                              MLflow <── metrics + model artifacts
                                                        │
                              S3 <────── deployment data/artifact storage
                                                        │
                         Docker image: soh-app
                              │                  │
                       FastAPI :8000      Streamlit :8501
```

Only one Docker image is built. EC2 starts two containers from that image: one
for the API and one for the dashboard.

## Repository layout

```text
.
├── config.yaml                         # Data, model, split, and training config
├── Dockerfile                          # Shared API/frontend image
├── pyproject.toml                      # Python package and dependencies
├── src/
│   ├── api/main.py                     # FastAPI application
│   ├── data/
│   │   ├── lab_process.py              # Laboratory MAT preprocessing
│   │   ├── data4_process.py            # Real-EV archive preprocessing
│   │   └── datasets.py                 # Windows, scaling, SOH image codec
│   ├── frontend/app.py                 # Streamlit dashboard
│   ├── inference/
│   │   ├── predict.py                  # Latest-window raw-CSV inference
│   │   ├── service.py                  # Retrospective test-fleet service
│   │   └── download_artifacts.py       # MLflow deployment artifacts
│   ├── model/model.py                  # Multimodal neural network
│   └── training/
│       ├── train.py                    # Two-stage training
│       └── evaluate.py                 # Test-set evaluation
├── tests/                              # API, inference, codec, artifact tests
└── .github/workflows/ci-cd.yml         # Test, build, push, deploy
```

## Local installation

Python 3.10 is used by CI and the Docker image.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
```

For a CUDA environment, install the appropriate PyTorch build before installing
the project.

## Configuration

The main settings are in `config.yaml`:

- raw and processed data locations;
- 100-event history and 10-event forecast lengths;
- laboratory and Data 4 feature lists;
- model size and dropout;
- training hyperparameters;
- vehicle-level train/validation/test splits;
- fixed dashboard snapshots;
- artifact directories and MLflow experiment name.

Relative paths are resolved from the repository root.

## Data preparation

### Laboratory data

Place the raw MATLAB files under `data/raw/data 1`, then run:

```bash
python -m src.data.lab_process
```

Output:

```text
data/processed/lab/lab.csv
```

### Real EV data (Data 4)

Place `battery_dataset1.tar.gz` under `data/raw/data 4`. The processor streams
the archive, ranks eligible vehicles, extracts charging-event features,
estimates nominal capacity, and calculates the offline EMD reference SOH.

```bash
python -m src.data.data4_process --overwrite
```

Useful development options:

```bash
python -m src.data.data4_process \
  --top-n 30 \
  --history-length 100 \
  --prediction-length 10 \
  --max-files-per-archive 1000 \
  --overwrite
```

Outputs:

```text
data/processed/data4/real_ev.csv
data/processed/data4/vehicle_selection.csv
```

## Training and evaluation

Create a local `.env` when MLflow tracking is enabled:

```dotenv
MLFLOW_URI=http://localhost:5000
```

Run both stages in order:

```bash
python -m src.training.train --stage all
```

Or run them separately. Data 4 fine-tuning requires the laboratory checkpoint:

```bash
python -m src.training.train --stage lab
python -m src.training.train --stage data4
```

Evaluate the saved checkpoints:

```bash
python -m src.training.evaluate --stage all
python -m src.training.evaluate --stage data4 --device cpu
```

Local artifacts are written to:

```text
data/artifacts/lab_pretraining/
data/artifacts/data4_finetuning/
```

Each training run logs parameters, metrics, configuration, checkpoints,
scalers, histories, and horizon metrics to the `soh-training` MLflow
experiment. Runs are tagged with `stage=lab` or `stage=data4`.

### Download deployment artifacts from MLflow

To install the latest finished Data 4 checkpoint and scaler into the configured
local artifact directory:

```bash
MLFLOW_URI=http://localhost:5000 \
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
python -m src.inference.download_artifacts
```

Pin a specific run when reproducibility is required:

```bash
MLFLOW_URI=http://localhost:5000 \
MLFLOW_RUN_ID=266d92b1908b4182bdc492b2f3a9eacb \
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
python -m src.inference.download_artifacts
```

## Real-life inference from one raw CSV

The CSV must describe exactly one vehicle. Every row is one telemetry sample,
and samples from the same charging event share `charge_segment` (or the accepted
alias `charge_block_id`). At least 100 charging events are required.

Required columns:

```text
charge_segment, capacity, volt, current, soc, max_single_volt,
min_single_volt, max_temp, min_temp, timestamp
```

`car`/`vehicle_id` and `mileage` are optional. Timestamps may be numeric seconds
or datetime values understood by pandas.

```bash
python -m src.inference.predict vehicle.csv \
  --output predictions.csv
```

Optional overrides:

```bash
python -m src.inference.predict vehicle.csv \
  --checkpoint data/artifacts/data4_finetuning/best_model.pth \
  --scaler data/artifacts/data4_finetuning/scaler.joblib \
  --device cpu \
  --output predictions.csv
```

The output contains the vehicle identifier, last observed charging segment,
forecast horizon, predicted SOH, and predicted capacity.

## API and Streamlit dashboard

The API loads the fine-tuned checkpoint and scaler once at startup. It serves
only the configured held-out Data 4 test vehicles.

Start FastAPI from the repository root:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Start Streamlit in another terminal:

```bash
API_URL=http://localhost:8000 streamlit run src/frontend/app.py
```

Open <http://localhost:8501>. A selection dialog supports one, several, or all
test vehicles. The interface has separate **Fleet Dashboard** and **Vehicle
Analysis** pages and follows the active Streamlit light/dark theme.

Interactive API documentation is available at <http://localhost:8000/docs>.

### API endpoints

```text
GET  /health
GET  /vehicles
GET  /vehicles/{vehicle_id}/forecast
POST /predict
```

Examples:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/vehicles
curl http://localhost:8000/vehicles/16/forecast

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"vehicle_ids":[16,67]}'

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"all_vehicles":true}'
```

## Docker

Build the shared image:

```bash
docker build -t soh-app:local .
```

With the processed CSV and artifacts already present under `data/`, start both
containers:

```bash
docker network inspect soh-network >/dev/null 2>&1 \
  || docker network create soh-network

docker run -d \
  --name soh-api \
  --network soh-network \
  -v "$(pwd)/data:/app/data:ro" \
  -p 8000:8000 \
  soh-app:local

docker run -d \
  --name soh-frontend \
  --network soh-network \
  -e API_URL=http://soh-api:8000 \
  -p 8501:8501 \
  soh-app:local \
  streamlit run src/frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true
```

Health checks and cleanup:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8501/_stcore/health
docker logs soh-api
docker logs soh-frontend
docker rm -f soh-api soh-frontend
```

## MLflow server on EC2

The deployment expects an MLflow tracking server reachable from the application
container. The following commands reproduce the current lightweight EC2 setup.
Keep port `5000` private to the instance/VPC rather than exposing it publicly.

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv awscli

mkdir -p ~/mlflow-server
cd ~/mlflow-server

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlflow boto3
mlflow --version

aws sts get-caller-identity
aws s3 ls s3://soh-project
```

Start the tracking server:

```bash
cd ~/mlflow-server
source .venv/bin/activate
export AWS_DEFAULT_REGION=eu-north-1

mlflow server \
  --backend-store-uri sqlite:////home/ubuntu/mlflow-server/mlflow.db \
  --artifacts-destination s3://soh-project/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

Verify it from EC2:

```bash
curl --fail http://localhost:5000/health
```

For production operation, run this command under `systemd` or another process
manager so MLflow survives SSH disconnections and instance restarts.

## CI/CD deployment

The workflow at `.github/workflows/ci-cd.yml` runs:

1. **Test:** install the package and run `pytest -q` for pushes and pull
   requests targeting `master`.
2. **Build:** build one image and publish both `latest` and commit-SHA tags to
   Docker Hub after a successful push to `master`.
3. **Deploy:** connect to EC2, download runtime data and model artifacts, replace
   the API/frontend containers, and run health checks.

### GitHub repository secrets

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
EC2_HOST
EC2_USER
EC2_SSH_KEY_B64
```

Encode the EC2 private key on Linux before storing it as
`EC2_SSH_KEY_B64`:

```bash
base64 -w 0 path/to/key.pem
```

On macOS:

```bash
base64 < path/to/key.pem | tr -d '\n'
```

### EC2 prerequisites

Install and enable Docker and the AWS CLI:

```bash
sudo apt update
sudo apt install -y docker.io awscli
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker

docker --version
aws --version
```

Create `$HOME/soh-app.env` on EC2:

```dotenv
MLFLOW_URI=http://host.docker.internal:5000
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
# Optional: pin a specific finished Data 4 run.
# MLFLOW_RUN_ID=266d92b1908b4182bdc492b2f3a9eacb
```

Do not add shell `export` statements or quotes to this Docker environment file.
When `MLFLOW_RUN_ID` is omitted, deployment selects the latest finished run
tagged `stage=data4` in the `soh-training` experiment. Disabling proxy multipart
download makes the artifact stream pass through MLflow instead of using a
presigned direct-S3 transfer from the application container.

The deployment downloads the processed fleet dataset from:

```text
s3://soh-project/soh-runtime/data/real_ev.csv
```

MLflow artifacts are stored under:

```text
s3://soh-project/mlflow-artifacts/
```

The EC2 instance role needs `s3:ListBucket` on the bucket, `s3:GetObject` for
the runtime CSV, and access to the MLflow artifact prefix. If the same MLflow
server uploads and manages artifacts, grant `GetObject`, `PutObject`,
`DeleteObject`, and `AbortMultipartUpload` on that prefix.

After deployment:

```bash
docker ps
docker logs --tail 100 soh-api
docker logs --tail 100 soh-frontend
curl --fail http://localhost:8000/health
curl --fail http://localhost:8501/_stcore/health
```

The deployed services listen on:

```text
FastAPI:   http://EC2_HOST:8000
Streamlit: http://EC2_HOST:8501
```

Allow the required ports in the EC2 security group. Only `8501` needs to be
public for dashboard users; port `8000` can remain restricted if it is not
consumed externally.

## Tests

Run the complete test suite:

```bash
pytest -q
```

The tests cover the SOH image codec, raw telemetry aggregation, nominal-capacity
estimation, latest-window selection, MLflow artifact resolution, and API
selection/error behavior.
