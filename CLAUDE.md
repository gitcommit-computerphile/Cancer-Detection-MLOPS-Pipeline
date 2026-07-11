# Mini Pipeline — CLAUDE.md

A minimal end-to-end MLOps project for practicing the full pipeline stack without
the scale of the credit card project. Same architecture, same tools, trains in under a minute.

## Goal

Practice every MLOps concept once, end to end:
MongoDB → pipeline → MLflow → FastAPI → Docker → GitHub Actions → ECR → EC2

## ML Problem

**Breast Cancer Detection** — binary classification (malignant = 1, benign = 0).
Dataset: `sklearn.datasets.load_breast_cancer()` — 569 rows, 30 numeric features, no download needed.
Target column: `target` (already binary, no relabeling).
No class imbalance issue — no SMOTE needed.

## Is this a complete MLOps pipeline?

Yes. Every layer is wired:

| Layer | Tool | Status |
|---|---|---|
| Data store | MongoDB Atlas | Code complete — needs Atlas cluster |
| Experiment tracking | MLflow + DagsHub | Code complete — needs DagsHub repo |
| Artifact / model store | AWS S3 | Code complete — needs S3 bucket |
| API | FastAPI | Code complete — runs locally now |
| Containerization | Docker | Code complete — Dockerfile present |
| Image registry | AWS ECR | Wired in CI/CD — needs ECR repo created |
| Deployment | AWS EC2 (self-hosted runner) | Wired in CI/CD — needs EC2 instance + runner |
| CI/CD | GitHub Actions | `.github/workflows/main.yaml` written — triggers on `git push` to `main` |

The only things not yet done are the one-time AWS/DagsHub resource setup (creating the Atlas cluster,
S3 bucket, ECR repo, EC2 instance). All the code that talks to them is already written.

## Recommended reading order

Read these before editing any file — each builds on the previous:

1. [cancer/constant/training_pipeline/__init__.py](cancer/constant/training_pipeline/__init__.py) — every tunable constant (paths, ratios, model thresholds, bucket name). Read this first; every other file pulls from here.
2. [cancer/entity/config_entity.py](cancer/entity/config_entity.py) — the five config dataclasses that define folder paths per pipeline step. Changing a path here changes it everywhere.
3. [cancer/entity/artifact_entity.py](cancer/entity/artifact_entity.py) — the artifact dataclasses each component returns. These are the "contracts" between pipeline steps.
4. [cancer/components/data_ingestion.py](cancer/components/data_ingestion.py) — reads from MongoDB Atlas, stratified train/test split on `target`.
5. [cancer/components/data_validation.py](cancer/components/data_validation.py) — column count check and KS drift detection (warning only, not blocking).
6. [cancer/components/data_transformation.py](cancer/components/data_transformation.py) — fits KNNImputer + StandardScaler on train, transforms both splits, saves preprocessor.pkl.
7. [cancer/components/model_trainer.py](cancer/components/model_trainer.py) — GridSearchCV over RandomForest / LogisticRegression / DecisionTree (CV=3), selects by F1, logs to MLflow, saves `final_model/model.pkl`.
8. [cancer/pipeline/training_pipeline.py](cancer/pipeline/training_pipeline.py) — orchestrates steps 4–7 end to end and syncs artifacts + model to S3.
9. [cancer/pipeline/batch_prediction.py](cancer/pipeline/batch_prediction.py) — loads `final_model/model.pkl` and scores a CSV.
10. [app.py](app.py) — FastAPI: `GET /train` triggers the pipeline, `POST /predict` scores an uploaded CSV, `GET /` shows the results UI.
11. [push_data.py](push_data.py) and [main.py](main.py) — CLI entry points.
12. [.github/workflows/main.yaml](.github/workflows/main.yaml) — CI/CD: lint → build Docker → push to ECR → pull on EC2 → run container.

Supporting modules (read as needed):
- [cancer/utils/main_utils/utils.py](cancer/utils/main_utils/utils.py) — yaml/pkl/npy I/O + `evaluate_models` (the GridSearchCV loop).
- [cancer/utils/ml_utils/metric/classification_metric.py](cancer/utils/ml_utils/metric/classification_metric.py) — F1/precision/recall.
- [cancer/utils/ml_utils/model/estimator.py](cancer/utils/ml_utils/model/estimator.py) — `BreastCancerModel` wrapper (preprocessor + model in one object).
- [cancer/cloud/s3_syncer.py](cancer/cloud/s3_syncer.py) — thin `aws s3 sync` wrapper.
- [data_schema/schema.yaml](data_schema/schema.yaml) — 31 expected columns consumed by data_validation.

## Execution steps — full walkthrough

### Phase 1: Local setup (no AWS needed)

```bash
# 1. Install dependencies
cd mini_pipeline
pip install -e .
pip install -r requirements.txt

# 2. Create .env with your credentials
cp .env.example .env
# Edit .env — fill in MONGODB_URL_KEY at minimum; MLflow/AWS optional for local testing
```

**What you need for Phase 1:**
- MongoDB Atlas free cluster (M0) — create at cloud.mongodb.com, get connection string

### Phase 2: Push data to MongoDB

```bash
python push_data.py
```

**What it does:** Loads sklearn's breast cancer dataset (569 rows, 30 features) and inserts
all records into Atlas collection `cancer.breast_cancer`. Safe to run repeatedly — skips if
collection already has data.

**Verify:** Open MongoDB Atlas → Browse Collections → `cancer.breast_cancer` — you should see 569 documents.

### Phase 3: Run the training pipeline

```bash
python main.py
```

**What it does (4 steps, ~30 seconds):**
1. **DataIngestion** — reads 569 rows from MongoDB, saves feature store CSV, splits 80/20 → `Artifacts/<timestamp>/data_ingestion/`
2. **DataValidation** — checks column count (31), runs KS drift test, writes `drift_report/report.yaml` → `Artifacts/<timestamp>/data_validation/`
3. **DataTransformation** — fits KNNImputer + StandardScaler on train split only, saves `.npy` arrays and `preprocessing.pkl` → `Artifacts/<timestamp>/data_transformation/`
4. **ModelTrainer** — GridSearchCV over 3 models (CV=3), picks best by F1, logs metrics to MLflow, saves `final_model/model.pkl` and `final_model/preprocessor.pkl`

**After this you have:** A working `final_model/` and a full `Artifacts/<timestamp>/` tree.

**Verify MLflow:** Check your DagsHub repo → Experiments — you should see a new run with train/test F1 metrics.

### Phase 4: Run the API

```bash
python app.py
# FastAPI starts on http://localhost:8000
```

**Endpoints:**
| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Web UI — shows prediction results table |
| `/train` | GET | Runs full training pipeline, returns F1 scores |
| `/predict` | POST | Upload a CSV → returns predictions as HTML table |
| `/docs` | GET | Auto-generated Swagger UI |

**To test `/predict`:** Export any rows from the feature store CSV (drop the `target` column), upload it at `http://localhost:8000`.

### Phase 5: CI/CD setup (one-time AWS resource creation)

Do these once in the AWS console / CLI before `git push` will work:

```
1. Create S3 bucket:   mini-pipeline-mlops   (us-east-1, private)
2. Create ECR repo:    mini-pipeline          (us-east-1)
3. Launch EC2:         t2.micro, Amazon Linux 2, open port 8000 inbound
4. Install Docker on EC2:  sudo yum install -y docker && sudo service docker start
5. Install GitHub Actions runner on EC2:
   → GitHub repo → Settings → Actions → Runners → New self-hosted runner → follow Linux steps
6. Create IAM user with AmazonS3FullAccess + AmazonEC2ContainerRegistryFullAccess
7. Create DagsHub repo named "mini_pipeline" — get MLflow tracking URI from it
```

**Add GitHub Secrets** (repo → Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | `us-east-1` |
| `AWS_ECR_LOGIN_URI` | `<account-id>.dkr.ecr.us-east-1.amazonaws.com/mini-pipeline` |
| `ECR_REPOSITORY_NAME` | `mini-pipeline` |
| `MONGODB_URL_KEY` | same as your `.env` |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<username>/mini_pipeline.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token |

### Phase 6: Deploy via CI/CD

```bash
git init
git remote add origin https://github.com/<you>/mini_pipeline.git
git add .
git commit -m "initial commit"
git push origin main
```

**What GitHub Actions does automatically:**
1. **Lint** — flake8 syntax check on the push
2. **Build** — Docker image built from `Dockerfile`
3. **Push** — image pushed to ECR as `:latest`
4. **Deploy** — self-hosted runner on EC2 pulls the image, stops old container, starts new one on port 8000

**Live URL after deploy:** `http://<ec2-public-ip>:8000`

## Project structure

```
mini_pipeline/
├── cancer/                      main package
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── entity/
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── batch_prediction.py
│   ├── utils/
│   │   ├── main_utils/utils.py
│   │   └── ml_utils/
│   │       ├── metric/classification_metric.py
│   │       └── model/estimator.py
│   ├── cloud/s3_syncer.py
│   ├── constant/training_pipeline/__init__.py
│   ├── exception/exception.py
│   └── logging/logger.py
├── data_schema/schema.yaml
├── templates/table.html
├── final_model/                 generated, gitignored
├── Artifacts/                   generated, gitignored
├── app.py                       FastAPI: /, /train, /predict
├── main.py                      CLI: runs training pipeline
├── push_data.py                 one-time: loads breast cancer data into MongoDB
├── setup.py                     makes cancer/ importable from project root
├── .github/workflows/main.yaml  CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── .env                         credentials, gitignored
└── .env.example
```

## Key design points when editing

- **Target column is `target`** (1 = benign, 0 = malignant) — already binary, no relabeling.
- **Stratify on `target`** at the train/test split — keeps class balance in both splits.
- **Preprocessor fits on train only** — never fit on test or full data.
- **KS drift is a warning, not a blocking failure** — with 569 rows it will often fire on a random split.
- **`BreastCancerModel` bundles preprocessor + model** — `final_model/model.pkl` is this wrapper, not the raw classifier. The `/predict` endpoint depends on this.
- **Do not hardcode credentials** — everything reads from `.env` via `load_dotenv()`.

## What you will have practiced after completing all 6 phases

- Pushing structured data to MongoDB and reading it back in a pipeline
- Config dataclasses and artifact dataclasses as pipeline contracts
- KS drift detection and schema validation
- MLflow experiment tracking logged to DagsHub
- S3 artifact and model sync
- FastAPI `/train` and `/predict` endpoints
- Dockerizing a FastAPI app
- Pushing a Docker image to AWS ECR via GitHub Actions
- Running a self-hosted GitHub Actions runner on EC2
- Full CI/CD: code push → lint → build → deploy → live endpoint