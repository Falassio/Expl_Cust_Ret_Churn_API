# Explainable Customer Retention & Churn API (xAI)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-ff69b4.svg)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-ready Machine Learning microservice that predicts customer churn risk and delivers **real-time local explainability (xAI)** using **SHAP TreeExplainer**.

Instead of returning an opaque churn probability score, this service breaks down *why* a customer is at risk—empowering Customer Success and Growth teams with actionable, feature-level drivers for retention intervention.

---

## 🎯 Key Highlights

- **Actionable Explainability (xAI)**: Integrates `shap.TreeExplainer` directly into the inference pipeline, computing exact Shapley contributions per feature in under 10ms.
- **Leakage-Free Preprocessing**: Scikit-learn `ColumnTransformer` bundling numerical standard scaling and categorical one-hot encoding into serialized artifacts.
- **High-Performance Classifier**: Gradient-boosted decision trees via `LightGBM` achieving **0.836+ ROC-AUC** on stratified validation sets.
- **Modern REST API**: Built on **FastAPI** and **Pydantic v2**, featuring strict schema validation, asynchronous lifespan management, and automated OpenAPI Swagger documentation.
- **Production & Cloud Ready**: Multi-platform `Dockerfile` with OpenMP acceleration (`libgomp1`), built-in health checks (`/health`), and seamless deployment on **Dokploy**, Docker Compose, or Kubernetes.
- **Thorough Test Suite**: 100% passing test coverage with `pytest` and `TestClient` verifying data generation, feature transformers, model training, and API endpoints.

---

## 🏗️ Architecture & Project Structure

```
Expl_Cust_Ret_Churn_API/
├── data/
│   └── telco_churn.csv              # Synthetic CRM / Telco dataset with realistic patterns
├── models/
│   ├── classifier.joblib            # Serialized Scikit-learn pipeline + LightGBM model
│   └── explainer.joblib             # Serialized SHAP TreeExplainer instance
├── src/
│   ├── __init__.py
│   ├── config.py                    # Environment settings, feature schemas, and risk tiers
│   ├── data_pipeline.py             # Data ingestion, synthetic generator, and ColumnTransformer
│   ├── explain.py                   # SHAP local attribution and business narrative engine
│   ├── train.py                     # Training orchestration, evaluation metrics, and model export
│   └── app.py                       # FastAPI application with Pydantic v2 data contracts
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py             # Unit tests for data pipeline, split, and training logic
│   └── test_api.py                  # Integration tests for /health, /predict, and validation
├── Dockerfile                       # Production container with OpenMP & healthcheck
├── docker-compose.yml               # Local orchestration & deployment config
├── requirements.txt                 # Pinned dependencies
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🧠 Machine Learning & Explainability Deep Dive

### 1. Feature Engineering & Preprocessing
Features are processed through a strictly fitted `ColumnTransformer` to prevent data leakage between training and inference:
- **Numerical Features** (`tenure_months`, `monthly_charges`, `total_tickets`): Scaled using `StandardScaler`.
- **Categorical Features** (`contract_type`, `payment_method`): Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.

### 2. Decision Tree Modeling with LightGBM
The classification engine leverages a tuned `LGBMClassifier` (150 estimators, max depth 5, learning rate 0.05). On hold-out test splits, it delivers balanced precision/recall trade-offs:
- **ROC-AUC Score**: `0.8364`
- **Overall Accuracy**: `77.8%`

### 3. Local Feature Attribution (SHAP)
For every incoming prediction payload, the API invokes the serialized `TreeExplainer` on the transformed vector. Feature impact scores represent log-odds deviations:
- **Positive SHAP Value ($>0$)**: Pushes the customer towards churning (e.g., elevated support ticket count, month-to-month contracts).
- **Negative SHAP Value ($<0$)**: Acts as a retention anchor (e.g., multi-year contract tenure, low dispute frequency).

---

## ⚡ Quickstart & Local Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Falassio/Expl_Cust_Ret_Churn_API.git
cd Expl_Cust_Ret_Churn_API

# Create and activate virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model & Generate Artifacts

```bash
python src/train.py
```
*This will generate the synthetic benchmark data, fit the transformer and LightGBM model, evaluate metrics, and export artifacts into `models/`.*

### 4. Run the API Server

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to explore interactive Swagger UI.

---

## 📡 API Reference

### Health Check
`GET /health`

Verifies that the microservice is healthy and that serialized ML artifacts are loaded in memory.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "service": "explainable-churn-api",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

### Predict & Explain Churn
`POST /predict`

Receives customer profile attributes and computes churn probability along with top 3 local risk drivers.

#### Request Example (High-Risk Customer):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 2,
       "monthly_charges": 115.0,
       "total_tickets": 6,
       "contract_type": "month-to-month",
       "payment_method": "electronic_check"
     }'
```

#### Response Example (`200 OK`):
```json
{
  "churn_risk_score": 0.9415,
  "will_churn": true,
  "risk_level": "High",
  "top_churn_drivers": [
    {
      "feature": "total_tickets",
      "impact_score": 2.1245,
      "description": "Increases churn risk"
    },
    {
      "feature": "contract_type_month-to-month",
      "impact_score": 1.0543,
      "description": "Increases churn risk"
    },
    {
      "feature": "tenure_months",
      "impact_score": -0.6321,
      "description": "Supports customer retention"
    }
  ]
}
```

#### Request Example (Loyal, Low-Risk Customer):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 65,
       "monthly_charges": 25.0,
       "total_tickets": 0,
       "contract_type": "two-year",
       "payment_method": "credit_card"
     }'
```

#### Response Example (`200 OK`):
```json
{
  "churn_risk_score": 0.0412,
  "will_churn": false,
  "risk_level": "Low",
  "top_churn_drivers": [
    {
      "feature": "total_tickets",
      "impact_score": -1.4521,
      "description": "Supports customer retention"
    },
    {
      "feature": "tenure_months",
      "impact_score": -1.2184,
      "description": "Supports customer retention"
    },
    {
      "feature": "contract_type_two-year",
      "impact_score": -0.9852,
      "description": "Supports customer retention"
    }
  ]
}
```

---

## 🧪 Testing

The repository includes comprehensive automated tests covering data generation, transformations, training pipelines, and REST endpoints:

```bash
pytest -v tests/
```

Test coverage includes:
- **`tests/test_pipeline.py`**: Synthetic data distributions, column transformations, training consistency, and SHAP attribution shapes.
- **`tests/test_api.py`**: Integration tests across `/health` and `/predict`, testing high-risk profiles, low-risk profiles, and Pydantic validation edge cases (`422 Unprocessable Entity`).

---

## 🐳 Docker & Cloud Deployment (Dokploy)

### Build & Run with Docker

```bash
# Build Docker image
docker build -t explainable-churn-api:latest .

# Run container exposing port 8000
docker run -d -p 8000:8000 --name explainable-churn-api explainable-churn-api:latest
```

### Run with Docker Compose

```bash
docker compose up --build -d
```

### Deploying on Dokploy
1. Create a new Application on your **Dokploy** instance.
2. Link your GitHub repository (`Falassio/Expl_Cust_Ret_Churn_API`).
3. Set **Build Type** to `Dockerfile` (or `Docker Compose`).
4. Set **Port** to `8000`.
5. Configure Health Check Path to `/health`.
6. Enable automatic deployments on branch `main`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
