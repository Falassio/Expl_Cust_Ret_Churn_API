# Explainable Customer Retention & Churn API (xAI)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-churn--api.alessiofalanga.it-success?style=flat-square&logo=fastapi)](https://churn-api.alessiofalanga.it/)
[![Interactive Docs](https://img.shields.io/badge/Swagger%20UI-API%20Docs-blue?style=flat-square&logo=swagger)](https://churn-api.alessiofalanga.it/docs)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7%2B-brightgreen.svg?style=flat-square)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-ff69b4.svg?style=flat-square)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

Production-grade Machine Learning microservice that predicts customer churn probability and returns **real-time local explainability (xAI)** using **SHAP TreeExplainer**.

Rather than outputting a black-box churn score, the service decomposes prediction drivers into exact feature contributions—giving Customer Success and Growth teams actionable insights to design tailored retention strategies.

🌐 **Live API**: [https://churn-api.alessiofalanga.it/](https://churn-api.alessiofalanga.it/)  
📖 **API Documentation**: [https://churn-api.alessiofalanga.it/docs](https://churn-api.alessiofalanga.it/docs)

---

## 📌 Architecture & System Flow

```
                      +-----------------------------+
                      |   Client Request (JSON)     |
                      +--------------+--------------+
                                     |
                                     v
                        +------------+------------+
                        |  FastAPI + Pydantic v2  |
                        +------------+------------+
                                     |
                                     v
                        +------------+------------+
                        | Leak-Free Preprocessing |
                        |   (ColumnTransformer)   |
                        +------------+------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
    +-----------+-----------+                 +-----------+-----------+
    |  LightGBM Classifier  |                 |  SHAP TreeExplainer   |
    |  (Churn Probability)  |                 |  (Local Attribution)  |
    +-----------+-----------+                 +-----------+-----------+
                |                                         |
                +--------------------+--------------------+
                                     |
                                     v
                      +--------------+--------------+
                      |  Ranked Risk Drivers & JSON |
                      |    Explainable Response     |
                      +-----------------------------+
```

---

## ⚡ Key Highlights

- **Real-Time Local Explainability (xAI)**: Integrated `shap.TreeExplainer` calculates per-feature Shapley attributions in sub-10ms latency.
- **Leakage-Free Feature Pipeline**: Scikit-learn `ColumnTransformer` encapsulating standard numerical scaling and one-hot encoding into a unified serialized pipeline.
- **High-Performance Classifier**: Gradient-boosted decision trees via `LightGBM` achieving **>0.83 ROC-AUC** on stratified validation sets.
- **Strict Data Contracts**: Built with **FastAPI** and **Pydantic v2**, including schema validation, asynchronous lifecycle management (`lifespan`), and automated Swagger UI documentation.
- **Containerized & Production Ready**: Multi-platform `Dockerfile` optimized with OpenMP acceleration (`libgomp1`), built-in health monitoring (`/health`), and automated deployment support via **Dokploy** or Docker Compose.
- **Automated Test Suite**: 100% test coverage with `pytest` and `TestClient` validating synthetic data generation, transformers, training logic, and API endpoints.

---

## 📁 Repository Structure

```
Expl_Cust_Ret_Churn_API/
├── data/
│   └── telco_churn.csv              # Benchmark CRM / Telco customer dataset
├── models/
│   ├── classifier.joblib            # Serialized Scikit-learn pipeline + LightGBM model
│   └── explainer.joblib             # Serialized SHAP TreeExplainer instance
├── src/
│   ├── __init__.py
│   ├── app.py                       # FastAPI application & endpoint definitions
│   ├── config.py                    # Environment settings, feature schemas, risk thresholds
│   ├── data_pipeline.py             # Data loading, synthetic generator, ColumnTransformer
│   ├── explain.py                   # SHAP local attribution and narrative synthesis
│   └── train.py                     # Training orchestration, metrics evaluation, model export
├── tests/
│   ├── __init__.py
│   ├── test_api.py                  # Integration tests for /health, /predict, edge cases
│   └── test_pipeline.py             # Unit tests for preprocessing, training, and SHAP output
├── Dockerfile                       # Production container with OpenMP & healthcheck
├── docker-compose.yml               # Local orchestration config
├── requirements.txt                 # Pinned dependencies
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🧠 Machine Learning & Explainability Details

### 1. Data Pipeline & Preprocessing
Features are processed through a strictly fitted `ColumnTransformer` to prevent data leakage:
- **Numerical Features** (`tenure_months`, `monthly_charges`, `total_tickets`): Scaled via `StandardScaler`.
- **Categorical Features** (`contract_type`, `payment_method`): Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.

### 2. LightGBM Classification Engine
The classification layer uses a tuned `LGBMClassifier` (150 estimators, max depth 5, learning rate 0.05).
- **ROC-AUC**: `0.8364`
- **Balanced Accuracy**: `77.8%`

### 3. Local Feature Attribution (SHAP)
For every inference request, the pipeline evaluates the feature vector through the pre-computed `TreeExplainer`:
- **Positive SHAP Value ($>0$)**: Drives risk higher (e.g., high support ticket frequency, month-to-month contracts).
- **Negative SHAP Value ($<0$)**: Acts as a retention anchor (e.g., high tenure, annual billing, low dispute rate).

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Falassio/Expl_Cust_Ret_Churn_API.git
cd Expl_Cust_Ret_Churn_API

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model & Export Artifacts

```bash
python src/train.py
```
*Generates dataset, trains LightGBM classifier, computes SHAP explainer, and exports `.joblib` artifacts to `models/`.*

### 4. Start the API Server

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** to test the interactive Swagger documentation.

---

## 📡 API Reference

### Health Check
`GET /health`

Verifies that the service is running and all model artifacts are loaded in memory.

```bash
curl http://localhost:8000/health
```

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

Accepts customer attributes and outputs predicted churn probability, risk classification tier, and the top local SHAP drivers.

#### Example: High-Risk Profile

```bash
curl -X POST "https://churn-api.alessiofalanga.it/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 2,
       "monthly_charges": 115.0,
       "total_tickets": 6,
       "contract_type": "month-to-month",
       "payment_method": "electronic_check"
     }'
```

**Response (`200 OK`):**
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

#### Example: Low-Risk / Loyal Profile

```bash
curl -X POST "https://churn-api.alessiofalanga.it/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 65,
       "monthly_charges": 25.0,
       "total_tickets": 0,
       "contract_type": "two-year",
       "payment_method": "credit_card"
     }'
```

**Response (`200 OK`):**
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

## 🧪 Automated Testing

Run the test suite with `pytest`:

```bash
pytest -v tests/
```

Test coverage includes:
- **`tests/test_pipeline.py`**: Dataset generation distribution, transformer validation, training pipeline output, SHAP matrix shape and properties.
- **`tests/test_api.py`**: Endpoint integration tests for `/health`, `/predict` across high-risk, low-risk, and invalid payload scenarios (`422 Unprocessable Entity`).

---

## 🐳 Containerization & Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t explainable-churn-api:latest .

# Run container
docker run -d -p 8000:8000 --name explainable-churn-api explainable-churn-api:latest
```

### Docker Compose

```bash
docker compose up --build -d
```

### Deployment on Dokploy
1. Create a new Application inside your Dokploy dashboard.
2. Link the repository (`Falassio/Expl_Cust_Ret_Churn_API`).
3. Set **Build Type** to `Dockerfile`.
4. Configure port `8000` and healthcheck path `/health`.
5. Automatic builds will trigger on pushes to the `main` branch.

---

## 💼 Technical Interview & Design Rationales

1. **Why SHAP TreeExplainer over KernelExplainer or LIME?**  
   * TreeExplainer leverages the internal tree structures of LightGBM to compute exact Shapley values in polynomial time $\mathcal{O}(TLD^2)$, achieving single-digit millisecond latency at inference time. KernelExplainer and LIME rely on sampling heuristics and are too slow for synchronous online APIs.
2. **How is data leakage prevented?**  
   * Preprocessing parameters (such as `StandardScaler` mean/variance and `OneHotEncoder` categories) are strictly fitted on the training split inside a Scikit-learn pipeline and serialized to disk. The API only applies `.transform()`, ensuring zero test/serving leakage.
3. **Why LightGBM for tabular customer retention?**  
   * Tree-based gradient boosting consistently outperforms deep learning on tabular data with mixed continuous/categorical features, handles non-linear boundaries gracefully, and natively integrates with tree-based explainability frameworks.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
