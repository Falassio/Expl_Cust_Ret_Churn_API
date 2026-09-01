# Explainable Customer Retention & Churn API (xAI Focus)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-xAI-ff69b4.svg)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

Un microservizio di Machine Learning end-to-end per la predizione del rischio di abbandono clienti (**churn**) in ambito CRM / e-commerce, integrato con **Explainable AI (SHAP TreeExplainer)** per restituire in tempo reale non solo la probabilità di churn, ma anche l'impatto locale di ciascuna feature sul singolo utente.

---

## 🏛️ Architettura del Progetto

```
Expl_Cust_Ret_Churn_API/
├── data/
│   └── telco_churn.csv         # Dataset CRM / Telco (generato o reale)
├── models/
│   ├── classifier.joblib       # Pipeline Scikit-Learn + Modello LightGBM serializzato
│   └── explainer.joblib        # SHAP TreeExplainer serializzato
├── src/
│   ├── __init__.py
│   ├── config.py               # Configurazione percorsi, feature e soglie di rischio
│   ├── data_pipeline.py        # Generazione dati sintetici e ColumnTransformer
│   ├── train.py                # Pipeline di addestramento, valutazione e serializzazione
│   ├── explain.py              # Logica xAI e calcolo SHAP values locali
│   └── app.py                  # API RESTful FastAPI con validazione Pydantic v2
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py        # Unit test per data pipeline e training
│   └── test_api.py             # Integration test per gli endpoint FastAPI
├── Dockerfile                  # Container Linux ottimizzato con OpenMP (libgomp1)
├── docker-compose.yml          # Setup per orchestrazione e test locale
├── requirements.txt            # Dipendenze Python
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🔬 Explainable AI (xAI) & Machine Learning

1. **Pre-processing Leakage-Free:**
   - Variabili numeriche (`tenure_months`, `monthly_charges`, `total_tickets`) scalate con `StandardScaler`.
   - Variabili categoriche (`contract_type`, `payment_method`) codificate con `OneHotEncoder(drop='first', handle_unknown='ignore')`.
2. **Classificatore LightGBM:**
   - Algoritmo ad alberi di decisione con boosting ottimizzato per dati tabulari ad alte prestazioni e bassa latenza di inferenza.
3. **Spiegabilità SHAP (SHapley Additive exPlanations):**
   - L'algoritmo `shap.TreeExplainer` calcola i valori Shapley per ciascuna feature sulla singola predizione.
   - Fornisce sia il valore numerico dell'impatto (positivo per incremento rischio, negativo per fidelizzazione) sia l'ordinamento decrescente dei top driver.

---

## 🚀 Setup Locale ed Esecuzione

### 1. Creazione ambiente virtuale & Installazione dipendenze

```bash
# Crea il virtual environment
python -m venv venv

# Attiva l'ambiente virtuale
# Su Windows:
venv\Scripts\activate
# Su Linux/macOS:
source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
```

### 2. Addestramento del Modello

```bash
python src/train.py
```
Questo comando:
- Genera / carica il dataset in `data/telco_churn.csv`.
- Addestra il modello LightGBM valutando ROC-AUC ed Accuracy.
- Salva gli artefatti in `models/classifier.joblib` e `models/explainer.joblib`.

### 3. Avvio del Server API

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
La documentazione interattiva Swagger UI sarà disponibile su: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🧪 Esecuzione dei Test

Per eseguire l'intera suite di test unitari e di integrazione:

```bash
pytest -v tests/
```

---

## 📡 API Endpoints

### 1. `GET /health`
Verifica lo stato del servizio e il corretto caricamento dei modelli in memoria.

**Risposta di esempio:**
```json
{
  "status": "ok",
  "service": "explainable-churn-api",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

### 2. `POST /predict`
Effettua la predizione del rischio di churn e calcola i top 3 driver con SHAP.

#### Esempio Richiesta (Cliente ad Alto Rischio):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 2,
       "monthly_charges": 110.0,
       "total_tickets": 5,
       "contract_type": "month-to-month",
       "payment_method": "electronic_check"
     }'
```

#### Esempio Risposta:
```json
{
  "churn_risk_score": 0.8921,
  "will_churn": true,
  "risk_level": "Alto",
  "top_churn_drivers": [
    {
      "feature": "total_tickets",
      "impact_score": 2.4512,
      "description": "Aumenta il rischio di abbandono"
    },
    {
      "feature": "contract_type_month-to-month",
      "impact_score": 1.1824,
      "description": "Aumenta il rischio di abbandono"
    },
    {
      "feature": "tenure_months",
      "impact_score": -0.8412,
      "description": "Fattore di fidelizzazione"
    }
  ]
}
```

---

## 🐳 Containerizzazione & Deploy con Dokploy

### Build ed Esecuzione con Docker:

```bash
# Build dell'immagine Docker
docker build -t explainable-churn-api:latest .

# Esecuzione container
docker run -d -p 8000:8000 --name explainable-churn-api explainable-churn-api:latest
```

### Build con Docker Compose:

```bash
docker compose up --build -d
```

### Configurazione su Dokploy:
1. Collega il repository GitHub su **Dokploy** (Application o Docker Compose).
2. Seleziona **Build Type**: `Dockerfile` (o `Docker Compose`).
3. Imposta **Port**: `8000`.
4. Path di Healthcheck: `/health`.
5. Deploy automatico attivato su push del branch `main`.

---

## 🛡️ Gestione Errori e Validazione

La validazione dell'input è garantita dagli schemi Pydantic V2:
- `tenure_months` e `total_tickets`: interi $\ge 0$.
- `monthly_charges`: float $\ge 0.0$.
- `contract_type`: limitato a `['month-to-month', 'one-year', 'two-year']`.
- `payment_method`: limitato a `['electronic_check', 'bank_transfer', 'credit_card']`.
- Input errati restituiscono status code `422 Unprocessable Entity` con spiegazione dettagliata del campo non valido.
