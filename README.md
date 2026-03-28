# Bank Loan Prediction System (ML + API + Docker)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

## Project Overview

This project is an **end-to-end Machine Learning system** that predicts whether a customer is likely to take a loan.

It is designed as a **production-ready application** with:
- Trained ML model  
- REST API using Flask  
- Docker containerization  

Kubernetes and CI/CD configurations are included for **future scalability**.

## Problem Statement

Banks need to identify potential customers who are likely to subscribe to a loan product.

This system helps in:
- Targeted marketing  
- Reducing customer acquisition cost  
- Improving conversion rates  

## Key Features

- Machine Learning model (Random Forest)  
- Data preprocessing pipeline  
- REST API using Flask  
- Input validation & error handling  
- Probability-based predictions  
- Risk categorization (VERY LOW → HIGH)  
- Dockerized application (production-ready)  
- Kubernetes deployment configs (optional)  
- CI/CD pipeline configuration  

## Machine Learning Details

| Item | Detail |
|------|--------|
| **Algorithm** | Random Forest Classifier (Logistic Regression baseline) |
| **Features** | Age, job, marital status, education, balance, housing loan, contact type, campaign details |
| **Target** | Binary: 1 = likely, 0 = unlikely |
| **Preprocessing** | Imputation, Encoding, Scaling |
| **Train/Test Split** | 80/20 |
| **Class Imbalance** | class_weight="balanced" |
| **Evaluation** | Accuracy, Precision, Recall, F1 |
| **Serialization** | joblib |

---

## 📁 Project Structure

```bash
ML/
│
├── app/
│   ├── app.py
│   ├── model_loader.py
│   └── utils.py
│
├── model/
│   ├── train.py
│   ├── preprocess.py
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── data/
│   └── dataset.csv
│
├── docker/
│   └── Dockerfile
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── ci-cd/
│   └── buildspec.yml
│
├── requirements.txt
└── README.md
```
## Tech Stack

- Python  
- Scikit-learn  
- Flask  
- Docker  
- Kubernetes (configs only)  
- CI/CD (configs only)  


## Machine Learning Details

- **Model Used:** Random Forest Classifier  
- **Baseline:** Logistic Regression  
- **Metrics:** Accuracy, Precision, Recall, F1 Score  

 Random Forest selected based on best F1 score.

## Quick Start — Local Development

- Model Used: Random Forest Classifier
- Baseline Model: Logistic Regression
- Metrics: Accuracy, Precision, Recall, F1 Score


Random Forest was selected based on higher F1 score.

## Workflow

User Input → API → Validation → Preprocessing → Model → Prediction → Response


## Running the Project

### Clone Repository
```bash
git clone https://github.com/your-username/bank-loan-ml-system.git
cd bank-loan-ml-system
```

### Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### Install 
```bash
pip install -r requirements.txt
```
### Train Model
```bash
python model/train.py
```
### Run API
```basg
python app/app.py

```
## Test API
```bash
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{"age":30,"job":"admin.","marital":"single","education":"tertiary","default":"no","balance":5000,"housing":"yes","loan":"no","contact":"cellular","day":5,"month":"may","duration":200,"campaign":1,"pdays":-1,"previous":0,"poutcome":"unknown"}'
```
## Docker Deployment

### Build Image
```bash
docker build -t bank-loan-app -f docker/Dockerfile .
```
### Run Container
```bash
docker run -p 5001:5000 bank-loan-app
```
## Sample Output
```bash
{
  "prediction_label": "No - Unlikely to take loan",
  "probability_of_loan": 0.22,
  "risk_band": "VERY LOW"
}
```
## Docker — Build & Run Locally

```bash
# Build the image (run from project root)
docker build -t bank-loan-ml:latest -f docker/Dockerfile .

# Run the container
docker run -p 5000:5000 bank-loan-ml:latest

# Test it
curl http://localhost:5000/health
```


## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness/readiness probe |
| `GET` | `/info` | Service metadata |
| `POST` | `/predict` | Loan prediction |

### POST /predict — Request Schema

| Field | Type | Required | Example |
|-------|------|----------|---------|
| `age` | int | ✅ | `35` |
| `job` | string | ✅ | `"management"` |
| `marital` | string | ✅ | `"married"` |
| `education` | string | ✅ | `"tertiary"` |
| `default` | string | ✅ | `"no"` |
| `balance` | number | ✅ | `3000` |
| `housing` | string | ✅ | `"yes"` |
| `loan` | string | ✅ | `"no"` |
| `contact` | string | ✅ | `"cellular"` |
| `day` | int | ✅ | `15` |
| `month` | string | ✅ | `"may"` |
| `duration` | int | ✅ | `300` |
| `campaign` | int | ✅ | `1` |
| `pdays` | int | ✅ | `-1` |
| `previous` | int | ✅ | `0` |
| `poutcome` | string | ✅ | `"unknown"` |


## Kubernetes:

Kubernetes deployment and service YAML files are included for learning and future scalability.The project is currently tested locally using Docker.

## Unique_aspects:
  - End-to-end ML pipeline
  - Production-ready REST API
  - Strong input validation
  - Business-oriented output (risk and probability)
  - Dockerized deployment
  - Modular and scalable architecture
  - Kubernetes and CI/CD configs included for future use

## Future_improvements:
  - Cloud deployment (AWS or GCP)
  - Add frontend UI
  - Add monitoring and logging
  - Automate model retraining

##  Unique Aspects
- End-to-end ML pipeline
- Production-ready API
- Dockerized deployment
- Business-oriented output
- Kubernetes-ready architecture
- CI/CD pipeline included


