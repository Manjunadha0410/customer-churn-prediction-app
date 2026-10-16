# Customer Churn Prediction
Production ML app for predicting customer churn using boosting models with automated CI/CD pipeline and aws deployment

## Features
- Multi-model ensemble
- Hyperparameter optimization with Optuna
- Docker containerization
- AWS ECR deployment
- GitHub Actions for CI/CD
- Fastapi for API development

## Tech Stack
- Python
- Fastapi
- Docker
- AWS ECR
- GitHub Actions

## Installation

```
git clone https://github.com/Manjunadha0410/customer-churn-prediction-app.git
cd customer-churn-prediction-app
pip install -r requirements.txt
python app.py
```

or Use Docker:
```
docker build -t customer-churn-prediction-app .
docker run -p 8000:8000 customer-churn-prediction-app
```

Required GitHub Secrets
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- ECR_REPOSITORY_NAME
- AWS_ECR_LOGIN_UR

# Project Structure
```
├── src/
│   ├── components/         
│   └── pipeline/           
├── artifacts/               
├── templates/             
├── app.py
├── Dockerfile
└── requirements.txt
```
