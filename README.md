### Student-Performance-Prediction-Production-ML-Pipeline-with-CI-CD

# Student Performance Prediction System 
A **production-ready machine learning system** with a modular pipeline architecture.  
The application predicts student academic performance based on demographic and academic attributes.  
It includes a **full CI/CD pipeline** with continuous integration, containerization, and deployment on **AWS (ECR + EC2)**.  

## 🚀 Features
- **Prediction System**: Predicts student performance based on multiple features  
- **Web Interface**: Simple form-based input for predictions (Flask)  
- **Production-Ready ML Pipeline**: Modular pipeline for ingestion, transformation, training, and prediction  
- **Pre-Trained Models**: Includes CatBoost and XGBoost models  
- **CI/CD Workflow**: Automated linting, Docker build & push, AWS deployment  
- **Containerized Deployment**: Runs inside Docker with Amazon ECR + EC2  

## 🛠️ Tech Stack
- **Programming**: Python (3.8+)  
- **Frameworks**: Flask  
- **ML Models**: Scikit-learn, XGBoost, CatBoost  
- **Pipelines**: Custom modular production-grade ML pipeline system  
- **DevOps**: Docker, GitHub Actions, AWS ECR, EC2  
- **Version Control**: Git, GitHub  

## 📂 Project Structure


```
mlproject-main/
├── src/                  # Core ML pipeline system (ingestion, transformation, training, prediction)
│   ├── components/       # Reusable components for pipeline stages
│   ├── pipeline/         # Training & prediction pipelines (production-ready)
│   ├── exception.py      # Custom exception handling
│   └── **init**.py
├── templates/            # HTML templates for web app
├── artifacts/            # Stored models, preprocessors
├── notebook/            
├── .github/workflows/    # CI/CD workflow configs
├── app.py                # Web application entry point
├── Dockerfile            # Containerization config
├── requirements.txt      # Python dependencies
└── setup.py              # Package setup

```

 🔑 **Note**: The modular structure enables scalability, maintainability, and ease of extending pipelines for production workloads.  

