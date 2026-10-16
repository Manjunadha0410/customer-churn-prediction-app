import warnings
import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns
from sklearn.model_selection import train_test_split
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score ,RocCurveDisplay
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import os
import sys
from dataclasses import dataclass
from src.exception import ProjectError
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train= train_array[:,:-1]
            y_train= train_array[:,-1]
            X_test= test_array[:,:-1]
            y_test= test_array[:,-1]

            def objective(trial):
                classifiers = trial.suggest_categorical("classifier",["XGBClassifier","CatBoostClassifier","LGBMClassifier","RandomForestClassifier","LogisticRegression"])
                if classifiers == "XGBClassifier":
                    params={"n_estimators": trial.suggest_int("n_XGB_estimators",100,400),
                    "max_depth": trial.suggest_int("max_XGB_depth",3,6),
                    "learning_rate": trial.suggest_float("learning_XGB_rate",0.01,0.2),
                    "subsample": trial.suggest_float("xgb_subsample",0.7,1.0),
                    "colsample_bytree": trial.suggest_float("xgb_colsample_bytree",0.7,1.0)
                    }
                    model = XGBClassifier(**params,random_state=42,n_jobs = 1)
                elif classifiers == "CatBoostClassifier":
                    params={
                        "iterations": trial.suggest_int("cat_iterations", 100, 600),
                        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.1, log=True),
                        "depth": trial.suggest_int("cat_depth", 3, 6), 
                        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0),
                        "bootstrap_type": "Bernoulli",
                        "subsample": trial.suggest_float("cat_subsample", 0.7, 1.0)
                    }
                    model = CatBoostClassifier(**params,random_state=42,thread_count= 1)
                elif classifiers == "LGBMClassifier":

                    params = {
                        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 400),
                        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.1, log=True),
                        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 6),
                        "subsample": trial.suggest_float("lgbm_subsample", 0.7, 1.0),
                        "subsample_freq": 1,
                        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.7, 1.0),
                    }
                    model = LGBMClassifier(**params,random_state=42,n_jobs = 1)
                elif classifiers == "RandomForestClassifier":
                    params ={
                        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 400),
                        "max_depth": trial.suggest_int("rf_max_depth", 3, 6),
                        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
                        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None])
                    }
                    model = RandomForestClassifier(**params,random_state=42,n_jobs = 1)
                elif classifiers == "LogisticRegression":
                    params = {
                        "penalty": trial.suggest_categorical("lr_penalty", ["l1", "l2"]),
                        "C": trial.suggest_float("lr_C", 1e-4, 10.0, log=True),
                        "solver": "saga",
                        "max_iter": trial.suggest_int("lr_max_iter", 500, 1500),
                    }
                    model = LogisticRegression(**params,random_state=42)
                    
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model,X_train,y_train,cv=cv,scoring="accuracy",n_jobs=-1)
                return scores.mean()  
            
            study = optuna.create_study(direction = "maximize")
            study.optimize(objective, n_trials=50)
            
            best_params = study.best_params
            best_classifier = best_params["classifier"]           
            if best_classifier == "XGBClassifier":
                params={"n_estimators": best_params["n_XGB_estimators"],
                        "max_depth": best_params["max_XGB_depth"],
                        "learning_rate": best_params["learning_XGB_rate"],
                        "subsample": best_params["xgb_subsample"],
                        "colsample_bytree": best_params["xgb_colsample_bytree"]
                        }
                model = XGBClassifier(**params,random_state=42,n_jobs = 1)
            elif best_classifier == "CatBoostClassifier":
                params={
                "iterations": best_params["cat_iterations"],
                "learning_rate": best_params["cat_learning_rate"],
                "depth": best_params["cat_depth"], 
                "l2_leaf_reg": best_params["cat_l2_leaf_reg"], 
                "bootstrap_type": "Bernoulli",
                "subsample": best_params["cat_subsample"]
                }
                model = CatBoostClassifier(**params,random_state=42,thread_count= 1)
            elif best_classifier == "LGBMClassifier":
                params = {
                "n_estimators": best_params["lgbm_n_estimators"],
                "learning_rate": best_params["lgbm_learning_rate"],
                "max_depth": best_params["lgbm_max_depth"],
                "subsample": best_params["lgbm_subsample"],
                "subsample_freq": 1,
                "colsample_bytree": best_params["lgbm_colsample_bytree"],
                }
                model = LGBMClassifier(**params,random_state=42,n_jobs = 1)
            elif best_classifier == "RandomForestClassifier":
                params ={
                "n_estimators": best_params["rf_n_estimators"],
                "max_depth": best_params["rf_max_depth"],
                "min_samples_split":best_params["rf_min_samples_split"],
                "min_samples_leaf": best_params["rf_min_samples_leaf"],
                "max_features": best_params["rf_max_features"]
                }
                model = RandomForestClassifier(**params,random_state=42,n_jobs = 1)
            elif best_classifier == "LogisticRegression":
                params = {
                "penalty": best_params["lr_penalty"],
                "C": best_params["lr_C"],
                "solver": "saga",
                "max_iter": best_params["lr_max_iter"]
                }
                model = LogisticRegression(**params,random_state=42)

            model.fit(X_train, y_train)
            acc_score = accuracy_score(y_test, model.predict(X_test))

            if acc_score<0.6:
                raise ProjectError("No good was model found",sys)
            logging.info(f"A Good enough model found for the dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=model
            )
            return acc_score
                   
        except Exception as e:
            raise ProjectError(e,sys)