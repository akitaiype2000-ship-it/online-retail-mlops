import os
import joblib
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


class ModelTrainer:

    def __init__(self):
        self.input_path = "artifacts/customer_data.csv"
        self.model_dir = "artifacts/models"

    def initiate_model_trainer(self):

        print("Reading transformed dataset...")

        df = pd.read_csv(self.input_path)

        # Features and Target
        X = df.drop(columns=["CustomerID", "high_value_repeat"])
        y = df["high_value_repeat"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Feature Scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        os.makedirs(self.model_dir, exist_ok=True)

        # MLflow Experiment
        mlflow.set_experiment("Online Retail High Value Customer Prediction")

        # =====================================================
        # Logistic Regression
        # =====================================================

        print("Training Logistic Regression...")

        log_model = LogisticRegression(random_state=42)

        with mlflow.start_run(run_name="Logistic Regression"):

            log_model.fit(X_train_scaled, y_train)

            log_pred = log_model.predict(X_test_scaled)

            log_f1 = f1_score(y_test, log_pred)
            log_roc = roc_auc_score(y_test, log_pred)

            mlflow.log_param("model", "Logistic Regression")
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("F1 Score", log_f1)
            mlflow.log_metric("ROC AUC", log_roc)

        # =====================================================
        # Random Forest
        # =====================================================

        print("Training Random Forest...")

        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        with mlflow.start_run(run_name="Random Forest"):

            rf_model.fit(X_train, y_train)

            rf_pred = rf_model.predict(X_test)

            rf_f1 = f1_score(y_test, rf_pred)
            rf_roc = roc_auc_score(y_test, rf_pred)

            mlflow.log_param("model", "Random Forest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("F1 Score", rf_f1)
            mlflow.log_metric("ROC AUC", rf_roc)

        # =====================================================
        # Save Models
        # =====================================================

        joblib.dump(
            log_model,
            os.path.join(self.model_dir, "logistic_model.pkl")
        )

        joblib.dump(
            rf_model,
            os.path.join(self.model_dir, "random_forest_model.pkl")
        )

        joblib.dump(
            scaler,
            os.path.join(self.model_dir, "scaler.pkl")
        )

        print("Model Training Completed")

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_scaled,
            X_test_scaled
        )