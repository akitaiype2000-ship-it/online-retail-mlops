import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelEvaluation:

    def __init__(self):

        self.input_path = "artifacts/customer_data.csv"

        self.log_model = "artifacts/models/logistic_model.pkl"

        self.rf_model = "artifacts/models/random_forest_model.pkl"

        self.scaler = "artifacts/models/scaler.pkl"

    def initiate_model_evaluation(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    ):

        log_model = joblib.load(self.log_model)

        rf_model = joblib.load(self.rf_model)

        scaler = joblib.load(self.scaler)

        # Cross Validation

        log_cv = cross_val_score(

            LogisticRegression(random_state=42),

            X_train_scaled,

            y_train,

            cv=5,

            scoring="f1"

        )

        rf_cv = cross_val_score(

            RandomForestClassifier(

                n_estimators=100,

                random_state=42

            ),

            X_train,

            y_train,

            cv=5,

            scoring="f1"

        )

        print("Logistic CV F1:", log_cv.mean())

        print("Random Forest CV F1:", rf_cv.mean())

        # Predictions

        log_pred = log_model.predict(X_test_scaled)

        rf_pred = rf_model.predict(X_test)

        print("\n========== Logistic Regression ==========")

        print(confusion_matrix(y_test, log_pred))

        print(classification_report(y_test, log_pred))

        print("ROC AUC:", roc_auc_score(y_test, log_pred))

        print("\n========== Random Forest ==========")

        print(confusion_matrix(y_test, rf_pred))

        print(classification_report(y_test, rf_pred))

        print("ROC AUC:", roc_auc_score(y_test, rf_pred))

        # F1

        log_f1 = f1_score(y_test, log_pred)

        rf_f1 = f1_score(y_test, rf_pred)

        print("\nLogistic Test F1:", log_f1)

        print("Random Forest Test F1:", rf_f1)

        # Misclassified

        misclassified = X_test.copy()

        misclassified["actual"] = y_test.values

        misclassified["predicted"] = rf_pred

        errors = misclassified[
            misclassified["actual"] != misclassified["predicted"]
        ]

        print("Misclassified Customers:", len(errors))

        # Comparison

        comparison = {

            "Model": [

                "Logistic Regression",

                "Random Forest"

            ],

            "CV F1": [

                log_cv.mean(),

                rf_cv.mean()

            ],

            "Test F1": [

                log_f1,

                rf_f1

            ],

            "ROC AUC": [

                roc_auc_score(y_test, log_pred),

                roc_auc_score(y_test, rf_pred)

            ]

        }

        comparison_df = pd.DataFrame(comparison)

        print("\nModel Comparison")

        print(comparison_df)

        best_model = comparison_df.loc[
            comparison_df["Test F1"].idxmax()
        ]

        print("\nBest Model")

        print(best_model)

        return comparison_df