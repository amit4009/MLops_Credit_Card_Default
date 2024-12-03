import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def hyperparameter_tuning(self, x_train, y_train, x_test, y_test):
        """
        Performs hyperparameter tuning for multiple models and returns the best model with its metrics.
        """
        # Define parameter grids
        param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        }

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }

        best_model = None
        best_score = 0
        best_params = {}
        best_metrics = {}

        # Loop through each model
        for model_name, model in models.items():
            try:
                logging.info(f"Tuning {model_name}...")
                param_grid = param_grids[model_name]

                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="accuracy",
                    cv=3,  # 3-fold cross-validation
                    verbose=2,
                    n_jobs=-1  # Parallel processing
                )

                grid_search.fit(x_train, y_train)

                # Get the best model and parameters
                tuned_model = grid_search.best_estimator_
                tuned_params = grid_search.best_params_
                accuracy = grid_search.best_score_

                logging.info(f"Best parameters for {model_name}: {tuned_params}")
                logging.info(f"Best cross-validated accuracy for {model_name}: {accuracy:.4f}")

                # Evaluate on the test set
                y_test_pred = tuned_model.predict(x_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, average='weighted')
                recall = recall_score(y_test, y_test_pred, average='weighted')
                f1 = f1_score(y_test, y_test_pred, average='weighted')

                logging.info(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")

                # Compare and store the best model
                if test_accuracy > best_score:
                    best_model = tuned_model
                    best_score = test_accuracy
                    best_params = tuned_params
                    best_metrics = {
                        "Accuracy": float(test_accuracy),
                        "Precision": float(precision),
                        "Recall": float(recall),
                        "F1 Score": float(f1)
                    }

            except Exception as e:
                logging.error(f"Error while tuning {model_name}: {str(e)}")

        # Return the best model details
        return {
            "Best Model": best_model,
            "Best Score": float(best_score),
            "Best Parameters": best_params,
            "Best Metrics": best_metrics
        }

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main function to train the model with hyperparameter tuning and return the best model's performance.
        """
        try:
            logging.info("Splitting train and test input data.")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Perform hyperparameter tuning
            logging.info("Starting hyperparameter tuning...")
            results = self.hyperparameter_tuning(x_train, y_train, x_test, y_test)

            # Save the best model
            best_model = results["Best Model"]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            # Return best model's metrics
            return results

        except Exception as e:
            raise CustomException(e, sys)
