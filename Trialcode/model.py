import os
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test input data.')
            
            # Split train and test arrays
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define candidate models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            }

            # Evaluate models
            logging.info("Evaluating models...")
            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models
            )

            # Find the best model based on Accuracy
            best_model_name = max(
                model_report, key=lambda name: model_report[name]["Accuracy"]
            )
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]["Accuracy"]

            # Check if the best model meets the threshold
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with a score above the threshold (0.6).")

            logging.info(f"Best model found: {best_model_name} with an accuracy of {best_model_score:.4f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate the best model on the test set
            logging.info("Making predictions on the test set with the best model.")
            y_test_pred = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_test_pred)

            logging.info(f"Model Accuracy on Test Set: {accuracy:.4f}")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
