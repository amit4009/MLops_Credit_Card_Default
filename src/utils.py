import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve


from sklearn.model_selection import GridSearchCV
# import yaml


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
      
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param_grids):
    """
    Evaluates multiple models with hyperparameter tuning and returns their metrics.
    
    Args:
        x_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        x_test (np.array): Testing feature data.
        y_test (np.array): Testing target data.
        models (dict): Dictionary of model name -> model object.
        param_grids (dict): Dictionary of model name -> hyperparameter grid.

    Returns:
        dict: Dictionary of model name -> evaluation metrics.
    """
    report = {}
    for model_name, model in models.items():
        try:
            print(f"Tuning {model_name}...")
            
            # Perform hyperparameter tuning
            param_grid = param_grids.get(model_name, {})
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="accuracy",
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(x_train, y_train)

            # Best model and predictions
            tuned_model = grid_search.best_estimator_
            y_test_pred = tuned_model.predict(x_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Store metrics
            report[model_name] = {
                "Best Parameters": grid_search.best_params_,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    return report