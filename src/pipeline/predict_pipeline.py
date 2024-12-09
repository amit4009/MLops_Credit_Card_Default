import sys
import pandas as pd
import pickle
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Predicts the default status for the provided features.

        Args:
            features (pd.DataFrame): Input features as a DataFrame.

        Returns:
            np.ndarray: Predicted default status (0 or 1).
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # Load the saved model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        limit_bal: float,
        sex: int,
        education: int,
        marriage: int,
        pay_0: int,
        pay_2: int,
        pay_3: int,
        pay_4: int,
        pay_5: int,
        pay_6: int,
        bill_amt1: float,
        bill_amt2: float,
        bill_amt3: float,
        bill_amt4: float,
        bill_amt5: float,
        bill_amt6: float,
        pay_amt1: float,
        pay_amt2: float,
        pay_amt3: float,
        pay_amt4: float,
        pay_amt5: float,
        pay_amt6: float
    ):
        self.limit_bal = limit_bal
        self.sex = sex
        self.education = education
        self.marriage = marriage
        self.pay_0 = pay_0
        self.pay_2 = pay_2
        self.pay_3 = pay_3
        self.pay_4 = pay_4
        self.pay_5 = pay_5
        self.pay_6 = pay_6
        self.bill_amt1 = bill_amt1
        self.bill_amt2 = bill_amt2
        self.bill_amt3 = bill_amt3
        self.bill_amt4 = bill_amt4
        self.bill_amt5 = bill_amt5
        self.bill_amt6 = bill_amt6
        self.pay_amt1 = pay_amt1
        self.pay_amt2 = pay_amt2
        self.pay_amt3 = pay_amt3
        self.pay_amt4 = pay_amt4
        self.pay_amt5 = pay_amt5
        self.pay_amt6 = pay_amt6

    def get_data_as_data_frame(self):
        """
        Converts the input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the input features.
        """
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.limit_bal],
                "SEX": [self.sex],
                "EDUCATION": [self.education],
                "MARRIAGE": [self.marriage],
                "PAY_0": [self.pay_0],
                "PAY_2": [self.pay_2],
                "PAY_3": [self.pay_3],
                "PAY_4": [self.pay_4],
                "PAY_5": [self.pay_5],
                "PAY_6": [self.pay_6],
                "BILL_AMT1": [self.bill_amt1],
                "BILL_AMT2": [self.bill_amt2],
                "BILL_AMT3": [self.bill_amt3],
                "BILL_AMT4": [self.bill_amt4],
                "BILL_AMT5": [self.bill_amt5],
                "BILL_AMT6": [self.bill_amt6],
                "PAY_AMT1": [self.pay_amt1],
                "PAY_AMT2": [self.pay_amt2],
                "PAY_AMT3": [self.pay_amt3],
                "PAY_AMT4": [self.pay_amt4],
                "PAY_AMT5": [self.pay_amt5],
                "PAY_AMT6": [self.pay_amt6],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
