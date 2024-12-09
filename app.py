from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import logging

# Initialize Flask application
application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Parse the JSON payload
            data = request.json
            logging.info(f"Received data: {data}")

            # Handle list payload if necessary
            if isinstance(data, list):
                data = data[0]

            # Create CustomData object
            custom_data = CustomData(
                limit_bal=float(data['LIMIT_BAL']),
                sex=int(data['SEX']),
                education=int(data['EDUCATION']),
                marriage=int(data['MARRIAGE']),
                pay_0=int(data['PAY_0']),
                pay_2=int(data['PAY_2']),
                pay_3=int(data['PAY_3']),
                pay_4=int(data['PAY_4']),
                pay_5=int(data['PAY_5']),
                pay_6=int(data['PAY_6']),
                bill_amt1=float(data['BILL_AMT1']),
                bill_amt2=float(data['BILL_AMT2']),
                bill_amt3=float(data['BILL_AMT3']),
                bill_amt4=float(data['BILL_AMT4']),
                bill_amt5=float(data['BILL_AMT5']),
                bill_amt6=float(data['BILL_AMT6']),
                pay_amt1=float(data['PAY_AMT1']),
                pay_amt2=float(data['PAY_AMT2']),
                pay_amt3=float(data['PAY_AMT3']),
                pay_amt4=float(data['PAY_AMT4']),
                pay_amt5=float(data['PAY_AMT5']),
                pay_amt6=float(data['PAY_AMT6'])
            )

            # Convert input to DataFrame
            pred_df = custom_data.get_data_as_data_frame()
            logging.info(f"Input DataFrame:\n{pred_df}")

            # Predict using pipeline
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(pred_df)

            # Convert the prediction result to a serializable format
            if isinstance(result, np.ndarray):
                result = result.tolist()  # Convert ndarray to list
                
                
            # Convert result to human-readable message
            message = (
                "The customer is likely to default" if result[0] == 1 
                else "The customer will not default"
            )
            
            # Return the prediction result as a JSON response
            logging.info(f"Prediction message: {message}")
            return jsonify({"message": message})  # Return the first element of the result
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
