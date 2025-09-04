from flask import Flask,request,jsonify,render_template
import joblib
from flask_cors import CORS
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)


try:
    model = joblib.load('best_model_stellar.joblib')
    features = joblib.load('feature_list.joblib')

    class_mapping = {
        0 : 'Galaxy',
        1 : 'QSO',
        2 : 'STAR'
    }
except Exception as e:
    print(f'Error occured while loading model / features')
    model = None
    features = None
    class_mapping = {}


@app.route('/')
def index():
    return render_template('index.html')

# Prediction Route

@app.route('/predict',methods=['POST'])
def predict():
    if model is None or features is None:
        return jsonify({'Error : Error while loading model or features, they are None'})
    
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error' : 'No data provided'}) , 400
    
    required_features = set(features)
    recieved_features = set(data.keys())
    
    if not required_features.issubset(recieved_features):
        missing_features = list(required_features - recieved_features)
        return jsonify({'error' : 'Missing features in request', "Missing" : missing_features}) , 400
    
    try:
        input_data = pd.DataFrame([data])

        input_data = input_data[features]

        prediction_encoded = model.predict(input_data)[0]
        prediction_class = class_mapping.get(prediction_encoded,'Unknown')

        probabilities = model.predict_proba(input_data)[0]

        prob_dict = {
            class_mapping[i] : float(probabilities[i]) for i in class_mapping
        }
        
        return jsonify({
            'prediction' : prediction_class,
            'probabilities' : prob_dict
        })
    
    except Exception as e:
        return jsonify({'Error' : str(e)}) , 500
    


if __name__ == "__main__":
    app.run(debug=True)

