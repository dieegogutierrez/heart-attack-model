import pickle
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify

model_file_path = '../models/model.bin'

with open(model_file_path, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('heart_attack')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    df = pd.DataFrame.from_dict([patient])

    X = df.values
    y_pred = model.predict_proba(X)[0, 1]
    heart_attack = y_pred >= 0.5

    result = {
        'heart_attack_probability': float(round(y_pred, 3)),
        'heart_attack': bool(heart_attack)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)