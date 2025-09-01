import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = joblib.load('models/glovebox_model.pkl')
os.makedirs('logs', exist_ok=True)

def log_prediction(features, prediction, risk_score):
    log_entry = {
        'hinge_stress': features[0][0],
        'temperature': features[0][1],
        'vibration': features[0][2],
        'prediction': prediction,
        'risk_score': risk_score
    }
    log_path = 'logs/prediction_log.csv'
    df = pd.DataFrame([log_entry])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hinge_stress = float(request.form['hinge_stress'])
        temperature = float(request.form['temperature'])
        vibration = float(request.form['vibration'])

        features = np.array([[hinge_stress, temperature, vibration]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        risk_score = proba[1] if len(proba) > 1 else 0.0

        log_prediction(features, prediction, risk_score)

        return render_template('index.html',
                               prediction=int(prediction),
                               risk_score=round(risk_score, 2))
    except Exception as e:
        return render_template('index.html',
                               prediction="Error",
                               risk_score=str(e))

if __name__ == '__main__':
    app.run(debug=True)