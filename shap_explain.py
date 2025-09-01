import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and sample data
model = joblib.load('models/glovebox_model.pkl')
data = pd.read_csv('data/simulated_sensor_data.csv')
X = data[['hinge_stress', 'temperature', 'vibration']]

# SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot summary
shap.summary_plot(shap_values, X)