import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load('models/glovebox_model.pkl')

# Define ranges
stress_range = np.linspace(40, 80, 100)
temp_range = np.linspace(20, 60, 100)
vibration_fixed = 0.7

# Create grid
Z = np.zeros((len(temp_range), len(stress_range)))
for i, temp in enumerate(temp_range):
    for j, stress in enumerate(stress_range):
        features = np.array([[stress, temp, vibration_fixed]])
        proba = model.predict_proba(features)[0]
        Z[i, j] = proba[1] if len(proba) > 1 else 0.0

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.contourf(stress_range, temp_range, Z, levels=20, cmap='coolwarm')
plt.colorbar(label='Failure Risk Score')
plt.xlabel('Hinge Stress (N·m)')
plt.ylabel('Temperature (°C)')
plt.title('Glove Box Failure Risk Heatmap (Vibration = 0.7g)')
plt.tight_layout()
plt.show()