# 🚫 Suppress warnings globally
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# 🧪 Generate 600 normal cases
normal = pd.DataFrame({
    'hinge_stress': np.random.normal(50, 5, 600),
    'temperature': np.random.normal(30, 5, 600),
    'vibration': np.random.normal(0.5, 0.05, 600),
    'failure': 0
})

# 🔥 Generate 600 failure cases
failure = pd.DataFrame({
    'hinge_stress': np.random.normal(70, 3, 600),
    'temperature': np.random.normal(45, 3, 600),
    'vibration': np.random.normal(0.7, 0.05, 600),
    'failure': 1
})

# 🧩 Combine and shuffle
data = pd.concat([normal, failure]).sample(frac=1, random_state=42).reset_index(drop=True)

# 💾 Save simulated data
os.makedirs('data', exist_ok=True)
data.to_csv('data/simulated_sensor_data.csv', index=False)

# 🧠 Prepare features and labels
X = data[['hinge_stress', 'temperature', 'vibration']].values
y = data['failure'].values  # Already 1D

# 🏋️‍♂️ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧪 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/glovebox_model.pkl')
print("✅ Model trained and saved. Data saved to data/simulated_sensor_data.csv")