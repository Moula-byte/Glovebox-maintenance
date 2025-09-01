

## 🧤 Glove Box Predictive Maintenance

This project predicts mechanical failure in vehicle glove boxes using simulated sensor data and machine learning. It includes a full-stack pipeline with data generation, model training, a Flask web app, and visual diagnostics.

---

### 🚀 Features

- Simulated sensor data: hinge stress, temperature, vibration  
- Random Forest classifier trained on balanced failure cases  
- Flask dashboard with real-time prediction and risk scoring  
- Logging of predictions for analysis  
- Heatmap visualization of failure risk across temperature zones  
- Responsive HTML UI with watermark styling and iconography

---

### 🛠️ Tech Stack

| Layer        | Tools Used                          |
|--------------|-------------------------------------|
| Data         | NumPy, Pandas                       |
| ML Model     | scikit-learn (RandomForest)         |
| Web App      | Flask, HTML/CSS                     |
| Visualization| Matplotlib                          |
| Deployment   | Local / Heroku / Render-ready       |

---

### 📂 Project Structure

```
glovebox-maintenance/
├── data/                  # Simulated sensor data
├── models/                # Trained ML model
├── logs/                  # Prediction logs
├── static/                # UI assets (background, icons)
├── templates/             # HTML dashboard
├── run_pipeline.py        # Data generation + model training
├── app.py                 # Flask app
├── visualize_risk.py      # Heatmap + decision boundary plots
├── requirements.txt       # Dependencies
```

---

### 📈 How It Works

1. `run_pipeline.py` generates 600 normal and 600 failure cases  
2. Model is trained to detect failure based on:
   - Hinge Stress > 65 N·m  
   - Vibration > 0.6 g  
   - Temperature > 40°C  
3. `app.py` serves predictions via a clean dashboard  
4. `visualize_risk.py` plots heatmaps showing risk across temperature zones

---

### 🧪 Sample Inputs

| Scenario           | Hinge Stress | Temperature | Vibration | Prediction |
|--------------------|--------------|-------------|-----------|------------|
| Normal use         | 50           | 25          | 0.4       | 0          |
| High temp fatigue  | 68           | 45          | 0.65      | 1          |
| Extreme stress     | 75           | 55          | 0.75      | 1          |

---

### 📊 Visualization

Run `visualize_risk.py` to generate:
- Heatmap of failure risk across hinge stress and temperature
- Decision boundary plot showing safe vs critical zones

---

### 🧑‍🏫 Mentor Challenge

Extend this project by:
- Adding SHAP interpretability  
- Deploying to Heroku or Render  
- Logging user sessions and feedback  
- Expanding to other vehicle components (e.g. seat sliders, HVAC knobs)

---

### 📦 Installation

```bash
pip install -r requirements.txt
python run_pipeline.py
python app.py
```

Then visit `http://localhost:5000` to use the dashboard.

