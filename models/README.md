# Models Directory

This directory contains the trained machine learning models and preprocessing artifacts.

## Files

### Primary Model Files
- `accidents.pkl` - Main trained XGBoost classifier model
- `scaler.pkl` - StandardScaler for feature normalization
- `model_columns.json` - Feature column names and order (JSON format)
- `model_columns.pkl` - Feature column names and order (pickle format)

### Alternative Model Files
- `severity_model.pkl` - Alternative trained model
- `severity_scaler.pkl` - Alternative scaler

## Model Details

### Algorithm
The primary model uses **XGBoost (Extreme Gradient Boosting)** classifier, which provides:
- High accuracy for classification tasks
- Robust handling of imbalanced datasets
- Feature importance analysis
- Fast prediction times

### Features Used
The model uses the following feature categories:
- **Weather conditions**: Temperature, humidity, pressure, visibility, wind speed
- **Road features**: Crossings, junctions, traffic signals, stop signs, etc.
- **Temporal features**: Hour of day, day of week, month, weekend indicator
- **Location data**: Latitude, longitude, distance
- **Derived features**: Complex road indicator, twilight conditions

### Target Variable
- **Binary Classification**: Low Severity (0) vs High Severity (1)
- Original severity levels (1-4) are converted to binary for better model performance

## Usage

These model files are automatically loaded by the Streamlit application. To use them in your own code:

```python
import pickle
import json

# Load the model
with open('models/accidents.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load column order
with open('models/model_columns.json', 'r') as f:
    model_columns = json.load(f)
```

## Retraining

To retrain the model with new data, refer to the Jupyter notebook in the `notebooks/` directory.
