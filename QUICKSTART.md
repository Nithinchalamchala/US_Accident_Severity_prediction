# Quick Start Guide

Get up and running with the Accident Severity Prediction application in minutes!

## 🚀 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Nithinchalamchala/US_Accident_Severity_prediction.git
cd US_Accident_Severity_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run apps/app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📱 Using the Application

### Tab 1: Predict Severity
1. Select a pre-filled test case from the sidebar (e.g., "High-Speed Highway (Fog)")
2. Or enter custom values for weather, road, and time conditions
3. Click "Predict Severity" to see the results
4. View confidence scores and probability distributions

### Tab 2: Accident Dashboard
1. Upload a CSV file with accident data (use files from `data/` folder for testing)
2. Apply filters (state, month, severity)
3. Explore interactive visualizations:
   - Severity distributions
   - Time-based trends
   - Geographic heatmaps
   - Weather correlations

### Tab 3: Safety Tips
Browse safety guidelines organized by:
- Before Driving checklist
- While Driving best practices
- Emergency response procedures
- Data-driven safety insights

## 🧪 Testing with Sample Data

Try the dashboard with our sample datasets:
```bash
# Use the 50k sample for comprehensive testing
data/accident_50k.csv

# Or smaller samples for quick tests
data/accident_sample_1000.csv
data/accident_sample_100.csv
```

## 📓 Exploring the Model

Open the Jupyter notebook to see the model training process:
```bash
jupyter notebook notebooks/Severity_Predictor.ipynb
```

## 🔧 Troubleshooting

### Issue: Module not found
**Solution**: Make sure you've installed all requirements
```bash
pip install -r requirements.txt
```

### Issue: Model files not loading
**Solution**: Ensure you're running the app from the project root directory
```bash
# Run from project root, not from apps/ directory
streamlit run apps/app.py
```

### Issue: CSV upload fails
**Solution**: Ensure your CSV has the required columns (see data/README.md for format)

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the [models/README.md](models/README.md) to understand the ML model
- Download the full dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- Explore the notebook for model training and evaluation

## 💡 Tips

- Use the test cases in the sidebar to quickly see different prediction scenarios
- The "High-Speed Highway (Fog)" case typically predicts high severity
- The "Clear Day - Highway" case typically predicts low severity
- Upload different sample sizes to see how visualizations scale

---

**Need Help?** Open an issue on [GitHub](https://github.com/Nithinchalamchala/US_Accident_Severity_prediction/issues)
