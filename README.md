# 🚗 Accident Severity Prediction using Machine Learning

A comprehensive machine learning project that predicts the severity of traffic accidents based on environmental, road, and weather conditions. This project includes both a trained ML model and an interactive Streamlit web application for real-time predictions and data visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project uses machine learning to predict accident severity levels (Low/High) based on various factors including:
- Weather conditions (temperature, humidity, visibility, wind speed)
- Road features (crossings, junctions, traffic signals)
- Time factors (hour of day, day of week, month)
- Location data (latitude, longitude)

The application provides three main functionalities:
1. **Severity Prediction**: Real-time accident severity prediction
2. **Data Dashboard**: Comprehensive visualization and analysis of accident data
3. **Safety Tips**: Educational content on road safety and accident prevention

## 📊 Dataset

The dataset used in this project is the **US Accidents Dataset** available on Kaggle:

**Dataset Link**: [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

### Dataset Description

This is a countrywide car accident dataset covering 49 states of the USA. The accident data are collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident data. The dataset contains approximately 7.7 million accident records.

### Key Features in Dataset:
- **Location**: Latitude, Longitude, City, State, County
- **Time**: Start/End time, timezone information
- **Weather**: Temperature, humidity, pressure, visibility, wind speed, precipitation
- **Road Features**: Crossing, junction, traffic signals, stop signs, etc.
- **Severity**: Accident severity level (1-4)

### Citation

If you use this dataset, please cite:
```
Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. 
"A Countrywide Traffic Accident Dataset.", 2019.
```

## ✨ Features

### 1. Accident Severity Predictor
- Interactive form with pre-filled test cases
- Real-time severity prediction (Low/High)
- Confidence scores and probability distribution
- Support for multiple ML models (XGBoost, Random Forest, Logistic Regression)

### 2. Comprehensive Dashboard
- **KPI Metrics**: Total records, severity distribution, state coverage
- **Interactive Filters**: Filter by state, month, and severity
- **Visualizations**:
  - Severity distribution charts
  - Time-series accident trends
  - Hourly and daily patterns
  - Geographic heatmaps
  - Weather condition analysis
  - Correlation matrices
  - State-wise accident statistics

### 3. Safety Tips & Insights
- Pre-driving checklist
- Safe driving practices
- Emergency response guidelines
- Data-driven safety insights

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd accident-severity-prediction
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Required Python packages**:
```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
matplotlib
seaborn
pickle
```

## 💻 Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Jupyter Notebook

To explore the model training process:

```bash
jupyter notebook Severity_Predictor.ipynb
```

### Making Predictions

1. Navigate to the "Predict Severity" tab
2. Choose a pre-filled test case or enter custom values
3. Fill in the required fields:
   - Weather conditions (temperature, humidity, visibility)
   - Road features (crossings, junctions, signals)
   - Time information (hour, day, month)
4. Click "Predict Severity" to get results

### Analyzing Data

1. Navigate to the "Accident Dashboard" tab
2. Upload your accident dataset (CSV format)
3. Use filters to explore specific subsets
4. View interactive visualizations and insights

## 🤖 Model Details

### Trained Models
The project includes multiple trained models:
- **XGBoost Classifier** (Primary model)
- **Random Forest Classifier**
- **Logistic Regression**

### Model Files
- `accidents.pkl`: Trained classification model
- `scaler.pkl`: Feature scaler for preprocessing
- `model_columns.json`: Feature column order
- `severity_model.pkl`: Alternative model file
- `severity_scaler.pkl`: Alternative scaler file

### Feature Engineering
The model uses the following feature categories:
- **Numerical Features**: Temperature, humidity, pressure, visibility, wind speed, distance
- **Categorical Features**: Road features (encoded as binary)
- **Temporal Features**: Hour of day, day of week, month, weekend indicator
- **Derived Features**: Complex road indicator, twilight conditions

### Model Performance
The model achieves strong performance in predicting accident severity with:
- High accuracy on test data
- Balanced precision and recall
- Robust handling of imbalanced classes

## 📁 Project Structure

```
accident-severity-prediction/
│
├── app.py                          # Main Streamlit application
├── app2.py                         # Alternative Streamlit app
├── Severity_Predictor.ipynb        # Model training notebook
│
├── accidents.pkl                   # Trained model
├── scaler.pkl                      # Feature scaler
├── model_columns.json              # Feature columns
├── model_columns.pkl               # Feature columns (pickle)
├── severity_model.pkl              # Alternative model
├── severity_scaler.pkl             # Alternative scaler
│
├── accident_50k.csv                # Sample dataset (50k records)
├── accident_sample_*.csv           # Various sample sizes
│
├── Accident-Severity-Prediction-using-Machine-Learning[1].pdf
│                                   # Project documentation
│
└── README.md                       # This file
```

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by [Sobhan Moosavi](https://www.kaggle.com/sobhanmoosavi) on Kaggle
- US Accidents Dataset contributors
- Streamlit community for excellent documentation

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project is for educational and research purposes. Always follow local traffic laws and safety guidelines while driving.
