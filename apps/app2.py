# 🚦 Accident Severity Prediction App (Final Fixed Version)
import streamlit as st
import pandas as pd
import pickle
import json
import os

# ----------------------------------------------------
# 1️⃣ Page Configuration
st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Accident Severity Prediction")
st.write("Predict whether an accident will be **Low** or **High** severity based on environmental and road conditions.")

# ----------------------------------------------------
# 2️⃣ Load Model, Scaler, and Column Info
#    (Using the *exact* filenames from your notebook)
MODEL_FILE = "../models/accidents.pkl"
SCALER_FILE = "../models/scaler.pkl"
COLS_FILE_JSON = "../models/model_columns.json"

model = scaler = model_columns = None

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(COLS_FILE_JSON, "r") as f:
        model_columns = json.load(f)
        
    st.success("✅ Model, scaler, and column order loaded successfully!")
    st.info(f"Loaded Model Type: **{type(model).__name__}**") # Show what model was loaded
    
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.error("Please make sure 'accidents.pkl', 'scaler.pkl', and 'model_columns.json' are in the same folder as this app.")

# ----------------------------------------------------
# 3️⃣ Sample Test Cases
sample_cases = {
    # --- NEW HIGH SEVERITY CASE 1 ---
    "🚨 High-Speed Highway (Fog) 🚨": {
        "Start_Lat": 40.7, "Start_Lng": -74.0, "Temperature(F)": 45.0, "Humidity(%)": 95.0, "Pressure(in)": 29.8, "Visibility(mi)": 0.5,
        "Wind_Speed(mph)": 15.0, "Distance(mi)": 2.5,  # <-- CRITICAL: Long distance
        "Crossing": "No", "Junction": "No", "Amenity": "No", "Stop": "No", "Traffic_Signal": "No", # Highway features
        "Sunrise_Sunset": "Night", "Civil_Twilight": "Night", "Nautical_Twilight": "Night", "Astronomical_Twilight": "Night",
        "Month": 1, "Hour_of_day": 5, "Day_of_week": "Saturday", "Is_Complex_Road": "Yes",
        "Bump": "No", "Give_Way": "No", "No_Exit": "No", "Railway": "No", "Roundabout": "No", "Station": "No", "Traffic_Calming": "No", "Turning_Loop": "No"
    },
   
    
    # --- Original Cases (Likely Low Severity) ---
    "☀️ Clear Day - Highway": {
        "Start_Lat": 37.5, "Start_Lng": -122.4, "Temperature(F)": 75.0, "Humidity(%)": 45.0, "Pressure(in)": 30.0, "Visibility(mi)": 10.0,
        "Wind_Speed(mph)": 5.0, "Distance(mi)": 0.8, 
        "Crossing": "No", "Junction": "No", "Amenity": "No", "Stop": "No", "Traffic_Signal": "No",
        "Sunrise_Sunset": "Day", "Civil_Twilight": "Day", "Nautical_Twilight": "Day", "Astronomical_Twilight": "Day",
        "Month": 6, "Hour_of_day": 14, "Day_of_week": "Tuesday", "Is_Complex_Road": "No",
        "Bump": "No", "Give_Way": "No", "No_Exit": "No", "Railway": "No", "Roundabout": "No", "Station": "No", "Traffic_Calming": "No", "Turning_Loop": "No"
    },
    "🌧 Rainy Night - Urban": {
        "Start_Lat": 40.0, "Start_Lng": -75.1, "Temperature(F)": 60.0, "Humidity(%)": 90.0, "Pressure(in)": 29.5, "Visibility(mi)": 2.0,
        "Wind_Speed(mph)": 10.0, "Distance(mi)": 0.3, # <-- This is why it's Low Severity
        "Crossing": "Yes", "Junction": "Yes", "Amenity": "Yes", "Stop": "No", "Traffic_Signal": "Yes",
        "Sunrise_Sunset": "Night", "Civil_Twilight": "Night", "Nautical_Twilight": "Night", "Astronomical_Twilight": "Night",
        "Month": 9, "Hour_of_day": 20, "Day_of_week": "Monday", "Is_Complex_Road": "Yes",
        "Bump": "No", "Give_Way": "No", "No_Exit": "No", "Railway": "No", "Roundabout": "No", "Station": "Yes", "Traffic_Calming": "No", "Turning_Loop": "No"
    },
}

# ----------------------------------------------------
# 4️⃣ Sidebar – Load Test Case
st.sidebar.header("🧪 Load a Test Case")
selected_case = st.sidebar.selectbox("Choose a test case:", ["Custom Input"] + list(sample_cases.keys()))
prefill = sample_cases.get(selected_case, {})

# Helper function to get prefill index
def get_index(options, key):
    val = prefill.get(key)
    return options.index(val) if val in options else 0

# ----------------------------------------------------
# 5️⃣ Input Form (Now includes ALL features)
with st.form("input_form"):
    st.subheader("Weather & Location")
    col1, col2, col3 = st.columns(3)

    with col1:
        start_lat = st.number_input("Start Latitude", value=float(prefill.get("Start_Lat", 37.5)))
        start_lng = st.number_input("Start Longitude", value=float(prefill.get("Start_Lng", -122.4)))
        distance = st.number_input("Distance (mi)", 0.0, 10.0, float(prefill.get("Distance(mi)", 0.5)))
        
    with col2:
        temperature = st.number_input("Temperature (°F)", 0.0, 120.0, float(prefill.get("Temperature(F)", 70.0)))
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, float(prefill.get("Humidity(%)", 60.0)))
        pressure = st.number_input("Pressure (in)", 25.0, 35.0, float(prefill.get("Pressure(in)", 30.0)))

    with col3:
        visibility = st.number_input("Visibility (mi)", 0.0, 20.0, float(prefill.get("Visibility(mi)", 10.0)))
        wind_speed = st.number_input("Wind Speed (mph)", 0.0, 100.0, float(prefill.get("Wind_Speed(mph)", 5.0)))
        
    st.markdown("---")
    st.subheader("Road Features (Points of Interest)")
    
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        crossing = st.selectbox("Crossing?", ["No", "Yes"], index=get_index(["No", "Yes"], "Crossing"))
        junction = st.selectbox("Junction?", ["No", "Yes"], index=get_index(["No", "Yes"], "Junction"))
        amenity = st.selectbox("Amenity?", ["No", "Yes"], index=get_index(["No", "Yes"], "Amenity"))
        bump = st.selectbox("Bump?", ["No", "Yes"], index=get_index(["No", "Yes"], "Bump"))

    with col5:
        give_way = st.selectbox("Give Way?", ["No", "Yes"], index=get_index(["No", "Yes"], "Give_Way"))
        no_exit = st.selectbox("No Exit?", ["No", "Yes"], index=get_index(["No", "Yes"], "No_Exit"))
        railway = st.selectbox("Railway?", ["No", "Yes"], index=get_index(["No", "Yes"], "Railway"))
        roundabout = st.selectbox("Roundabout?", ["No", "Yes"], index=get_index(["No", "Yes"], "Roundabout"))

    with col6:
        station = st.selectbox("Station?", ["No", "Yes"], index=get_index(["No", "Yes"], "Station"))
        stop = st.selectbox("Stop Sign?", ["No", "Yes"], index=get_index(["No", "Yes"], "Stop"))
        traffic_calming = st.selectbox("Traffic Calming?", ["No", "Yes"], index=get_index(["No", "Yes"], "Traffic_Calming"))
        traffic_signal = st.selectbox("Traffic Signal?", ["No", "Yes"], index=get_index(["No", "Yes"], "Traffic_Signal"))
        
    with col7:
        turning_loop = st.selectbox("Turning Loop?", ["No", "Yes"], index=get_index(["No", "Yes"], "Turning_Loop"))
        is_complex_road = st.selectbox("Complex Road?", ["No", "Yes"], index=get_index(["No", "Yes"], "Is_Complex_Road"))

    st.markdown("---")
    st.subheader("Time & Date")
    
    col8, col9, col10 = st.columns(3)
    
    with col8:
        day_options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        day_of_week_str = st.selectbox("Day of Week", day_options, index=get_index(day_options, "Day_of_week"))
        day_of_week = day_options.index(day_of_week_str) # Convert string to number (0-6)
        
        is_weekend = 1 if day_of_week_str in ["Saturday", "Sunday"] else 0
        month = st.slider("Month", 1, 12, int(prefill.get("Month", 6)))
        hour_of_day = st.slider("Hour of Day", 0, 23, int(prefill.get("Hour_of_day", 12)))

    with col9:
        sunrise_sunset = st.selectbox("Sunrise/Sunset", ["Day", "Night"], index=get_index(["Day", "Night"], "Sunrise_Sunset"))
        civil_twilight = st.selectbox("Civil Twilight", ["Day", "Night"], index=get_index(["Day", "Night"], "Civil_Twilight"))
        
    with col10:
        nautical_twilight = st.selectbox("Nautical Twilight", ["Day", "Night"], index=get_index(["Day", "Night"], "Nautical_Twilight"))
        astronomical_twilight = st.selectbox("Astronomical Twilight", ["Day", "Night"], index=get_index(["Day", "Night"], "Astronomical_Twilight"))


    submitted = st.form_submit_button("🚀 Predict Severity")

# ----------------------------------------------------
# 6️⃣ Prediction Logic (WITH THE FINAL FIX)
# ----------------------------------------------------
if submitted and model is not None:
    
    # --- 1. Create the full input dictionary ---
    input_dict = {
        "Start_Lat": start_lat,
        "Start_Lng": start_lng,
        "Distance(mi)": distance,
        "Temperature(F)": temperature,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Amenity": 1 if amenity == "Yes" else 0,
        "Bump": 1 if bump == "Yes" else 0,
        "Crossing": 1 if crossing == "Yes" else 0,
        "Give_Way": 1 if give_way == "Yes" else 0,
        "Junction": 1 if junction == "Yes" else 0,
        "No_Exit": 1 if no_exit == "Yes" else 0,
        "Railway": 1 if railway == "Yes" else 0,
        "Roundabout": 1 if roundabout == "Yes" else 0,
        "Station": 1 if station == "Yes" else 0,
        "Stop": 1 if stop == "Yes" else 0,
        "Traffic_Calming": 1 if traffic_calming == "Yes" else 0,
        "Traffic_Signal": 1 if traffic_signal == "Yes" else 0,
        "Turning_Loop": 1 if turning_loop == "Yes" else 0,
        "Sunrise_Sunset": 1 if sunrise_sunset == "Day" else 0,
        "Civil_Twilight": 1 if civil_twilight == "Day" else 0,
        "Nautical_Twilight": 1 if nautical_twilight == "Day" else 0,
        "Astronomical_Twilight": 1 if astronomical_twilight == "Day" else 0,
        "Hour_of_day": hour_of_day,
        "Day_of_week": day_of_week,
        "Month": month,
        "Is_Weekend": is_weekend,
        "Is_Complex_Road": 1 if is_complex_road == "Yes" else 0,
        "Duration": 0.5 # Default value
    }

    # --- 2. Create and reindex the DataFrames ---
    
    # A) Create the UNSCALED DataFrame, ordered correctly
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # B) Create the SCALED version (for Logistic Regression)
    scaler_cols = getattr(scaler, "feature_names_in_", None)
    input_scaled_df = input_df.copy() # Start with the unscaled, ordered data

    if scaler_cols is not None:
        try:
            # Scale *only* the columns the scaler was fit on
            input_scaled_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
        except Exception as e:
            st.error(f"❌ Error during scaling transform: {e}")
            st.stop()
    else:
        st.error("❌ Scaler has no `feature_names_in_` attribute. Was it saved correctly?")
        st.stop()
    
    # --- 3. Predict (Using the CORRECT DataFrame) ---
    try:
        # Check the class name of the loaded model
        model_type = type(model).__name__
        
        prediction_df = None # This will hold the data we send to the model

        if model_type == "LogisticRegression":
            st.info("ℹ️ Model is LogisticRegression. Using SCALED data for prediction.")
            prediction_df = input_scaled_df # Use the scaled data
        
        elif model_type in ["XGBClassifier", "RandomForestClassifier"]:
            st.info(f"ℹ️ Model is {model_type}. Using UNSCALED data (as per training).")
            prediction_df = input_df # Use the original, unscaled data
        
        else:
            st.warning(f"⚠️ Unknown model type: {model_type}. Defaulting to scaled data.")
            prediction_df = input_scaled_df # Default fallback

        # Now, make the prediction using the correctly chosen dataframe
        pred = model.predict(prediction_df)[0]
        prob_all = model.predict_proba(prediction_df)[0]
        prob_low = prob_all[0]
        prob_high = prob_all[1]

        # --- 4. Display Result ---
        st.subheader("🎯 Prediction Result")
        
        if pred == 1:
            label = "🔴 High Severity"
            st.markdown(f"### {label}")
            st.progress(float(prob_high))
            st.caption(f"Model confidence in **High Severity**: **{prob_high*100:.2f}%**")
        else:
            label = "🟢 Low Severity"
            st.markdown(f"### {label}")
            st.progress(float(prob_low))
            st.caption(f"Model confidence in **Low Severity**: **{prob_low*100:.2f}%**")

        with st.expander("Show Detailed Probabilities & Input Data"):
            st.write(f"Probability of Low Severity (0): **{prob_low*100:.2f}%**")
            st.write(f"Probability of High Severity (1): **{prob_high*100:.2f}%**")
            st.write("🔧 Final input data sent to model (matches training format):")
            st.dataframe(prediction_df) # Show the actual data used

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug: Final DataFrame columns:", prediction_df.columns)
        st.write("Debug: Expected model columns:", model_columns)


elif submitted:
    st.error("⚠️ Model or scaler not loaded correctly. Check your files.")

# ----------------------------------------------------
# 7️⃣ Footer
st.markdown("---")
st.caption("Accident Severity Prediction Model")