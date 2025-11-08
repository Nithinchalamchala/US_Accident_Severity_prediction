import streamlit as st
import pandas as pd
import pickle
import json
import os
import numpy as np  # Added for the dashboard's heatmap
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# 1️⃣ Page Configuration (Global)
# ----------------------------------------------------
st.set_page_config(page_title="Accident Analytics & Prediction", page_icon="🚗", layout="wide")

st.title("🚗 Accident Analytics & Prediction")

# ----------------------------------------------------
# 2️⃣ Tab Creation
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Predict Severity", "Accident Dashboard", "Safety Tips"])

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BEGIN TAB 1: PREDICTION APP (Your original app)
# -------------------------------------------------------------------
# -------------------------------------------------------------------

with tab1:
    st.header("🔮 Predict Accident Severity")
    st.write("Use this tool to predict the severity of an accident based on road and weather conditions.")
    
    # --- Load Model, Scaler, and Column Info ---
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
            
        st.success("✅ Prediction model, scaler, and columns loaded successfully!")
        st.info(f"Loaded Model Type: **{type(model).__name__}**")
        
    except Exception as e:
        st.error(f"❌ Error loading prediction files: {e}")
        st.error("Please make sure 'accidents.pkl', 'scaler.pkl', and 'model_columns.json' are in the same folder.")

    # --- Sample Test Cases ---
    sample_cases = {
        "🚨 High-Speed Highway (Fog) 🚨": {
            "Start_Lat": 40.7, "Start_Lng": -74.0, "Temperature(F)": 45.0, "Humidity(%)": 95.0, "Pressure(in)": 29.8, "Visibility(mi)": 0.5,
            "Wind_Speed(mph)": 15.0, "Distance(mi)": 2.5, "Crossing": "No", "Junction": "No", "Amenity": "No", "Stop": "No", "Traffic_Signal": "No",
            "Sunrise_Sunset": "Night", "Civil_Twilight": "Night", "Nautical_Twilight": "Night", "Astronomical_Twilight": "Night",
            "Month": 1, "Hour_of_day": 5, "Day_of_week": "Saturday", "Is_Complex_Road": "Yes",
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
        "☀️ Clear Day - Highway": {
            "Start_Lat": 37.5, "Start_Lng": -122.4, "Temperature(F)": 75.0, "Humidity(%)": 45.0, "Pressure(in)": 30.0, "Visibility(mi)": 10.0,
            "Wind_Speed(mph)": 5.0, "Distance(mi)": 0.8, "Crossing": "No", "Junction": "No", "Amenity": "No", "Stop": "No", "Traffic_Signal": "No",
            "Sunrise_Sunset": "Day", "Civil_Twilight": "Day", "Nautical_Twilight": "Day", "Astronomical_Twilight": "Day",
            "Month": 6, "Hour_of_day": 14, "Day_of_week": "Tuesday", "Is_Complex_Road": "No",
            "Bump": "No", "Give_Way": "No", "No_Exit": "No", "Railway": "No", "Roundabout": "No", "Station": "No", "Traffic_Calming": "No", "Turning_Loop": "No"
        },
    }

    st.sidebar.header("🧪 Load a Test Case (for Prediction)")
    selected_case = st.sidebar.selectbox("Choose a test case:", ["Custom Input"] + list(sample_cases.keys()))
    prefill = sample_cases.get(selected_case, {})

    def get_index(options, key):
        val = prefill.get(key)
        return options.index(val) if val in options else 0

    # --- Prediction Input Form ---
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
            day_of_week = day_options.index(day_of_week_str)
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

    # --- Prediction Logic ---
    if submitted and model is not None:
        input_dict = {
            "Start_Lat": start_lat, "Start_Lng": start_lng, "Distance(mi)": distance, "Temperature(F)": temperature,
            "Humidity(%)": humidity, "Pressure(in)": pressure, "Visibility(mi)": visibility, "Wind_Speed(mph)": wind_speed,
            "Amenity": 1 if amenity == "Yes" else 0, "Bump": 1 if bump == "Yes" else 0, "Crossing": 1 if crossing == "Yes" else 0,
            "Give_Way": 1 if give_way == "Yes" else 0, "Junction": 1 if junction == "Yes" else 0, "No_Exit": 1 if no_exit == "Yes" else 0,
            "Railway": 1 if railway == "Yes" else 0, "Roundabout": 1 if roundabout == "Yes" else 0, "Station": 1 if station == "Yes" else 0,
            "Stop": 1 if stop == "Yes" else 0, "Traffic_Calming": 1 if traffic_calming == "Yes" else 0,
            "Traffic_Signal": 1 if traffic_signal == "Yes" else 0, "Turning_Loop": 1 if turning_loop == "Yes" else 0,
            "Sunrise_Sunset": 1 if sunrise_sunset == "Day" else 0, "Civil_Twilight": 1 if civil_twilight == "Day" else 0,
            "Nautical_Twilight": 1 if nautical_twilight == "Day" else 0, "Astronomical_Twilight": 1 if astronomical_twilight == "Day" else 0,
            "Hour_of_day": hour_of_day, "Day_of_week": day_of_week, "Month": month, "Is_Weekend": is_weekend,
            "Is_Complex_Road": 1 if is_complex_road == "Yes" else 0, "Duration": 0.5 # Default value
        }
        
        input_df = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)
        
        scaler_cols = getattr(scaler, "feature_names_in_", None)
        input_scaled_df = input_df.copy()

        if scaler_cols is not None:
            try:
                input_scaled_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
            except Exception as e:
                st.error(f"❌ Error during scaling transform: {e}")
                st.stop()
        else:
            st.error("❌ Scaler has no `feature_names_in_` attribute.")
            st.stop()
        
        try:
            model_type = type(model).__name__
            prediction_df = None

            if model_type == "LogisticRegression":
                st.info("ℹ️ Model is LogisticRegression. Using SCALED data for prediction.")
                prediction_df = input_scaled_df
            elif model_type in ["XGBClassifier", "RandomForestClassifier"]:
                st.info(f"ℹ️ Model is {model_type}. Using UNSCALED data (as per training).")
                prediction_df = input_df
            else:
                st.warning(f"⚠️ Unknown model type: {model_type}. Defaulting to scaled data.")
                prediction_df = input_scaled_df

            pred = model.predict(prediction_df)[0]
            prob_all = model.predict_proba(prediction_df)[0]
            prob_low = prob_all[0]
            prob_high = prob_all[1]

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
                st.write("🔧 Final input data sent to model:")
                st.dataframe(prediction_df)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif submitted:
        st.error("⚠️ Model or scaler not loaded correctly. Check your files.")


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BEGIN TAB 2: DATA DASHBOARD (Your new code)
# -------------------------------------------------------------------
# -------------------------------------------------------------------

with tab2:
    st.header("📊 Comprehensive Accident Data Dashboard")

    uploaded = st.file_uploader("📂 Upload your accident dataset (CSV)", type=['csv'])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, low_memory=False)
            st.subheader("👀 Dataset Preview")
            st.dataframe(df.head())

            # Handle datetime if available
            for col in df.columns:
                if 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass

            # --- KPI Summary Section ---
            st.markdown("### 🚦 Key Insights Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(df):,}")
            if 'Severity' in df.columns:
                col2.metric("Unique Severities", df['Severity'].nunique())
                col3.metric("Avg Severity", round(df['Severity'].mean(), 2))
            if 'State' in df.columns:
                col4.metric("Total States", df['State'].nunique())

            # --- Filters Section ---
            st.markdown("### 🎚️ Filters")
            c1, c2, c3 = st.columns(3)

            states = sorted(df['State'].dropna().unique()) if 'State' in df.columns else []
            months = sorted(df['Start_month'].dropna().unique()) if 'Start_month' in df.columns else []
            severities = sorted(df['Severity'].dropna().unique()) if 'Severity' in df.columns else []

            selected_state = c1.selectbox("State", ["All"] + list(states))
            selected_month = c2.selectbox("Month", ["All"] + [int(m) for m in months]) if months else "All"
            selected_severity = c3.selectbox("Severity", ["All"] + [int(s) for s in severities]) if severities else "All"

            filtered_df = df.copy()
            if selected_state != "All":
                filtered_df = filtered_df[filtered_df['State'] == selected_state]
            if selected_month != "All":
                filtered_df = filtered_df[filtered_df['Start_month'] == int(selected_month)]
            if selected_severity != "All":
                filtered_df = filtered_df[filtered_df['Severity'] == int(selected_severity)]

            st.info(f"🔍 {len(filtered_df)} records displayed after applying filters")

            # --- 1. Severity Distribution ---
            if 'Severity' in filtered_df.columns:
                st.markdown("### 🚦 Severity Distribution")
                fig = px.histogram(filtered_df, x='Severity', color='Severity',
                                   title="Distribution of Accident Severity Levels")
                st.plotly_chart(fig, use_container_width=True)

            # --- 2. Accidents Over Time ---
            time_cols = [c for c in df.columns if 'time' in c.lower()]
            if time_cols:
                st.markdown("### ⏱️ Accidents Over Time")
                time_col = time_cols[0]
                ts = filtered_df.copy()
                ts[time_col] = pd.to_datetime(ts[time_col], errors='coerce')
                ts = ts.dropna(subset=[time_col])
                ts['date'] = ts[time_col].dt.date
                trend = ts.groupby('date').size().reset_index(name='Count')
                fig = px.line(trend, x='date', y='Count', title="Trend of Accidents Over Time")
                st.plotly_chart(fig, use_container_width=True)

            # --- 3. Accidents by Hour ---
            if 'Start_hour' in filtered_df.columns:
                st.markdown("### 🕐 Accidents by Hour of Day")
                fig = px.bar(filtered_df.groupby('Start_hour').size().reset_index(name='Count'), 
                             x='Start_hour', y='Count', title="Frequency of Accidents by Hour")
                st.plotly_chart(fig, use_container_width=True)

            # --- 4. Accidents by Day of Week ---
            if 'Day_of_week' in filtered_df.columns:
                st.markdown("### 📅 Accidents by Day of Week")
                fig = px.bar(filtered_df.groupby('Day_of_week').size().reset_index(name='Count'),
                             x='Day_of_week', y='Count', title="Day-wise Accident Frequency")
                st.plotly_chart(fig, use_container_width=True)

            # --- 5. Accidents by Month ---
            if 'Start_month' in filtered_df.columns:
                st.markdown("### 🗓️ Accidents by Month")
                fig = px.bar(filtered_df.groupby('Start_month').size().reset_index(name='Count'),
                             x='Start_month', y='Count', title="Month-wise Accident Frequency")
                st.plotly_chart(fig, use_container_width=True)

            # --- 6. Top States by Accident Count ---
            if 'State' in filtered_df.columns:
                st.markdown("### 🌎 Top 15 States by Accidents")
                top_states = filtered_df['State'].value_counts().nlargest(15).reset_index()
                top_states.columns = ['State', 'Count']
                fig = px.bar(top_states, x='State', y='Count', color='Count', title="Top 15 States by Accident Count")
                st.plotly_chart(fig, use_container_width=True)

            # --- 7. Weather Condition vs Severity ---
            if {'Weather_Condition', 'Severity'}.issubset(filtered_df.columns):
                st.markdown("### 🌦️ Weather Conditions vs Severity")
                top_weather = filtered_df['Weather_Condition'].value_counts().nlargest(10).index
                sub = filtered_df[filtered_df['Weather_Condition'].isin(top_weather)]
                fig = px.box(sub, x='Weather_Condition', y='Severity', color='Weather_Condition',
                             title="Top 10 Weather Conditions and Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # --- 8. Correlation Heatmap ---
            st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
            num_df = filtered_df.select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr()
                fig = px.imshow(corr, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

            # --- 9. Wind Speed vs Severity Scatter ---
            if {'Wind_Speed(mph)', 'Severity'}.issubset(filtered_df.columns):
                st.markdown("### 🌪️ Wind Speed vs Severity")
                fig = px.scatter(filtered_df, x='Wind_Speed(mph)', y='Severity', color='Severity',
                                 title="Relation between Wind Speed and Severity", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

            # --- 10. Temperature vs Humidity ---
            if {'Temperature(F)', 'Humidity(%)'}.issubset(filtered_df.columns):
                st.markdown("### 🌡️ Temperature vs Humidity")
                fig = px.scatter(filtered_df, x='Temperature(F)', y='Humidity(%)', color='Severity' if 'Severity' in filtered_df.columns else None,
                                 title="Temperature vs Humidity (Colored by Severity)")
                st.plotly_chart(fig, use_container_width=True)

            # --- 11. Duration vs Severity ---
            if {'Duration', 'Severity'}.issubset(filtered_df.columns):
                st.markdown("### ⏳ Duration vs Severity")
                fig = px.violin(filtered_df, x='Severity', y='Duration', box=True, color='Severity',
                                 title="Distribution of Duration Across Severity Levels")
                st.plotly_chart(fig, use_container_width=True)

            # --- 12. Accident Map (Geospatial) ---
            if {'Start_Lat', 'Start_Lng'}.issubset(filtered_df.columns):
                st.markdown("### 🗺️ Accident Hotspot Map")
                sample = filtered_df.sample(min(len(filtered_df), 3000))
                fig = px.scatter_mapbox(
                    sample,
                    lat='Start_Lat', lon='Start_Lng',
                    color='Severity' if 'Severity' in sample.columns else None,
                    hover_data=['State', 'Weather_Condition'] if 'Weather_Condition' in sample.columns else ['State'],
                    zoom=3, height=500, title="Accident Hotspots Map"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading dataset: {e}")
    else:
        st.info("📥 Please upload a CSV file to explore accident insights.")


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BEGIN TAB 3: SAFETY PRECAUTIONS (Your new code)
# -------------------------------------------------------------------
# -------------------------------------------------------------------

with tab3:
    st.header("🧠 Smart Safety Precautions & Driving Tips")

    st.markdown("### ⚙️ Choose a Category to Explore:")
    category = st.radio(
        "Select a section:",
        ["🚗 Before Driving", "🌧️ While Driving", "🚨 In Emergencies", "📈 Safety Insights"],
        horizontal=True
    )

    if category == "🚗 Before Driving":
        st.subheader("🚗 Before Driving")
        st.markdown("""
        ✅ **Vehicle Checks**
        - Inspect **brakes, tires, lights**, and **windshield wipers**.
        - Ensure fuel and fluids (oil, coolant) are sufficient.
        - Check mirrors for proper alignment.

        ⚠️ **Personal Readiness**
        - Avoid alcohol or fatigue before starting.
        - Ensure seat belts for all passengers.
        - Plan your route and check weather updates.
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995574.png", width=150)

    elif category == "🌧️ While Driving":
        st.subheader("🌧️ While Driving")
        st.markdown("""
        🛣️ **Driving Discipline**
        - Maintain a **safe following distance**.
        - Obey **speed limits** and **traffic lights**.
        - Keep headlights on in **fog, rain, or low visibility**.

        📵 **Avoid Distractions**
        - Do **not use mobile phones**.
        - Avoid multitasking like eating or adjusting the radio.
        """)
        st.image("https.cdn-icons-png.flaticon.com/512/2682/2682065.png", width=150)

    elif category == "🚨 In Emergencies":
        st.subheader("🚨 In Emergencies")
        st.markdown("""
        🚗 **Stay Calm**
        - Move your vehicle to a **safe zone** if possible.
        - Turn on **hazard lights** and place reflective triangles.

        📞 **Seek Help**
        - Call emergency services or roadside assistance.
        - Provide **location and condition details** accurately.

        ❤️ **First Aid**
        - Check for injuries and apply first aid if trained.
        - Avoid moving seriously injured people unless necessary.
        """)
        st.image("https.cdn-icons-png.flaticon.com/512/3050/3050525.png", width=150)

    elif category == "📈 Safety Insights":
        st.subheader("📊 Safety Insights from Data")
        st.markdown("""
        See how accident patterns change with conditions.
        """)
        st.info("Upload a CSV in the 'Accident Dashboard' tab for deeper insights before exploring these visualizations.")

        sample_data = pd.DataFrame({
            "Condition": ["Clear", "Rain", "Fog", "Snow", "Night"],
            "Avg_Severity": [0.4, 0.65, 0.78, 0.72, 0.58]
        })

        st.bar_chart(sample_data.set_index("Condition"))
        st.caption("⚡ Observations: Accidents in fog and snow tend to have higher severity rates.")

# ----------------------------------------------------
# 8️⃣ Footer (Global)
# ----------------------------------------------------
st.markdown("---")
st.caption("Powered by XGBoost | Built with Streamlit")