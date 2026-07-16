import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Styling for the title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Input section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2F3E46;
        border-bottom: 2px solid #FF4B2B;
        padding-bottom: 5px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Prediction output card */
    .prediction-card {
        background: linear-gradient(135deg, #1f4068, #162447);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #00F2FE;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
        margin: 0.5rem 0;
    }
    
    .prediction-label {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #e4e4e4;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-title">FastPredict 🚴</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Food Delivery Time Estimation with Random Forest</div>', unsafe_allow_html=True)

# Load paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
encoders_path = os.path.join(BASE_DIR, "label_encoders.pkl")

@st.cache_resource
def load_resources():
    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        return None, None, f"Required files not found. Ensure `random_forest_model.pkl` and `label_encoders.pkl` are in: {BASE_DIR}"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders, None
    except Exception as e:
        return None, None, f"Error loading model resources: {str(e)}"

# Load resources
model, encoders, error_msg = load_resources()

if error_msg:
    st.error(error_msg)
    st.stop()

# Define features expected by model
feature_names = ['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 
                 'Preparation_Time_min', 'Courier_Experience_yrs']

# Information banner
st.info("💡 Fill out the details below to predict the estimated delivery time in minutes.")

# Form setup
with st.form("delivery_details_form"):
    st.markdown('<div class="section-header">📍 Trip & Location Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input(
            "Delivery Distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Total distance between the restaurant and the delivery location.",
            format="%.2f"
        )
    with col2:
        prep_time = st.number_input(
            "Preparation Time (minutes)",
            min_value=1,
            max_value=120,
            value=15,
            step=1,
            help="Time taken by the restaurant to prepare the food."
        )

    st.markdown('<div class="section-header">🌤️ Environment & Courier Details</div>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        weather_options = list(encoders['Weather'].classes_) if 'Weather' in encoders else ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy']
        weather = st.selectbox(
            "Weather Conditions",
            options=weather_options,
            help="Current weather conditions during delivery."
        )
    with col4:
        traffic_options = list(encoders['Traffic_Level'].classes_) if 'Traffic_Level' in encoders else ['High', 'Low', 'Medium']
        traffic = st.selectbox(
            "Traffic Level",
            options=traffic_options,
            help="Expected traffic levels on the route."
        )
    with col5:
        time_options = list(encoders['Time_of_Day'].classes_) if 'Time_of_Day' in encoders else ['Afternoon', 'Evening', 'Morning', 'Night']
        time_of_day = st.selectbox(
            "Time of Day",
            options=time_options,
            help="Time window when the delivery is happening."
        )

    col6, col7 = st.columns(2)
    with col6:
        vehicle_options = list(encoders['Vehicle_Type'].classes_) if 'Vehicle_Type' in encoders else ['Bike', 'Car', 'Scooter']
        vehicle_type = st.selectbox(
            "Vehicle Type",
            options=vehicle_options,
            help="Mode of transport used by the courier."
        )
    with col7:
        courier_exp = st.number_input(
            "Courier Experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
            help="Courier's total delivery experience in years.",
            format="%.1f"
        )

    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(
        label="🚀 Calculate Delivery Time",
        use_container_width=True
    )

if submit_button:
    # Prepare input dictionary
    input_data = {
        'Distance_km': distance,
        'Weather': weather,
        'Traffic_Level': traffic,
        'Time_of_Day': time_of_day,
        'Vehicle_Type': vehicle_type,
        'Preparation_Time_min': prep_time,
        'Courier_Experience_yrs': courier_exp
    }
    
    # Process categorical variables using label encoders
    processed_data = input_data.copy()
    encoding_error = False
    
    for col, encoder in encoders.items():
        if col in processed_data:
            val = processed_data[col]
            try:
                # Transform using the pre-fit label encoder
                processed_data[col] = encoder.transform([val])[0]
            except Exception as e:
                st.error(f"Error encoding feature '{col}' with value '{val}': {str(e)}")
                encoding_error = True
                
    if not encoding_error:
        # Create DataFrame and order columns to match training features
        input_df = pd.DataFrame([processed_data])
        input_df = input_df[feature_names]
        
        try:
            # Predict
            prediction = model.predict(input_df)[0]
            
            # Display premium prediction card
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Delivery Time</div>
                <div class="prediction-value">{prediction:.1f} Mins</div>
                <div style="font-size: 0.9rem; color: #8a99ad;">
                    Based on standard route conditions and Courier Experience of {courier_exp} years
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Boundary warning messages (Boundary Value Analysis check)
            out_of_bounds = []
            if distance < 0.59 or distance > 19.99:
                out_of_bounds.append(f"Distance ({distance} km) is outside the typical training range (0.59 - 19.99 km).")
            if prep_time < 5 or prep_time > 29:
                out_of_bounds.append(f"Preparation Time ({prep_time} mins) is outside the typical training range (5 - 29 mins).")
            if courier_exp < 0.0 or courier_exp > 9.0:
                out_of_bounds.append(f"Courier Experience ({courier_exp} yrs) is outside the typical training range (0.0 - 9.0 yrs).")
                
            if out_of_bounds:
                st.warning("⚠️ **Boundary Value Note**: Some inputs are outside historical training bounds. Because Random Forest is a tree-based model, predictions for out-of-bound inputs are capped at training limits to maintain accuracy and prevent extreme/invalid predictions.\n\n" + "\n".join([f"- {msg}" for msg in out_of_bounds]))
            
            # Visual feedback indicators
            if prediction < 20:
                st.success("🟢 Fast delivery expected!")
            elif prediction < 35:
                st.info("🔵 Standard delivery time expected.")
            else:
                st.warning("🟡 Potential delays expected due to traffic/weather/distance.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

