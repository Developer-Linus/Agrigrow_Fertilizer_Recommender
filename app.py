import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# 🔄 Load Cached Resources
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("fertilizer_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_label_encoders():
    return joblib.load("label_encoders.pkl")

model = load_model()
scaler = load_scaler()
label_encoders = load_label_encoders()

# -----------------------------
# 🌿 Fertilizer Remarks
# -----------------------------
remarks_dict = {
    "Compost": "Enhances organic matter and improves soil structure. Ideal for building long-term soil health.",
    "Balanced NPK Fertilizer": "Provides equal nitrogen, phosphorus, and potassium for general crop nutrition.",
    "Water Retaining Fertilizer": "Helps soil retain moisture, perfect for drought-prone areas.",
    "Organic Fertilizer": "Natural nutrients from plant/animal sources. Improves soil biology.",
    "Gypsum": "Improves soil structure and reduces compaction in clay soils.",
    "Lime": "Raises soil pH in acidic soils. Essential for calcium/magnesium.",
    "DAP": "Diammonium phosphate - high phosphorus starter fertilizer.",
    "Urea": "High nitrogen content for vigorous vegetative growth.",
    "Muriate of Potash": "Provides potassium for fruit quality and disease resistance.",
    "General Purpose Fertilizer": "Balanced blend suitable for most crops.",
}

# -----------------------------
# 🌱 Input Choices
# -----------------------------
SOIL_TYPES = sorted([
    'Loamy Soil', 'Peaty Soil', 'Acidic Soil', 'Neutral Soil',
    'Alkaline Soil'
])

CROPS = sorted([
    'rice', 'wheat', 'corn', 'soybean', 'barley', 'millet', 'maize',
    'Tea', 'Coffee', 'Cotton', 'Ground Nut', 'Peas', 'Rubber',
    'Sugarcane', 'Tobacco', 'Kidney Beans', 'Mung Bean', 'Lentil',
    'Jute', 'Black gram', 'Adzuki Beans', 'Pigeon Peas', 'Chickpea',
    'banana', 'grapes', 'apple', 'mango', 'muskmelon', 'orange',
    'papaya', 'pomegranate', 'watermelon'
])

# -----------------------------
# 🎨 Custom CSS Styling
# -----------------------------
def load_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4f5e8 100%);
            font-family: 'Arial', sans-serif;
        }
        .header {
            font-size: 1.2em;
            color: #2c3e50;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255,255,255,0.8);
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 30px;
            font-size: 0.9em;
            color: #7f8c8d;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Main App
# -----------------------------
def main():
    load_css()

    st.title('🌱 AfriGrow Fertilizer Recommender')
    st.markdown("""
    <div class="header">
    "Your Soil Speaks – We Translate! Get personalized fertilizer recommendations for optimal crop growth."
    </div>
    """, unsafe_allow_html=True)

    # Input Form
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0)
            moisture = st.slider('Soil Moisture', 0.0, 1.0, 0.5, help="0 = Dry, 1 = Saturated")
            rainfall = st.number_input('Rainfall (mm)', 0, 1000, 200)
            ph = st.slider('Soil pH Level', 0.0, 14.0, 6.5)
            carbon = st.slider('Carbon Content (%)', 0.0, 5.0, 1.0)

        with col2:
            nitrogen = st.number_input('Nitrogen (mg/kg)', 0, 200, 50)
            phosphorous = st.number_input('Phosphorous (mg/kg)', 0, 200, 50)
            potassium = st.number_input('Potassium (mg/kg)', 0, 200, 50)
            soil_type = st.selectbox('Soil Type', SOIL_TYPES)
            crop = st.selectbox('Crop', CROPS)

        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        try:
            # Construct raw DataFrame
            raw_input = pd.DataFrame([{
                'Temperature': temperature,
                'Moisture': moisture,
                'Rainfall': rainfall,
                'PH': ph,
                'Nitrogen': nitrogen,
                'Phosphorous': phosphorous,
                'Potassium': potassium,
                'Carbon': carbon,
                'Soil': soil_type,
                'Crop': crop
            }])

            # Encode categorical features
            raw_input['Soil'] = label_encoders['Soil'].transform(raw_input['Soil'])
            raw_input['Crop'] = label_encoders['Crop'].transform(raw_input['Crop'])

            # Scale all features including encoded Soil & Crop
            scaled_input = scaler.transform(raw_input)

            # Make prediction
            prediction = model.predict(scaled_input)
            predicted_fertilizer = label_encoders['Fertilizer'].inverse_transform(prediction)[0]

            # Show result
            st.markdown(f"""
            <div style="background-color:#e8f5e9; padding: 20px; border-left: 6px solid #43a047; border-radius: 10px; font-family: 'Segoe UI', sans-serif;">
                <h3 style="color:#2e7d32; margin-top: 0;">🌿 Recommended Fertilizer: <span style="color:#1b5e20;">{predicted_fertilizer}</span></h3>
                <p style="font-size: 16px; color: #444;"><strong>Why this recommendation:</strong> {remarks_dict.get(predicted_fertilizer, "This fertilizer is optimal for your current soil and crop conditions.")}</p>
                <p style="font-size: 15px; color: #555;"><strong>For:</strong> <span style="color:#008caf;">{crop}</span> in <span style="color:#6d4c41;">{soil_type}</span> soil</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div style="background-color:#ffebee; padding: 20px; border-left: 6px solid #f44336; border-radius: 10px; font-family: 'Segoe UI', sans-serif;">
                <h3 style="color:#d32f2f; margin-top: 0;">⚠️ Prediction Error</h3>
                <p style="font-size: 16px; color: #555;"><strong>Could not generate recommendation:</strong> {str(e)}</p>
                <p style="font-size: 15px; color: #666;">Please check your inputs and try again.</p>
            </div>
            """, unsafe_allow_html=True)
    # Footer
    st.markdown("""
    <div class="footer">
    AfriGrow - Smart Fertilizer Recommendations for African Farmers 🌍<br>
    Data-driven agriculture for better yields.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
