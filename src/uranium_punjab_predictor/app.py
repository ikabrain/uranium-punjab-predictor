import streamlit as st
import pandas as pd
from pathlib import Path
from uranium_punjab_predictor.utils.model import load_model, prepare_features

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"

# Major districts for dropdown
DISTRICTS = [
    "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka",
    "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana",
    "Malerkotla", "Mansa", "Moga", "Mohali", "Muktsar", "Pathankot", "Patiala",
    "Rupnagar", "Sangrur", "Tarn Taran"
]

def main():
    st.set_page_config(page_title="Punjab Groundwater Uranium Predictor", layout="centered")
    st.title("Punjab Groundwater Uranium Predictor")
    st.markdown(
        """
        Enter your Punjab district, latitude, and longitude to get a prediction of uranium concentration in groundwater.
        
        *(Based on the data from the 2023 Punjab Groundwater Survey)*
        """
    )

    with st.form("prediction_form", clear_on_submit=False):
        district = st.selectbox("District", DISTRICTS)
        lat = st.number_input("Latitude", format="%.6f", min_value=27.0, max_value=34.0)
        lon = st.number_input("Longitude", format="%.6f", min_value=73.0, max_value=77.0)
        predict_btn = st.form_submit_button("Predict")

    result_html = None
    err_msg = None

    if predict_btn:
        # Input validation
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            err_msg = "Latitude and Longitude must be valid numbers."

        if not err_msg and (lat_f < 27 or lat_f > 34 or lon_f < 73 or lon_f > 77):
            err_msg = "Latitude/Longitude out of valid Punjab range."

        features = None
        model = None
        if not err_msg:
            try:
                model = load_model(str(MODEL_PATH))
            except Exception as e:
                err_msg = f"Model could not be loaded: {str(e)}"

        if not err_msg:
            try:
                  # Only use latitude and longitude for features
                features = pd.DataFrame({
                    "latitude": [lat_f],
                    "longitude": [lon_f]
                })
            except Exception as e:
                err_msg = f"Error preparing input features: {str(e)}"

        if not err_msg:
            try:
                pred        = model.predict(features)[0]
                result_html = f"<div style='padding:1.5em 1em;background:#F3F6FA;border-radius:8px;font-size:1.25em;font-weight:600;color:#173042;text-align:center;'>\n<strong>District:</strong> {district}<br>Predicted Uranium Concentration:<br><span style='font-size:2em;color:#4F8A10'>{pred:.2f} \u03bcg/L</span></div>"
            except Exception as e:
                err_msg = f"Prediction failed: {str(e)}"

    if result_html:
        st.markdown(result_html, unsafe_allow_html=True)
    if err_msg:
        st.error(err_msg)

if __name__ == "__main__":
    main()
