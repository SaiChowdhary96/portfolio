import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
@st.cache_resource
def load_model():
    model = load("logistic_regression_pipeline.joblib")  # Load the model using joblib
    return model

model = load_model()

# Define the Streamlit app
def main():
    st.title("Mushroom Classification Prediction App")
    st.write(
        "This app predicts the class of a mushroom based on its features."
    )

    # Input fields for the user
    cap_diameter = st.number_input("Cap Diameter (float)", value=1.0, step=0.1)
    cap_shape = st.selectbox("Cap Shape (integer)", options=[2, 6])
    gill_attachment = st.selectbox("Gill Attachment (integer)", options=[2])
    gill_color = st.selectbox("Gill Color (integer)", options=[10])
    stem_height = st.number_input("Stem Height (float)", value=1.0, step=0.1)
    stem_color = st.selectbox("Stem Color (integer)", options=[11])
    season = st.number_input("Season (float)", value=1.0, step=0.1)

    # Create a dataframe for the input features
    input_data = pd.DataFrame(
        {
            "cap_diameter": [cap_diameter],
            "cap_shape": [cap_shape],
            "gill_attachment": [gill_attachment],
            "gill_color": [gill_color],
            "stem_height": [stem_height],
            "stem_color": [stem_color],
            "season": [season],
        }
    )

    # Show the input data
    st.write("Input Data:", input_data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Class: {prediction[0]}")

if __name__ == "__main__":
    main()
