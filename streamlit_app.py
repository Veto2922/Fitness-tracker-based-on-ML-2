import streamlit as st
from src.models.predict_model import Tracker

# Set page title and background color
st.set_page_config(page_title="Exercises Prediction", page_icon=":weight_lifter:", layout="centered", initial_sidebar_state="collapsed")

# Add title and image
st.title("Fitness Tracker ML App")
st.image("barbell.jpg", caption="Image Source: Google")

# File uploaders for accelerometer and gyroscope data
acc_file = st.file_uploader("Upload an accelerometer file", type=["csv"])
gyr_file = st.file_uploader("Upload a gyroscope file", type=["csv"])

exercise = None
repetitions = None

if acc_file is not None and gyr_file is not None:
    # Get file names
    acc_filename = acc_file.name
    gyr_filename = gyr_file.name
    
    # Create Tracker object
    tr = Tracker(acc_path=acc_filename, gyr_path=gyr_filename)
else:
    st.warning("Please upload both accelerometer and gyroscope files.")
# Button to predict exercise type
if st.button("Predict the exercise type"):
    # Predict exercise type
    exercise = tr.model()
    # Display predicted exercise
    st.info(f"Predicted Exercise: {exercise}")
    

# Button to count repetitions
if st.button("Count repetitions"):
    # Count repetitions
    repetitions = tr.count_rep()
    # Display repetitions count
    st.info(f"Number of Repetitions: {repetitions}")
        
