import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv() ## load all the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel('gemini-pro')

# Class mapping
class_mapping = {
    0: 'Apple scab',
    1: 'Apple Black rot',
    2: 'Apple Cedar apple rust',
    3: 'Apple healthy',
    4: 'Cherry Powdery mildew',
    5: 'Cherry healthy',
    6: 'Grape Black rot',
    7: 'Grape Esca(Black Measles)',
    8: 'Grape Leaf blight(Isariopsis Leaf Spot)',
    9: 'Grape healthy',
    10: 'Peach Bacterial spot',
    11: 'Peach healthy',
    12: 'Pepper bell Bacterial spot',
    13: 'Pepper bell healthy'
}

# Load the three models
model_red_filter = load_model('Red_Model.h5')
model_green_filter = load_model('Green_Model.h5')
model_blue_filter = load_model('Blue_Model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize image to match the input shape expected by the models
    image = image.resize((224, 224))
    # Convert image to array and normalize pixel values
    image_array = np.array(image) / 255.0
    return image_array

# Function to get ensemble predicted class for an image
def get_ensemble_pred_class(image):
    # Reshape the image to match the input shape expected by the models
    image_red = image[:, :, 0].reshape(1, 224, 224, 1)
    image_green = image[:, :, 1].reshape(1, 224, 224, 1)
    image_blue = image[:, :, 2].reshape(1, 224, 224, 1)
    
    # Get predictions from each model for the image
    pred_red = model_red_filter.predict(image_red)
    pred_green = model_green_filter.predict(image_green)
    pred_blue = model_blue_filter.predict(image_blue)
    
    # Average the predicted probabilities for each class
    ensemble_pred_proba = (pred_red + pred_green + pred_blue) / 3.0
    
    # Get the predicted class
    ensemble_pred_class = np.argmax(ensemble_pred_proba)
    
    return ensemble_pred_class

# Streamlit UI
st.title('Image Classification with Ensemble Models')
st.write('Upload an image for plant disease detection')

uploaded_file = st.file_uploader("Choose an image...", type=[ "JPG", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Add a submit button
    if st.button('Detect Disease'):
        # Get the ensemble predicted class
        ensemble_pred_class = get_ensemble_pred_class(processed_image)
        ensemble_pred_class_name = class_mapping[ensemble_pred_class]

        if "healthy" in ensemble_pred_class_name.lower():
            st.write(ensemble_pred_class_name)
        else:
            st.write('Output:', ensemble_pred_class_name)
            response = model.generate_content([ensemble_pred_class_name, "Give precautions and pest control measurements for the disease"])
            st.write(response.text)
