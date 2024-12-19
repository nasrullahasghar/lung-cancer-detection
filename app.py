import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Model path (update if necessary)
model_path = "trained_lung_cancer_model.h5"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.write("Here are the files in the directory:")
    st.write(os.listdir())  # Debugging output to show directory contents
else:
    # Load the pre-trained model
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        st.write(f"Model input shape: {model.input_shape}")  # Display expected input shape
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
    
    # Class labels corresponding to your dataset
    class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

    # Streamlit app
    st.title("Lung Cancer Image Classification")
    st.write("Upload an image of a lung tissue slide to classify.")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for prediction
        img = Image.open(uploaded_file)
        
        # Convert to RGB if the image is grayscale
        img = img.convert('RGB')  # Ensure 3 channels even for grayscale images
        img = img.resize((350, 350))  # Resize to match model input shape
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        try:
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)

            # Show the predicted class and probability
            st.write(f"Prediction: {class_labels[predicted_class[0]]}")
            st.write(f"Prediction Probability: {prediction[0][predicted_class[0]]:.2f}")
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
