import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def predict(img):
    img_size = (128, 128)
    unique_labels =  ['pituitary', 'notumor', 'meningioma', 'glioma']
    img = img.resize(img_size)
    img = Image.fromarray(np.uint8(img))
    # img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8,1.2))
    # img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8,1.2))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img)
    pred = np.argmax(pred, axis=-1)
    print(pred)
    pred = unique_labels[pred[0]]
    return pred

def app():
    st.title('Brain Tumor Detection (Project by Toufiq Rahatwilkar)')
    st.markdown('This app detects if the patient has pituitary, meningioma, glioma Brain Tumor')
    st.markdown('The app is based on MobileNet model pre-trained on ImageNet dataset.')
    st.markdown("#")


    uploaded_image = st.file_uploader('Upload an image to predict')
    

    if uploaded_image:
        st.image(uploaded_image)
        img = Image.open(uploaded_image)
        pred_button = st.button("Predict")
        if pred_button:
            prediction = predict(img)
            print(prediction)
            if prediction == "notumor":
                st.subheader('The patient is not suffering from Brain Tumor 😄🎉🎉')
                st.balloons()
            else:
                st.subheader(f'The patient is suffering from {prediction} 😔')
            

if __name__ == "__main__":
    model = tf.keras.models.load_model("mobile_net.h5")
    app()

