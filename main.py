import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Skin Diseases 🩺", page_icon="🩺", layout="centered")
st.title('Skin Diseases 🩺')

# model = YOLO(r'./models/yolo-medical_10epoch.pt')
model = YOLO(r'./models/yolo-medical_30epoch.pt')
class_name = model.names

st.subheader("Upload Image")
file_uploader = st.file_uploader('Upload The Image', type=['jpg', 'png', 'jpeg'])

if file_uploader is not None:
    try:
        with st.spinner("Waiting for the prediction..."):
            st.subheader("Model output")
            image = Image.open(file_uploader)
            results = model.predict(image)
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()


        st.info(f'Prediction: {names_dict[np.argmax(probs)]} | Probability:  {round(np.max(probs) * 100, 3)}%', icon='🤖')
        st.image(image, caption='Output Image', width=550)

        st.write("""
            **Disclaimer**: All AI-generated diagnoses and advice are for informational purposes only and should not replace professional medical consultation.
            """)

    except: st.info('Pleases Enter The Correct Image.')


# Add Footer at the Bottom
footer = """
<style>
footer {
    visibility: hidden;
}
.main-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #f9f9f9;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #333;
}
</style>
<div class="main-footer">
    🌐 Created by Osama-Abo-Bakr | Skin Diseases Detection.
    🌐 +20-1274011748
</div>
"""
st.markdown(footer, unsafe_allow_html=True)