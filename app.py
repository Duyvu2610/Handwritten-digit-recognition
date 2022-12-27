import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('D:\Work space\\ai\mnist3.hdf5')
st.title("Nhận dạng chữ số viết tay")


# Create a canvas component
canvas_result = st_canvas(
    fill_color="#ffffff",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#fff",
    background_color="#000",
    height=150,
    width=150,
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data, (28, 28))

predict = st.button('Predict')
if predict:
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(pred[0])}')
