
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model('D:\Work space\\ai\mnist.hdf5')
st.title("MNIST Digit Recognizer")


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
col1, col2 = st.columns(2)

SIZE = 192
# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(
        img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    with col1:
        st.write('Input Image')

predict = st.button('Predict')

if predict:
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(test_x.reshape(1, 28, 28, 1))
    with col2:
        st.write(f'result: {np.argmax(pred[0])}')
        st.bar_chart(pred[0])
