from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st
import requests
import json

#judul
st.title("Waste Image Classification")
st.write("Upload a waste for image classification as Organic or Recyclable")
st.write("*This page is made by Fadhilah Amani*")
st.markdown('---')

#load model
model = load_model('model/classification_model.hdf5')

def predict(image):
  loaded_model = model
  img = image.resize((220, 220))
  x = np.asarray(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = loaded_model.predict(images)

  # input ke model
  new_data = x.tolist()
  input_data_json = json.dumps({
      "signature_name": "serving_default",
      "instances": new_data
  })

  # inference
  URL = "http://waste-back-end.herokuapp.com/v1/models/classification_model:predict"
  r = requests.post(URL, data=input_data_json)

  if r.status_code == 200:
      res = r.json()
      if res['predictions'][0][0] == 1 :
          st.write('**RECYCLABLE**')
      else:
          st.write('**ORGANIC**')
  else:
    st.write('Error')

uploaded = st.file_uploader('Choose An Image To Upload')
if uploaded:
        image = Image.open(uploaded)
        new_image = image.resize((220, 220))
        st.image(new_image, caption='Your Uploaded Image')
        st.write("")
        # st.image(image, caption='Your Uploaded Image', use_column_width=False,)
        st.write("")
        if st.button('Predict'):
                st.write("The given image has been classified as:")
                label = predict(image)



