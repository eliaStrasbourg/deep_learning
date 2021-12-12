import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import glob, random
import streamlit.components.v1 as components
from tensorflow import keras
from tensorflow.keras import layers


# Layout

st.set_page_config(layout="wide")
C1, C2, C3 = st.columns(3)

random_image = []
for i in range(1, 11):
    file_path_type = ["./leaf dataset/Azadirachta Indica (Neem)/*.jpg", "./leaf dataset/Carissa Carandas (Karanda)/*.jpg", "./leaf dataset/Ficus Religiosa (Peepal Tree)/*jpg"]
    images = glob.glob(random.choice(file_path_type))
    random_image.insert(0, random.choice(images))

with C2:
    st.markdown("<h1 style='text-align: center; color: white;'>Input part</h1>", unsafe_allow_html=True)
    #st.header('Input part')
    st.write('Select the picture to test')
    #pick_img = st.sidebar.radio("Which image?", [x for x in range(1, len(random_image))])
    st.write('mosaic for the picture')
    #st.image(random_image, width=100)
    
    slected_image = []
    st.subheader('select one image')
    x for x in range(1, len(random_image)):
        st.image(random_image[x], caption=x)
        slected_image = st.checkbox(st.image(random_image[x], width=100, caption=x))
    if slected_image:
        st.image(slected_image, width=100)



#if pick_img:
#    st.header('Result of the analyze')
#    st.write('For this picture:')
#    print(pick_img)
#    st.write('display the picture')
#    st.write('Our AI deterined this tree:')
#    st.write('The real result is :')
#    st.write('The result is good / is not')



#    st.button('Test AI')



