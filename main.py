import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import glob, random
from tensorflow import keras
from tensorflow.keras import layers


# Layout

st.set_page_config(layout="wide")
input_col, AI_col = st.columns([1, 3])


with input_col:
    st.header('Input part')
    st.write('Select the picture to test')
    random_image = []
    for i in range(1, 10):
        file_path_type = ["./leaf dataset/Azadirachta Indica (Neem)/*.jpg", "./leaf dataset/Carissa Carandas (Karanda)/*.jpg", "./leaf dataset/Ficus Religiosa (Peepal Tree)/*jpg"]
        images = glob.glob(random.choice(file_path_type))
        random_image.insert(0, random.choice(images))
    #pick_img = st.sidebar.radio("Which image?", [x for x in range(1, len(random_image))])
    st.write('mosaic for the picture')
    for x in range(1, len(random_image)):
        st.button(st.image(x, width=100))

#if pick_img:
#    st.header('Result of the analyze')
#    st.write('For this picture:')
#    print(pick_img)
#    st.write('display the picture')
#    st.write('Our AI deterined this tree:')
#    st.write('The real result is :')
#    st.write('The result is good / is not')



#    st.button('Test AI')



