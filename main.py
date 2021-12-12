
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
st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)
good_tree = ["Azadirachta Indica","Carissa Carandas", "Ficus Religiosa"]

def main():
    with input_col:
        st.header('Input part')
        st.write('Select the picture to test')
        random_image = []
        for i in range(1, 10):
            file_path_type = ["./leaf dataset/Azadirachta Indica (Neem)/*.jpg", "./leaf dataset/Carissa Carandas (Karanda)/*.jpg", "./leaf dataset/Ficus Religiosa (Peepal Tree)/*jpg"]
            images = glob.glob(random.choice(file_path_type))
            random_image.insert(0, random.choice(images))
        pick_img = st.sidebar.radio("Which image?", [x for x in range(1, len(random_image) + 1)])
        st.write('mosaic for the picture')
        st.image(random_image, width=100)

        result = st.button('Test AI')
        #if result:
        #    st.image(random_image[pick_img - 1])

    if result:
        with AI_col:
            st.header('Result of the analyze')
            st.write('For this picture:')
            st.image(random_image[pick_img - 1], width=200)
            witch_tree = random_image[pick_img - 1][15:16]
            st.write(witch_tree)
            st.write('Our AI deterined this tree:')
            st.write('The real result is :')
            st.write('The result is good / is not')

if __name__ == '__main__':
	main()
