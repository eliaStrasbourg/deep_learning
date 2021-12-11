import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


# Layout

st.set_page_config(layout="wide")
input_col, AI_col = st.columns([1, 2])

with input_col:
    st.header('Input part')
    st.write('Select the picture to test')
    st.write('mosaic for the picture')
with AI_col:
    st.header('Result of the analyze')
    st.write('For this picture:')
    st.write('display the picture')
    st.write('Our AI deterined this tree:')
    st.write('The real result is :')
    st.write('The result is good / is not')

