import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

st.header('Input part')
st.expander('Select the picture to test')
st.write('mosaic for the picture')
st.header('Result of the analyze')
st.expander('For this picture:')
st.write('display the picture')
st.expander('Our AI deterined this tree:')
st.expander('The real result is :')
st.write('The result is good / is not')

