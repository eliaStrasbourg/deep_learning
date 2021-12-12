import seaborn
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import glob, random
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras import layers


# Layout

st.set_page_config(layout="wide")
input_col, AI_col = st.columns([1, 2])
good_tree = ["Azadirachta Indica","Carissa Carandas", "Ficus Religiosa"]

random_image = []
pick_img = []
def randomImageGenerator():
	i = 1
	while i < 10:
		file_path_type = ["./leaf dataset/Azadirachta Indica (Neem)/*.jpg", "./leaf dataset/Carissa Carandas (Karanda)/*.jpg", "./leaf dataset/Ficus Religiosa (Peepal Tree)/*jpg"]
		images = glob.glob(random.choice(file_path_type))
		random_image.insert(0, random.choice(images))
		i += 1
	pick_img = st.sidebar.radio("Which image?", [x for x in range(1, len(random_image) + 1)])

def tree(witch_tree):
    if witch_tree == 'A':
        return(good_tree[0])
    if witch_tree == 'C':
        return(good_tree[1])
    if witch_tree == 'F':
        return(good_tree[2])


def main():
    with input_col:
        st.header('Input part')
        st.write('Select the picture to test')
        st.write('mosaic for the picture')
	
        plt.axis('off')
        fig, ax = plt.subplots(nrows = 3, ncols = 3)

        for x in range(3):
            for y in range(3):
                img = mpimg.imread(random_image[y + x * 3])
                ax[x, y].imshow(img)
                #ax[x, y].set_title(y + x * 3 + 1)
                ax[x, y].axis('off')
                ax[x, y].text(0.5,-0.1, y + x * 3 + 1, size=12, ha="center", 
         transform=ax[x, y].transAxes)
        st.pyplot(fig)

        result = st.button('Test AI')

    if result:
        with AI_col:
            st.header('Result of the analyze')
            st.write('For this picture:')
            st.image(random_image[pick_img - 1], width=200)
            witch_tree = random_image[pick_img - 1][15:16]
            st.write('Our AI deterined this tree:')
            ia_tree = ""
            st.write(ia_tree)
            st.write('The real result is :')
            real_tree = tree(witch_tree)
            st.write(real_tree)
            if ia_tree == real_tree:
                st.markdown('<p style="font-family:sans-serif; color:Green; font-size: 42px;">the result is correct</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 42px;">the result isn t correct</p>', unsafe_allow_html=True)



if __name__ == '__main__':
	randomImageGenerator()
	main()
