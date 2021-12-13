import seaborn
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob, random
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras import layers

import os


# Layout

st.set_page_config(layout="wide")
input_col, AI_col = st.columns([1, 2])
good_tree = ["Azadirachta Indica","Carissa Carandas", "Ficus Religiosa"]

random_image = []
pick_img = []


@st.cache()
def loadModel():
    CLASS_NAMES = ['Azadirachta Indica (Neem)',
              'Carissa Carandas (Karanda)',
              'Ficus Religiosa (Peepal Tree)']
    CLASS_NUMBER = 3

    IMAGE_SIZE = (150, 200)
    IMAGE_SHAPE = (150, 200, 3)

    SHUFFLE_SIZE = 10000
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.8
    TEST_SPLIT = 0.2


    # 1. First we load the datasets

    def image_parser(filename, label):
        image_str = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image = tf.image.resize(image, IMAGE_SIZE)

        return image, label

    filenames = []
    labels = []
    df_size = 0

    for label_idx in range(len(CLASS_NAMES)):
        path = f'./leaf dataset/{CLASS_NAMES[label_idx]}'
        directory_finames = [os.path.join(path, image_file) for image_file in os.listdir(path)]

        filenames = filenames + directory_finames
        labels = labels + [tf.one_hot(label_idx, CLASS_NUMBER) for i in range(0, len(directory_finames))]

        print(label_idx, ':', CLASS_NAMES[label_idx], '->', tf.one_hot(label_idx, CLASS_NUMBER))

        df_size += len(directory_finames)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(image_parser)

    dataset = dataset.shuffle(SHUFFLE_SIZE)

    train_size = int(TRAIN_SPLIT * df_size)

    train = dataset.take(train_size)
    test = dataset.skip(train_size)

    train = train.batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)

    train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.prefetch(buffer_size=tf.data.AUTOTUNE)


    # 2. Create the model

    ## inputs and normalize
    inputs = layers.Input(shape=IMAGE_SHAPE)
    image = layers.Lambda(lambda img: img/255)(inputs)

    # conv1
    image = layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.MaxPooling2D((2, 2))(image)

    # conv2
    image = layers.Conv2D(64, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.Conv2D(64, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.MaxPooling2D((2, 2))(image)

    # flatten
    image = layers.Flatten()(image)

    # fully connected
    image = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(image)
    outputs = layers.Dense(CLASS_NUMBER, activation='softmax')(image)


    # 3. Train the model

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(
        train,
        epochs=10,
        verbose=2,
        validation_data=test
    )

    return model


def pickRandImage():
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
    model = loadModel()

    with input_col:
        pickRandImage()
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
	main()
