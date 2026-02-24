import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
import nibabel
from scipy import ndimage
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import os.path
import time
import gc
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from random import randint
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import visualkeras
from PIL import ImageFont


#/content/drive/MyDrive/btyse_2025/data_arrays
negative_scans = np.load("data_arrays/negative_scans.npy")
positive_scans = np.load("data_arrays/positive_scans.npy")
negative_labels = np.load("data_arrays/negative_labels.npy")
positive_labels = np.load("data_arrays/positive_labels.npy")

print(positive_scans.shape[0])
print(positive_labels.shape[0])
print(negative_scans.shape[0])
print(negative_labels.shape[0])
random.seed(42)
random_start = randint(0, negative_scans.shape[0]-300)
negative_scans = negative_scans[random_start:random_start+300]
negative_labels = negative_labels[random_start:random_start+300]

x = np.concatenate((positive_scans, negative_scans), axis=0)
y = np.concatenate((positive_labels, negative_labels), axis=0)
x, y = shuffle(x, y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print("Number of scans in test, train with 75/25 split: ", x_test.shape[0],",", x_train.shape[0])
del(positive_scans)
del(positive_labels)
del(negative_scans)
del(negative_labels)
del(x)
del(y)
gc.collect()


def rotate(volume):
    def scipy_rotate(volume):
        angles = [-20,-15,-10,-5,0,5,10,15,20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume,angle,reshape=False)
        volume[volume<0] = 0
        volume[volume>1] = 1
        return volume
    augmented_volume = tf.numpy_function(scipy_rotate,[volume],tf.float32)
    return augmented_volume

rotate=tf.function(rotate)

#process data, add channel
def train_process(volume,label):
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def test_process(volume,label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_loader = tf.data.Dataset.from_tensor_slices((x_test,y_test))

batch_size = 4

#augment the data during training
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_process)
    .batch(batch_size)
    .prefetch(32)
)

#Rescale test data
test_dataset = (
    test_loader.shuffle(len(x_test))
    .map(test_process)
    .batch(batch_size)
    .prefetch(32)
)


########
#NN    #
########

def  get_model(width=128, height=128, depth=18):
    #Define a 3D Conv Neural Network
    inputs = keras.Input((width,height,depth,1))
    print(inputs.shape)

    x=layers.Conv3D(filters=32, kernel_size=(5,5,5), padding='same', activation="relu")(inputs)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    x=layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    x=layers.Conv3D(filters=128, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    #x=layers.Conv3D(filters=256, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    #x=layers.MaxPool3D(pool_size=2)(x)
    #x=layers.BatchNormalization()(x)

    #x=layers.Conv3D(filters=512, kernel_size=7, padding='same', activation="relu")(x)
    #x=layers.MaxPool3D(pool_size=2)(x)
    #x=layers.BatchNormalization()(x)

    x=layers.GlobalAveragePooling3D()(x)
    x=layers.Dense(units=256,activation="relu")(x)
    x=layers.Dropout(0.6)(x)

    x = layers.Dense(3, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs,outputs, name="3dcnn")
    return model

model=get_model(width=128,height=128,depth=18)
model.summary()

initial_learning_rate = 0.000001
learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=2000, decay_rate=0.992, staircase=True)
#changed to same lr
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate_schedule),metrics=["binary_accuracy"])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("image_class_model.keras", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=100)
font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 32)
visualkeras.layered_view(model, draw_volume=True, legend=True).show()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

epochs = 800
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

fig,ax = plt.subplots(1,2, figsize=(25,5))
ax = ax.ravel()

for i, metric in enumerate(["binary_accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)