import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
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
#import plotly.graph_objects as go
from random import randint
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from PIL import Image as im 
from tensorflow.keras.utils import plot_model
import visualkeras
from PIL import ImageFont



#/content/drive/MyDrive/btyse_2025/data_arrays
negative_scans = np.load("/Users/ronanhawkins/Desktop/coding/btyse25/pneumonia_xray/data_arrays/negative_scans.npy")
positive_scans = np.load("/Users/ronanhawkins/Desktop/coding/btyse25/pneumonia_xray/data_arrays/positive_scans.npy")
negative_labels = np.load("/Users/ronanhawkins/Desktop/coding/btyse25/pneumonia_xray/data_arrays/negative_labels.npy")
positive_labels = np.load("/Users/ronanhawkins/Desktop/coding/btyse25/pneumonia_xray/data_arrays/positive_labels.npy")
print(negative_scans[66].shape)
print("PS:",positive_scans.shape[0])
print("PL:",positive_labels.shape[0])
print("NS:",negative_scans.shape[0])
print("NL:",negative_labels.shape[0])


random.seed(42)
random_start = randint(0, positive_scans.shape[0]-1533)
positive_scans = positive_scans[random_start:random_start+1533]
positive_labels = positive_labels[random_start:random_start+1533]

print("PS",positive_scans.shape[0])
print("PL",positive_labels.shape[0])
print("NS",negative_scans.shape[0])
print("NL",negative_labels.shape[0])
x = np.concatenate((positive_scans, negative_scans), axis=0)
y = np.concatenate((positive_labels, negative_labels), axis=0)
x, y = shuffle(x, y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print("xtrain:",x_train.shape[0])
print("ytrain:",y_train.shape[0])
print("xtest:",x_test.shape[0])
print("ytest:",y_test.shape[0])

del(positive_scans)
del(positive_labels)
del(negative_scans)
del(negative_labels)
del(x)
del(y)
gc.collect()

def scipy_rotate(image):
        angles = [-20,-15,-10,-5,0,5,10,15,20]
        angle = random.choice(angles)
        image = ndimage.rotate(image,angle,reshape=False)
        image[image<0] = 0
        image[image>1] = 1
        return image

def rotate(image):
    augmented_image = tf.numpy_function(scipy_rotate,[image],tf.float32)
    return augmented_image

rotate=tf.function(rotate)

def train_process(image,label):
    image = rotate(image)
    image = tf.expand_dims(image, axis=2)
    return image, label

def test_process(volume,label):
    volume = tf.expand_dims(volume, axis=2)
    return volume, label

train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_loader = tf.data.Dataset.from_tensor_slices((x_test,y_test))

batch_size = 8

train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_process)
    .batch(batch_size)
    .prefetch(32)
)
test_dataset = (
    test_loader.shuffle(len(x_test))
    .map(test_process)
    .batch(batch_size)
    .prefetch(32)
)

### CNN ###
###########

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 768, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

initial_learning_rate = 0.00000001
learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1875, decay_rate=0.97, staircase=True)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=Adam(learning_rate=initial_learning_rate),metrics=['accuracy'])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("chest_xray_classification.keras", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=100)
#font = ImageFont.load_default()
font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 32)
visualkeras.layered_view(model, draw_volume=False, legend=True, font=font, scale_xy=1, scale_z=1, max_z=500).show()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.summary()

history = model.fit(x_train, 
                    y_train, 
                    epochs=120, 
                    shuffle=True,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint_cb, early_stopping_cb])

fig,ax = plt.subplots(1,2, figsize=(25,5))
ax = ax.ravel()
#Orange is validation
for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)





"""
data = im.fromarray(rotate(x_test[5]))
data.save('test.png')
"""
