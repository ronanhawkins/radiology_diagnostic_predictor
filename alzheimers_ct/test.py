import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers import BatchNormalization
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
import os.path
import gc

#load npy files
negative_scans = np.load("./positive_scans.npy")
positive_scans = np.load("./negative_scans.npy")
negative_labels = np.load("./positive_labels.npy")
positive_labels = np.load("./negative_labels.npy")


#Compile the imaging from the positive/negative datasets into the testing and training datasets
print(positive_scans.shape)
print(negative_scans.shape)
print(positive_labels.shape[0])
print(negative_labels.shape[0])
negative_labels = negative_labels[:300]
negative_scans = negative_scans[:300]
print(positive_scans.shape[0])
print(positive_labels.shape[0])
print(negative_labels.shape[0])
#x_train = np.concatenate((positive_scans[:70], negative_scans[:70]), axis=0)
#y_train = np.concatenate((positive_labels[:70], negative_labels[:70]), axis=0)
#x_test = np.concatenate((positive_scans[70:], negative_scans[70:]), axis=0)
#y_test = np.concatenate((positive_labels[70:], negative_labels[70:]), axis=0)
#y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
#y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

x = np.concatenate((positive_scans, negative_scans), axis=0)
y = np.concatenate((positive_labels, negative_labels), axis=0)
x, y = shuffle(x, y, random_state=0)
print("HHH",x.shape[0], y.shape[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print("Number of scans in test, train with 75/25 split: ", x_test.shape[0],",", x_train.shape[0])
print(x_test.shape[0])
print(x_train.shape[0])
print(y_test.shape[0])
print(y_train.shape[0])

del(positive_scans)
del(positive_labels)
del(negative_scans)
del(negative_labels)
del(x)
del(y)
gc.collect()
#############################
# Data augmentation section #
#############################

#define rotation angles
#pick angle at random
#rotate image
def rotate(volume):
    def scipy_rotate(volume):
        angle = [-20,-17.5,-15,-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10,12.5,15,17.5,20]
        angle = random.choice(angle)
        volume = ndimage.rotate(volume,angle,reshape=False)
        if volume < 0:
          volume = 0
        elif volume > 1:
          volume = 1
        return volume
    rotated_volume = tf.numpy_function(scipy_rotate,[volume],tf.float32)
    return rotated_volume

rotate = tf.function(rotate)

#process training data by rotating and adding a channel
def train_process(volume,label):
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

#Process testing data by adding a channel
def test_process(volume,label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#define data loaders
train_loader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_loader = tf.data.Dataset.from_tensor_slices((x_test,y_test))

#defines the batch size (test this and change until optimised for speed, accuracy, lower=less cpu,accuracy) power of 2
batch_size = 2

#augment the data during training
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_process)
    .batch(batch_size)
    .prefetch(4)
)

#Rescale test data
test_dataset = (
    test_loader.shuffle(len(x_test))
    .map(test_process)
    .batch(batch_size)
    .prefetch(4)
)


#############################
# DEFINE THE NEURAL NETWORK #
#############################

#come back to optimise
#**
def get_model(width=128, height=128, depth=32):
    #Define a 3D Conv Neural Network
    inputs = keras.Input((width,height,depth,1))

    x=layers.Conv3D(filters=64, kernel_size=(5,5,5), padding='same', activation="relu")(inputs)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    x=layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    x=layers.Conv3D(filters=128, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    x=layers.Conv3D(filters=256, kernel_size=(3,3,3), padding='same', activation="relu")(x)
    x=layers.MaxPool3D(pool_size=2)(x)
    x=layers.BatchNormalization()(x)

    #x=layers.Conv3D(filters=512, kernel_size=7, padding='same', activation="relu")(x)
    #x=layers.MaxPool3D(pool_size=2)(x)
    #x=layers.BatchNormalization()(x)

    x=layers.GlobalAveragePooling3D()(x)
    x=layers.Dense(units=512,activation="relu")(x)
    x=layers.Dropout(0.5)(x)

    x = layers.Dense(3, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs,outputs, name="3dcnn")
    return model

#Build the Model
model=get_model(width=128,height=128,depth=32)
model.summary()

initial_learning_rate = 0.00001
learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=725, decay_rate=0.96, staircase=True)
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=learning_rate_schedule),metrics=[keras.metrics.BinaryAccuracy()])
#metrics=["acc"])
#keras.losses.BinaryCrossentropy()
#define callbacks
checkpoint = keras.callbacks.ModelCheckpoint("image_class_model.keras", save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc",patience=25)

#train model, testing after each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint, early_stopping],

)
"""ex
#Visualize model accuracy
fig,ax = plt.subplots(1,2, figsize=(25,5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
"""