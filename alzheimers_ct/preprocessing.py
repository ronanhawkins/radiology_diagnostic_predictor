import os
import numpy as np
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.layers import BatchNormalization
import nibabel
from scipy import ndimage
#from sklearn.model_selection import train_test_split
#import random
import matplotlib.pyplot as plt
import os.path
import time
import gc
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from skimage.filters import median, gaussian
from skimage.morphology import disk

"""
negative_dataset = "filepath to folder with negative scans (nifti images)"
positive_dataset  = "filepath to folder with positive scans (nifti images)"
"""

########################################
#function to read files and get raw data
def read_file(fp):
    raw = nibabel.load(fp)
    raw = raw.get_fdata()
    raw = raw.astype("float32")
    gc.collect()
    return raw

#temporary normal
def normal(volume):
    min = 0
    max = 80
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

#dimension resize
def dimensions_resize(volume):
    if volume.ndim == 4:
        volume = volume[:,:,:,0]
    else:
        pass
    required_x = 128
    required_y = 128
    required_z = 18
    current_x = volume.shape[0]
    current_y = volume.shape[1]
    current_z = volume.shape[2]

    x_factor = 1 / (current_x / required_x)
    y_factor = 1 / (current_y / required_y)
    z_factor = 1 / (current_z / required_z)

    print("Original Shape: ", volume.shape)
    volume = ndimage.zoom(volume, (x_factor,y_factor,z_factor),order=1)
    print("Zoomed shape", volume.shape)
    return volume

def process(filepath):
    print(filepath)
    volume = dimensions_resize(normal(read_file(filepath)))
    return volume

"""
positive_scans = [
    os.path.join(positive_dataset, x)
    for x in os.listdir(positive_dataset)
]

negative_scans = [
    os.path.join(negative_dataset, x)
    for x in os.listdir(negative_dataset)
]

print("Number of Positive Scans: ", str(len(positive_scans)))
print("Number of Negative Scans: ", str(len(negative_scans)))
"""
"""
positive_scans = np.array([process(filepath) for filepath in positive_scans])
negative_scans = np.array([process(filepath) for filepath in negative_scans])
"""
"""
#assign labels, 0 for positive, 1 for negative
positive_labels = np.array([1 for _ in range(len(positive_scans))])
negative_labels = np.array([0 for _ in range(len(negative_scans))])
"""
"""
np.save("data_arrays/positive_scans", positive_scans)
np.save("data_arrays/negative_scans", negative_scans)
"""
"""
np.save("new_data_arrays/positive_labels", positive_labels)
np.save("new_data_arrays/negative_labels", negative_labels)
"""

"""
x = positive_scans
print(read_file(x[115]))
volume = median(process(x[112]), footprint=disk(4))

#volume = np.clip(read_file(x[115]), 0, 100)
print(volume)
for i in range(0, 74, 1):  # Show every 10th slice
    plt.imshow(volume[:,:,i], cmap='gray',vmin=0,vmax=1)
    plt.title(f'Slice {i}')
    plt.axis('off')
    plt.show()

"""
