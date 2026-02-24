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
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


#testing file for either ct scans or x-rays, just change paths

negative_scans = np.load("filepath to negative scans numpy array")
positive_scans = np.load("filepath to positive scans numpy array")
negative_labels = np.load("filepath to negative labels")
positive_labels = np.load("filepath to positive labels")
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
del(x_train)
del(y_train)
del(x)
del(y)
gc.collect()

model = load_model("filepath to trained model")

def ttest(x,y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    prob = model.predict(x, batch_size = 2)
    prob_classes = (model.predict(x, batch_size =2) > 0.6).astype("int32")
    prob = prob[:,0]
    prob_classes = prob_classes[:,0]

    accuracy = accuracy_score(y,prob_classes)
    precision = precision_score(y,prob_classes)
    f1 = f1_score(y,prob_classes)
    recall = recall_score(y,prob_classes)

    print("Acc:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("f1:",f1)
    for i in range(len(y)):
        if y[i] == 1 and prob_classes[i] == 1:
            tp += 1
        elif y[i] == 0 and prob_classes[i] == 0:
            tn += 1
        elif y[i] == 0 and prob_classes[i] == 1:
            fp += 1
        elif y[i] == 1 and prob_classes[i] == 0:
            fn += 1
        else:
            return "ERR"
    print("TP:",tp,
          "\nTN:",tn,
          "\nFP:",fp,
          "\nFN:",fn)

ttest(x_test,y_test)
