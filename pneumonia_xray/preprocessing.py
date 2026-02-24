import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from numpy import asarray

w = 768
h = 512

def convert_to_numpy(filepath):
    print(filepath)
    image = Image.open(filepath)
    image = asarray(image)

    print(np.shape(image))
    return image

def image_resize(image):
    if image.ndim == 3:
        image = image[:,:,0]
    #uses inter area interpolation for resizing
    return cv2.resize(image, (w,h), interpolation=cv2.INTER_CUBIC)

def process(image):
    image = convert_to_numpy(image)
    return image_resize(image)

negative_scans = "filepath to normal"
positive_scans = "filepath to positive"
positive_scans = [
    os.path.join(positive_scans, x)
    for x in os.listdir(positive_scans)
]

negative_scans = [
    os.path.join(negative_scans, x)
    for x in os.listdir(negative_scans)
]


#positive_scans = np.array([process(filepath) for filepath in positive_scans])
#negative_scans = np.array([process(filepath) for filepath in negative_scans])
#convert to for loops; for fp in scans: process, add return values to new list; scans = n_list
cut_out_negative = []
cut_out_positive = []
negative_scans_list = []
positive_scans_list = []
l = []
for filepath in negative_scans:
    npy_image = convert_to_numpy(filepath)
    size = list(np.shape(npy_image))
    print(size[0])
    print(size[1])
    if (size[0] < h) or (size[1] < w) or ( (size[0] / size[1] > 1) or (size[0] / size[1] < 1/2)):
        cut_out_negative.append([size[0],size[1]])
    else:
        negative_scans_list.append(process(npy_image))
    l.append((size[0] / size[1]))

for filepath in positive_scans:
    npy_image = convert_to_numpy(filepath)
    size = list(np.shape(npy_image))
    print(size[0])
    print(size[1])
    if (size[0] < h) or (size[1] < w) or ( (size[0] / size[1] > 1) or (size[0] / size[1] < 1/2)):
        cut_out_positive.append([size[0],size[1]])
    else:
        positive_scans_list.append(process(npy_image))
    l.append((size[0] / size[1]))

positive_scans_list = np.array(positive_scans_list)
negative_scans_list = np.array(negative_scans_list)

print(len(positive_scans_list))
print(len(negative_scans_list))
print(len(cut_out_negative))
print(len(cut_out_positive))


#change scans to len new list
positive_labels = np.array([1 for _ in range(len(positive_scans_list))])
negative_labels = np.array([0 for _ in range(len(negative_scans_list))])

np.save("data_arrays/positive_scans", positive_scans_list)
np.save("data_arrays/negative_scans", negative_scans_list)
np.save("data_arrays/positive_labels", positive_labels)
np.save("data_arrays/negative_labels", negative_labels)

# setting the ranges and no. of intervals
range = (0, 3)
bins = 30

# plotting a histogram
plt.hist(l, bins, range, color = 'blue',
        histtype = 'bar', rwidth = 0.8)

# x-axis label
plt.xlabel('ratio x:y')
# frequency label
plt.ylabel('Number')
# plot title
plt.title('Image Size Ratios')

# function to show the plot
plt.show()

"""
x_norm_ct = 0
x_low_ct = 0
y_low_ct = 0
y_norm_ct = 0
count = 0
o_ct = 0
list = []
out = []
for filepath in negative_scans:
    size = process(filepath)
    
    if size[0] < 768:
        x_low_ct +=1
    elif size[0] >= 512:
        x_norm_ct +=1

    if size[1] < 256:
        y_low_ct +=1
    elif size[0] >= 256:
        y_norm_ct +=1
    
    if size[0] / size[1] > 1.0 and size[0] / size[1] < 2:
        count+=1
    else:
        o_ct+=1
    list.append((size[0] / size[1]))
    if (size[0] < 768) or (size[1] < 512) or (size[0] / size[1] < 1.0 and size[0] / size[1] > 2):
        out.append(filepath)
        

for filepath in positive_scans:
    size = process(filepath)
    
    if size[0] < 768:
        x_low_ct +=1
    elif size[0] >= 1024:
        x_norm_ct +=1

    if size[1] < 512:
        y_low_ct +=1
    elif size[0] >= 512:
        y_norm_ct +=1
        
    if size[0] / size[1] > 1.0 and size[0] / size[1] < 2:
        count+=1
    else:
        o_ct+=1
    list.append((size[0] / size[1]))
    if (size[0] < 768) or (size[1] < 512) or (size[0] / size[1] < 1.0 or size[0] / size[1] > 2):
        out.append(filepath)


print(x_low_ct)
print(x_norm_ct)
print(y_low_ct)
print(y_norm_ct)
print(count)
print(o_ct)

# setting the ranges and no. of intervals
range = (0, 3)
bins = 30

# plotting a histogram
plt.hist(list, bins, range, color = 'blue',
        histtype = 'bar', rwidth = 0.8)

# x-axis label
plt.xlabel('ratio x:y')
# frequency label
plt.ylabel('Number')
# plot title
plt.title('Image Size Ratios')

# function to show the plot
plt.show()

print(sum(list)/len(list))
print(len(out))
"""
