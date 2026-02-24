from keras.models import load_model
from pneumonia_xray.preprocessing import process as twod_process
from alzheimers_ct.preprocessing import process as threed_process
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


type = input("Would you like to make a prediction on:\n (P) An X-ray for Pneumonia \n (A) A CT for Alzheimer's\nPlease enter a capital letter (P/A): ")
if type == "P":
    model = load_model("/Users/ronanhawkins/Desktop/coding/btyse25/pneumonia_xray/saved_models/n.keras")
    #chest_xray_classificatio
    uinput = input("Please input filepath to image for prediction: ")
    visinp = input("Would you like to visualise the image? (Y/N): ")
    pred = str(model.predict(np.expand_dims(twod_process(uinput), axis=0))[0])
    image = Image.open(uinput)
    image.show()

elif type == "A":
    model = load_model("/Users/ronanhawkins/Desktop/coding/btyse25/alzheimers_ct/saved_models/image_class_model.keras")
    uinput = input("Please input filepath to image for prediction: ")
    visinp = input("Would you like to visualise the image? (Y/N): ")
    pred = str(model.predict(np.expand_dims(threed_process(uinput), axis=0))[0])
    plt.imshow((threed_process(uinput))[:,:,9], interpolation='nearest',cmap='gray')
    plt.show()




#uinput = "/content/drive/MyDrive/abtyse_exhibit/p0sub-OAS30007_sess-d1641_CT.nii" po
#uinput = "/content/drive/MyDrive/sub-OAS31193_sess-d0166_run-01_CT.nii" neg
pred=pred.replace("[","")
pred=pred.replace("]","")
pred = float(pred)

print(pred)
if pred < .5:
  print("The model predicts that the image is", str(100-round((pred*100),2)) + "%", "likely to be negative")
else:
  print("The model predicts that the image is", str(round((pred*100),2)) + "%", "likely to be positive")