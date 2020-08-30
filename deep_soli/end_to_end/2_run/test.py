import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import TimeDistributed
import random
from sklearn.metrics import confusion_matrix

imagepaths = []
for root,dirs,files in os.walk("/home/ee/mtech/eet192341/data/h5_test",topdown=False):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith("h5"):
            imagepaths.append(path)
            
print("Total files : ",len(imagepaths))
random.shuffle(imagepaths)

def load_data(paths_list):
    with h5py.File(paths_list[0], 'r') as f:
        data = f['ch{}'.format(0)][()]
        if(data.shape[0]<=100):
            data=np.pad(data,((0,100-data.shape[0]),(0,0)))
        else:
            data=data[:100]
#         data=data.reshape(100,32,32,1)
        y = f['label'][()]
        label=y[0]

    for hfile in paths_list:
        with h5py.File(hfile, 'r') as f:
            for channel in range(4):
                rdata = f['ch{}'.format(channel)][()]
                if(rdata.shape[0]<=100):
                    rdata=np.pad(rdata,((0,100-rdata.shape[0]),(0,0)))
                else:
                    rdata=rdata[:100]
                data=np.dstack((data,rdata))
                y = f['label'][()]
                label=np.concatenate((label,y[0]))
    data=np.swapaxes(data,1,2)
    data=np.swapaxes(data,0,1)
    return data,label


X,y=load_data(imagepaths)

X_test=X.reshape(X.shape[0],X.shape[1],32,32,1)


model = tf.keras.models.load_model('/home/ee/mtech/eet192341/codes/deep_soli/end_to_end/2_run/model_cnn_lstm.h5')
model.summary()

evu=model.evaluate(X_test,y)
print("Evalution",evu)

y_pred=model.predict_classes(X_test)
c=confusion_matrix(y,y_pred)
label_name=["pinch index","palm tilt","finger slide","pinch pinky","slow swip","fast swip","push","pull","finger rub","circle","hold","background"]

fig = plt.figure(figsize=[22,18])
import seaborn as sns
sns.heatmap(c, annot=True,annot_kws={"size": 13},xticklabels=label_name,yticklabels=label_name)
plt.savefig("/home/ee/mtech/eet192341/codes/deep_soli/end_to_end/2_run/confusion.png")
