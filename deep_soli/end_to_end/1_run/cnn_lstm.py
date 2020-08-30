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

imagepaths = []
for root,dirs,files in os.walk("/home/ee/mtech/eet192341/data/h5_train",topdown=False):
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

x=X.reshape(X.shape[0],X.shape[1],32,32,1)

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


model=tf.keras.Sequential()
model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu',input_shape=(100,32,32,1))))
model.add(TimeDistributed(Conv2D(64,(3,3),activation='relu')))
model.add(TimeDistributed(Conv2D(128,(3,3),activation='relu')))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512,activation='relu')))
model.add(TimeDistributed(Dense(512,activation='relu')))
model.add(LSTM(512))
model.add(Dense(12,activation='softmax'))

checkpoint_cb=tf.keras.callbacks.ModelCheckpoint("/home/ee/mtech/eet192341/codes/deep_soli/end_to_end/1_run/model_cnn_lstm.h5",save_best_only=True)


optimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics=["sparse_categorical_accuracy"])


history = model.fit(X_train,y_train,verbose=2,batch_size=16, epochs=150,validation_data=(X_val,y_val),callbacks=[checkpoint_cb])

pd.DataFrame(history.history).plot(figsize=(16,10))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig("/home/ee/mtech/eet192341/codes/deep_soli/end_to_end/1_run/LA_curve_ete.png")
print("finished")
