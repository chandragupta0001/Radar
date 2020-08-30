print("loading module")
import os
import random 
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
print("loading images")
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

print("train data: ")
train_x,train_y=load_data(imagepaths[:4000])
print(train_x.shape)
print("test data: ")

test_x,test_y=load_data(imagepaths[4000:4400])
print(test_x.shape)
val_x,val_y=load_data(imagepaths[4400:])

model= keras.models.Sequential([
 keras.layers.Dense(512,activation='relu'),
 keras.layers.LSTM(512),
 keras.layers.Dense(12,activation='softmax')])
checkpoint_cb=tf.keras.callbacks.ModelCheckpoint("/home/ee/mtech/eet192341/codes/rnn_deep/rnn_deep.h5",save_best_only=True)
optimizer = keras.optimizers.Adam(lr=0.002)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics=["sparse_categorical_accuracy"])
history = model.fit(train_x,train_y,verbose=2, epochs=200,validation_data=(val_x,val_y),callbacks=[checkpoint_cb])

pd.DataFrame(history.history).plot(figsize=(16,10))
plt.grid(True)
#plt.show()
plt.savefig("/home/ee/mtech/eet192341/codes/rnn_deep/rnn_deep_loss_acc.png")

e=model.evaluate(test_x,test_y)
print(e)
