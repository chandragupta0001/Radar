print("Hold! going deep.....")
import cv2
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

imagepaths = []
for root,dirs,files in os.walk("/home/ee/mtech/eet192341/data/train",topdown=False):     # system dependent
    for name in files:
        path = os.path.join(root,name)
        if path.endswith("jpg"):
            imagepaths.append(path)

print(len(imagepaths))
X = []
y = []


for path in imagepaths:
    img = cv2.imread(path)
    #img= img.astype("uint8")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    X.append(img)
    category = path.split("/")[7]                                                                      # system dependent
    cat=category.split("_")
    label = int(cat[0])
    y.append(label)


X = np.array(X,dtype="uint8")

X = X.reshape(len(imagepaths),32,32,1)
y = np.array(y)

print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))

X_train, X_val,y_train,y_val = train_test_split(X,y,test_size=0.05, random_state=42)
# to look for the distribution of classes in full dataset , train dataset and val data set
unique, counts = np.unique(y, return_counts=True)
print("full data distribution",dict(zip(unique, counts)))

unique, counts = np.unique(y_train, return_counts=True)
print("train data distribution",dict(zip(unique, counts)))

unique, counts = np.unique(y_val, return_counts=True)
print("validation_data distribution",dict(zip(unique, counts)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
# network from deepsoli paper tabel 1 colmn 3; CNN with pooling
model = Sequential()
model.add(Conv2D(96,(7,7),activation='relu',input_shape=(32,32,1)))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='valid'))
model.add(Conv2D(256,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='valid'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='valid'))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dense(2048,activation='relu'))
model.add(Dense(11,activation='softmax'))

checkpoint_cb=tf.keras.callbacks.ModelCheckpoint("/home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/cnn_deep_model_2.h5",save_best_only=True)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=80, batch_size= 15, verbose = 2, validation_data = (X_val, y_val),callbacks=[checkpoint_cb])

pd.DataFrame(history.history).plot(figsize=(16,10))
plt.grid(True)
plt.gca().set_ylim(0,4)
#plt.show()
plt.savefig("/home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/cnn_deep_loss_acc_curve.png")
