import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
gestures = ["PinchIndex", "PinchPinky", "FingerSlider", "FingerRub","SlowSwipeRL", "FastSwipeRL", "Push", "Pull", "PalmTilt","Circle", "PalmHold"]

def load_data(datasetPath):
   
    persons=25
    people = list(range(1,persons,1))
    rows=(persons-1)*11*105
    X=np.empty((rows,5,32,492,2),dtype=np.float32)
    y=np.empty((rows,5,1),int)
    i=0
    temp=0
    for gdx,gestureName in enumerate(gestures):
        
         for pdx, person in enumerate(people):
                path=datasetPath+"/"+"p"+str(person)+"/"+gestureName+"_"+"1s_"+"wl32_doppl.npy"
                # print(path)
                x=np.load(path)
                x_len=x.shape[0]
                temp=x_len
                X[i:i+x_len]=x
                y[i:i+x_len]=np.zeros((5,1))+gdx
                i=i+x_len
                      
    return X[:i-temp],y[:i-temp]
win=5
X,y=load_data("/scratch/ee/mtech/eet192341/tinyradar/11G_feat")
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
del(X)
del(y)

from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Add
from tensorflow.keras.layers import TimeDistributed
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/ee/mtech/eet192341/tinyradar/doppler/')

from tcn import TCN
from tensorflow.keras import Input, Model,optimizers
i=Input(shape=(win,32,492,2))
o=TimeDistributed(Conv2D(16,(3,5),padding='same',activation='relu'))(i)
o=TimeDistributed(MaxPooling2D(pool_size=(3,5),padding='valid'))(o)
o=TimeDistributed(Conv2D(32,(3,5),padding='same',activation='relu'))(o)
o=TimeDistributed(MaxPooling2D(pool_size=(3,5),padding='valid'))(o)
o=TimeDistributed(Conv2D(64,(1,7),padding='same',activation='relu'))(o)
o=TimeDistributed(MaxPooling2D(pool_size=(1,7),padding='valid'))(o)
o=TimeDistributed(Flatten())(o)
a=Conv1D(32,1,padding="causal")(o)
b=Conv1D(32,2,padding="causal", dilation_rate=1)(a)
added1=Add()([a,b])
c=Conv1D(32,2,padding="causal",dilation_rate=2)(added1)
added2=Add()([added1,c])
d=Conv1D(32,2,padding="causal",dilation_rate=4)(added2)
o=TimeDistributed(Dense(64,activation='relu'))(d)
o=TimeDistributed(Dense(32,activation='relu'))(o)
o=TimeDistributed(Dense(11,activation='softmax'))(o)


# o = TCN(return_sequences=False)(o) # The TCN layers are here.
# o = Dense(13,activation='softmax')(o)

m = Model(inputs=[i], outputs=[o])
optimizer =optimizers.Adam(lr=0.001)
m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['acc'])
m.summary()
history=m.fit(X_train, y_train,verbose=1,batch_size=64, epochs=100, validation_split=0.2)

pd.DataFrame(history.history).plot(figsize=(16,10))
plt.grid(True)
#plt.show()
plt.savefig("/home/ee/mtech/eet192341/tinyradar/doppler/cnn_tcn/2_model/rerun/loss_acc.png")
m.evaluate(X_test,y_test,verbose=1)
from sklearn.metrics import confusion_matrix
import time
t1=time.time()
y_pred=m.predict(X_test)
t2=time.time()
print(y_pred.shape,"time taken:",t2-t1)
y_pred=np.argmax(y_pred,axis=2)
print(y_test.shape,y_pred.shape)
c=confusion_matrix(y_test[:,4,0],y_pred[:,4])
label_name=gestures
fig = plt.figure(figsize=[22,18])
import seaborn as sns
sns.heatmap(c, annot=True,annot_kws={"size": 13},xticklabels=label_name,yticklabels=label_name)
plt.savefig("/home/ee/mtech/eet192341/tinyradar/doppler/cnn_tcn/2_model/rerun/confusion.png")
