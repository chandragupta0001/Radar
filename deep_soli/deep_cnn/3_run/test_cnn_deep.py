import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


imagepaths = []
for root,dirs,files in os.walk("/home/ee/mtech/eet192341/data/test",topdown=False):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith("jpg"):
            imagepaths.append(path)
            
print("total imagepaths: ",len(imagepaths))

X = []
y = []

for path in imagepaths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    X.append(img)
    category = path.split("/")[7]
    cat=category.split("_")
    label = int(cat[0])
    y.append(label)
    
    
X = np.array(X,dtype="float16")
y = np.array(y)
X = X.reshape(len(imagepaths),32,32,1)

model = tf.keras.models.load_model('/home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/cnn_deep_model_2.h5')
model.summary()

evu=model.evaluate(X,y)
print("Evalution",evu)

y_pred=model.predict_classes(X)
c=confusion_matrix(y,y_pred)
label_name=["pinch index","palm tilt","finger slide","pinch pinky","slow swip","fast swip","push","pull","finger rub","circle","hold"]
fig = plt.figure(figsize=[16,16])
import seaborn as sns
sns.heatmap(c, annot=True,annot_kws={"size": 13},xticklabels=label_name,yticklabels=label_name)
plt.savefig("/home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/confusion_matrix_deep_cnn_run_1_frame_level.png")

from scipy.stats import mode

y_true=[]
y_pred=[]
for root,dirs,files in os.walk("/home/ee/mtech/eet192341/data/test",topdown=True):
#     print("*******************************************************************************************")
    testimage = []
    for name in files:
        path = os.path.join(root,name)
#         print(path)
        if path.endswith("jpg"):
            testimage.append(path)
    X_test=[]
    y_test=[]
    for p in testimage:
        img = cv2.imread(p)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        X_test.append(img)
        category = p.split("/")[7]
        
        cat=category.split("_")
        label = int(cat[0])
        y_test.append(label)
#     print(X_test)
    if(X_test==[]):
        continue
#     plt.imshow(X_test[0])
    X_test=np.array(X_test,dtype="float16")
    X_test=X_test.reshape(len(testimage),32,32,1)
    y=model.predict_classes(X_test)
    y_pred.append(mode(y).mode[0])
    y_true.append(y_test[0])
    unique, counts = np.unique(y, return_counts=True)
   # print("dist ",dict(zip(unique,counts))," prdic ",mode(y).mode[0]," y_actual ",y_test[0]," folder ",category)
       


y_true=np.array(y_true)
y_pred=np.array(y_pred)

c_new=confusion_matrix(y_true,y_pred)
fig = plt.figure(figsize=[16,16])
import seaborn as sns
sns.heatmap(c_new, annot=True,annot_kws={"size": 13},xticklabels=label_name,yticklabels=label_name)
plt.savefig("/home/ee/mtech/eet192341/codes/deep_soli/deep_cnn/3_run/confusion_matrix_batch_cnn_deep_run1.png")

acc=np.sum((y_true==y_pred)*1)/len(y_true)
print("Batch acc+",acc)     

