# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import pandas as pd

print("[INFO] loading network...")
model = load_model("/content/model")
mlb = pickle.loads(open("/content/tags", "rb").read())

cs=pd.read_csv('/content/drive/My Drive/recognizance/recognizance1/sample.csv')
# labels=list(cs['label'])
image_name=list(cs['image'])

data = []

for i in range(len(image_name)):
    path='/content/drive/My Drive/recognizance/recognizance1/test/'+image_name[i]
    print(path)
    try:
        image=cv2.imread(path)
        image = cv2.resize(image, (IMAGE_DIMS[0],IMAGE_DIMS[1]))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        proba = model.predict(image)
        # print(proba)
        ids=np.argmax(proba,axis=1)
        print(ids)
        cs['label'][i]=ids
        # data.append(image)
    except:
      continue
cs.to_csv('pred_class.csv',index=False)
