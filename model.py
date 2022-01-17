import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image,ImageOps

X,y= fetch_openml("mnist_784",version=1, return_X_y=True)
print(pd.Series(y).value_counts())
X= np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

def getPred(img):

    pilImage=Image.open(img)

    image_bw=pilImage.convert('L')
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)

    image_bw_resized_inverted= ImageOps.invert(image_bw_resized)

    pxlFil=20
    minPxl=np.percentile(image_bw_resized_inverted,pxlFil)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-minPxl,0,255)

    maxPxl=np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxPxl

    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_predict=clf.predict(test_sample) 
    return test_predict[0]

    
