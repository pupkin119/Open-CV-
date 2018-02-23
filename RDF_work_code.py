import pandas as pd
import numpy as np
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
def get_pixel_sum(x, y, img):
    A = np.zeros(3,dtype = int)
    for i in np.arange(-3,4):
        for j in np.arange(-3,4):
            A[0] = A[0] + img[y+j, x+i][0]
            A[1] = A[1] + img[y+j, x+i][1]
            A[2] = A[2] + img[y+j, x+i][2]
    A = A/(7**2)
    return A
df1 = pd.read_csv('example1.csv')
df0 = pd.read_csv('example0.csv')
df0.drop(['Unnamed: 0','R/G','R/B'], axis=1, inplace=True)
df1.drop(['Unnamed: 0','R/G','R/B'], axis=1, inplace=True)
df0 = df0.append(df1, ignore_index=True)
df0.drop(['B','L','B.1','Y','H','V'], axis=1, inplace=True)

y = df0['pat']
X = df0.drop(['pat'],axis = 1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
RF = RandomForestClassifier(n_estimators=300, random_state=42)
RF.fit(X_train, y_train)
m = RF.predict(df0.drop(['pat'],axis = 1))
print(m)
# ['R', 'G', 'A', 'Cr', 'Cb', 'S', 'pat']
# RDF to image dataset test
img = cv2.imread('dataset1.jpg')
rez = cv2.imread('dataset1.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #lab
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # Ycrcb
f = 4
step = 10
for x in np.arange(f,img.shape[1]-f,step): 
    for y in np.arange(f,img.shape[0]-f,step): 
        BGR = get_pixel_sum(x,y,img)
        HSV = get_pixel_sum(x,y,img1)
        LAB = get_pixel_sum(x,y,img2)
        YCRCB = get_pixel_sum(x,y,img3)
        df_new = pd.DataFrame({'R':[BGR[2]], 'G':[BGR[1]], 
                    'A':[LAB[1]], 'Cr':[YCRCB[1]],'Cb':[YCRCB[2]],
                   'S':[HSV[1]]},
                   columns = ['R', 'G', 'A', 'Cr', 'Cb', 'S'])
        #df = df.append(df_new, ignore_index=True)
        predict_color = RF.predict(df_new)
        if predict_color == 1:
            rez[y, x] = np.array([0, 0, 255])
        else:
            rez[y, x] = np.array([0, 255, 0])

cv2.imshow('Rezult', rez)
cv2.waitKey(0)
cv2.destroyAllWindows()
