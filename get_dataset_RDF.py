import cv2
import numpy as np
import pandas as pd

# create data prototype

df = pd.DataFrame(columns = ['R', 'G', 'B', 'L', 'A', 'B', 'R/G', 'R/B', 'Y', 'Cr', 'Cb', 'H', 'S', 'V','pat'])
# mouse callback function
def get_pixel_sum(x, y, img):
    A = np.zeros(3,dtype = int)
    for i in np.arange(-3,4):
        for j in np.arange(-3,4):
            A[0] = A[0] + img[y+j, x+i][0]
            A[1] = A[1] + img[y+j, x+i][1]
            A[2] = A[2] + img[y+j, x+i][2]
    A = A/(7**2)
    return A
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global df
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        #bgr
        #np.array([1,1,1]) to addition
##        BGR = img[y, x] # bgr format!
        BGR = get_pixel_sum(x,y,img)
        print (BGR)
##        HSV = img1[y, x]
        HSV = get_pixel_sum(x,y,img1)
        print (HSV)
##        LAB = img2[y, x]
        LAB = get_pixel_sum(x,y,img2)
        print (LAB)
##        YCRCB = img3[y, x]
        YCRCB = get_pixel_sum(x,y,img3)
        print (YCRCB)
##        RG=BGR[2]/BGR[1]
##        RB=BGR[2]/BGR[0]
        RG = 0
        RB = 0
        print (RG, RB)

        pat = 1 # pathology: 0 - good tissue, 1 - bad tissue
        
        df3 = pd.DataFrame({'R':[BGR[2]], 'G':[BGR[1]], 'B':[BGR[0]], 
                    'L':[LAB[0]], 'A':[LAB[1]], 'B':[LAB[2]],
                    'R/G':[RG], 'R/B':[RB],
                    'Y':[YCRCB[0]], 'Cr':[YCRCB[1]],'Cb':[YCRCB[2]],
                    'H':[HSV[0]], 'S':[HSV[1]], 'V':[HSV[2]],'pat':[pat]},
                   columns = ['R', 'G', 'B', 'L', 'A', 'B', 'R/G',
                              'R/B', 'Y', 'Cr', 'Cb', 'H', 'S', 'V', 'pat'])
        df = df.append(df3, ignore_index=True)
# Create a black image, a window and bind the function to window

#img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('dataset1.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #lab
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # Ycrcb

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
df.to_csv('example111_1.csv')
cv2.destroyAllWindows()
