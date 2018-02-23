import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
image1 = cv2.imread('img11.png',0)
image2 = cv2.imread('img22.png',0)
#x=0.25
#image1 = cv2.resize(image1,None,fx=x, fy=x, interpolation = cv2.INTER_CUBIC)
#image2 = cv2.resize(image2,None,fx=x, fy=x, interpolation = cv2.INTER_CUBIC)
#контурный препарат
m = 3
image1 = cv2.GaussianBlur(image1,(m,m),-1)
image1 = cv2.absdiff(image2,image1)
tres,image1 = cv2.threshold(image1, 50, 255, cv2.THRESH_BINARY)
# фильтрация морфологическая по элементу квадратному
n = 3
kernel = np.ones((n,n),np.uint8)
erosion = cv2.erode(image1,kernel,iterations = 1)
#erosiom = cv2.erode()
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
dilation = cv2.dilate(opening,kernel,iterations = 3)
#cv2.circle(img,(447,63), 63, (0,0,255), -1)
cv2.arrowedLine(dilation, (0,0), (14,14), (255,255,255), 1)
cv2.imshow('img11',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
