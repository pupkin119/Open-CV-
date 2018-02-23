import cv2
import numpy as np
img = cv2.imread('111.jpeg', 0)
img1 = cv2.imread('222.jpeg',0)
#a = [5,5,5]
#c = [5,5,5]
#ac = np.correlate(a,c,'full')
global vecx,vecy,vecx2,vecx3,vecy2,vecy3,d,ii,jj
def number(arg):
    if arg == 0:
        vecx = -4
        vecy = -4
    if arg == 1:
        vecx = 0
        vecy = -4
    if arg == 2:
        vecx = 4
        vecy = -4
    if arg == 3:
        vecx = -4
        vecy = 0
    if arg == 4:
        vecx = 0
        vecy = 0
    if arg == 5:
        vecx = 4
        vecy = 0
    if arg == 6:
        vecx = -4
        vecy = 4
    if arg == 7:
        vecx = 0
        vecy = 4
    if arg == 8:
        vecx = 4
        vecy = 4
    return vecx,vecy

arg = np.zeros(9)
mass = np.zeros(64)
macc = np.zeros(64)
rez = np.zeros([img.shape[0],img.shape[1]])
arrowx = np.zeros([img.shape[1],img.shape[0]])
arrowy = np.zeros([img.shape[1],img.shape[0]])

def get_corr(x,y,vecx,vecy,vecx2,vecy2,ii,jj): # x,y - 0:0 pixel of squre 
    mass = np.zeros(64)
    macc = np.zeros(64)
    i=0
    for xx in np.arange(0, 8, 1):
        for yy in np.arange(0, 8, 1):
            mass[i] = img[y+yy,x+xx]
            macc[i] = img1[y+yy+vecy+vecy2+jj, x+xx+vecx+vecx2+ii]
            i = i+1
            #if i >= 64:# 0:63 -> 64
            #   i = 0
    corr = max(np.correlate(mass, macc, 'full'))
    return corr

#cv2.arrowedLine(dilation, (0,0), (14,14), (255,255,255), 1)
f = 20
for x in np.arange(f,img.shape[1]+1-f,10): #500
    for y in np.arange(f,img.shape[0]+1-f,10): #800
        d = 4
        vecx,vecy,vecx2,vecx3,vecy2,vecy3 = 0,0,0,0,0,0
        for m in np.arange(1,2): # iteration of 3SS
            j=0
            for ii in np.arange(-d,d+1,d):
                for jj in np.arange(-d,d+1,d):
                    arg[j] = get_corr(x,y,vecx,vecy,vecx2,vecy2,ii,jj)
                    j=j+1              
            vecx,vecy = number(arg.argmax(axis=0))
            if m == 2:
                vecx2,vecy2 = number(arg.argmax(axis=0))
                vecx2 = int(vecx2/2)
                vecy2 = int(vecy2/2)
            if m == 3:
                vecx3,vecy3 = number(arg.argmax(axis=0))
                vecx3 = int(vecx3/4)
                vecy3 = int(vecy3/4)
            d = d/2
            arrowx[x,y] = (x+vecx+vecx2+vecx3)
            arrowy[x,y] = (y+vecy+vecy2+vecy3)
for x in np.arange(f,img.shape[1]+1-f,10): #500
    for y in np.arange(f,img.shape[0]+1-f,10): #800
        cv2.arrowedLine(rez, (x,y), (int(arrowx[x,y]),int(arrowy[x,y])), (255,255,255), 1)
print(img.shape[1]) # ширина x 800
print(img.shape[0]) # высота y 500
cv2.imshow('original', img1)
cv2.imshow('vectors', rez)
cv2.waitKey(0)
cv2.destroyAllWindows()

