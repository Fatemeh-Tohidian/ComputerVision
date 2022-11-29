import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('01.JPG')        
img2 = cv.imread('02.JPG')

def draw_line(img,a,b,c,color):
    x1 = 0
    y1 = int(-c/b)
    x2 = int(img.shape[1])
    y2 = int((-c - x2*(a))/b)
    img = cv.line(img,(x1,y1),(x2,y2),color,10)
    return img

def drawlines(img1,img2,lines1,lines2,points1,points2):
    for i in range(0,10):
        l1 = lines1[i]
        l2 = lines2[i]
        p1 = points1[i]
        p2 = points2[i]
        a1,b1,c1 = l1
        a2,b2,c2 = l2
        color = tuple(np.random.randint(0,255,3).tolist())
        img2 = draw_line(img2,a1,b1,c1,color)
        img1 = draw_line(img1,a2,b2,c2,color)
        cv.circle(img1, (p1[0],p1[1]), 30, color ,thickness=-1)
        cv.circle(img2, (p2[0],p2[1]), 30, color ,thickness=-1)
    return img1,img2

sift = cv.SIFT_create()
key_points1, descriptors1 = sift.detectAndCompute(img1,None)
key_points2, descriptors2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2, k=2)
points1 = []
points2 = []
good_key_points1 = []
good_key_points2 = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        points1.append(np.float32(key_points1[m.queryIdx].pt))
        good_key_points1.append(key_points1[m.queryIdx])
        points2.append(np.float32(key_points2[m.trainIdx].pt))
        good_key_points2.append(key_points2[m.trainIdx])

points1 = np.asarray(points1)
points2 = np.asarray(points2)
good_key_points1 = np.asarray(good_key_points1).reshape((-1,1))
good_key_points2 = np.asarray(good_key_points2).reshape((-1,1))
F, mask = cv.findFundamentalMat(points1,points2,cv.FM_RANSAC)
print(F)
c_img1 = np.copy(img1)
c_img2 = np.copy(img2)
c_img1 = cv.drawKeypoints(c_img1,good_key_points1[mask==0],c_img1, color=(0,0,255))
c_img1 = cv.drawKeypoints(c_img1,good_key_points1[mask==1],c_img1, color=(0,255,0))
c_img2 = cv.drawKeypoints(c_img2,good_key_points2[mask==0],c_img2, color=(0,0,255))
c_img2 = cv.drawKeypoints(c_img2,good_key_points2[mask==1],c_img2, color=(0,255,0))
img = np.hstack((c_img1, c_img2))
cv.imwrite('res05.jpg‬‬',img)

u, s, vh = np.linalg.svd(F, full_matrices=True)
e = vh[2,:]/vh[2,2]
u, s, vh = np.linalg.svd(F.T, full_matrices=True)
ee = vh[2,:]/vh[2,2]

h = 1000
w = 4100
A = 2100
B = 300

print(e)
print(ee)

e=(e[0:2]/10)+[A,B]
ee=(ee[0:2]/10)+[A,B]

big_img1 = np.ones((h,w,3),np.uint8)*255
big_img2 = np.ones((h,w,3),np.uint8)*255
img_half1 = cv.resize(img1, (0,0), fx=0.1, fy=0.1)
img_half2 = cv.resize(img2, (0,0), fx=0.1, fy=0.1)
big_img1[B:B+img_half1.shape[0],A:A+img_half1.shape[1]] = img_half1
cv.circle(big_img1, (int(e[0]),int(e[1])), 10, color = (0, 0, 0),thickness=-1)
big_img2[B:B+img_half2.shape[0],A:A+img_half2.shape[1]] = img_half2
cv.circle(big_img2, (int(ee[0]),int(ee[1])), 10, color = (0, 0, 0),thickness=-1)
cv.imwrite('res06.jpg‬‬',big_img1)
cv.imwrite('res07.jpg‬‬',big_img2)

one = np.ones((points2[mask.ravel()==1].shape[0],1))
h_points1 = np.hstack((points1[mask.ravel()==1],one))
h_points2 = np.hstack((points2[mask.ravel()==1],one))
lines1 = np.matmul(F,h_points1.T).T
lines2 = np.matmul(F.T,h_points2.T).T
img1,img2 = drawlines(img1,img2,lines1,lines2,points1[mask.ravel()==1],points2[mask.ravel()==1])


plt.subplot(121),plt.imshow(img1)
plt.subplot(122),plt.imshow(img2)
plt.savefig('res08.jpg')