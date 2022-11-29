import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
img1 = cv2.imread('im03.jpg')        
img2 = cv2.imread('im04.jpg')
sift = cv2.SIFT_create()
key_points1, descriptors1 = sift.detectAndCompute(img1,None)
key_points2, descriptors2 = sift.detectAndCompute(img2,None)
k_img1 = np.copy(img1)
k_img2 = np.copy(img2)
k_img1 = cv2.drawKeypoints(img1,key_points1,k_img1, color=(0,255,0))
k_img2 = cv2.drawKeypoints(img2,key_points2,k_img2, color=(0,255,0))
new_img2 = np.zeros((img1.shape[0],img2.shape[1],3),dtype=np.uint8)
new_img2[:img2.shape[0],:] = k_img2
img = np.hstack((k_img1, new_img2))
cv2.imwrite('‫‪res13_corners.jpg‬‬',img)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
src_pts = []
dst_pts = []
good_key_points1=[]
good_key_points2=[]

for m in good:

    src_pts.append(np.float32(key_points1[m.queryIdx].pt))
    dst_pts.append(np.float32(key_points2[m.trainIdx].pt))
    good_key_points1.append(key_points1[m.queryIdx])
    good_key_points2.append(key_points2[m.queryIdx])
c_img1 = np.copy(k_img1)
c_img2 = np.copy(k_img2)
c_img1 = cv2.drawKeypoints(c_img1,good_key_points1,c_img1, color=(255,0,0))
c_img2 = cv2.drawKeypoints(c_img2,good_key_points2,c_img2, color=(255,0,0))
new_img2 = np.zeros((img1.shape[0],img2.shape[1],3),dtype=np.uint8)
new_img2[:img2.shape[0],:] = c_img2
img = np.hstack((c_img1, new_img2))
cv2.imwrite('‫‪res14_correspondences.jpg‬‬',img)
img = cv2.drawMatches(img1, key_points1, img2, key_points2, good, None, (255,0,0), (0,255,0),None,cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite('‫‪res15_matches.jpg‬‬‬‬',img)
good_20 = random.choices(good,k=20)
img = cv2.drawMatches(img1, key_points1, img2, key_points2, good_20, None, (255,0,0), (0,255,0),None,cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite('‫‪res16.jpg‬‬‬‬',img)
src_pts = np.asarray(src_pts)
dst_pts = np.asarray(dst_pts)
M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0,maxIters=1000)
img = cv2.drawMatches(img1, key_points1, img2, key_points2, good, None, (255,0,0), (0,255,0),None,cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img1_blue = img[:,:img1.shape[1],:]
img2_blue = img[:,img1.shape[1]:,:]
img = cv2.drawMatches(img1_blue, key_points1, img2_blue, key_points2, good, None, (0,0,255), (0,255,0),mask,cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite('‫‪res17.jpg‬‬‬‬',img)
ans = cv2.warpPerspective(img2, M, (2000,2000))
cv2.imwrite('res19.jpg' , ans)
