import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
def find_homographi(destination_sample,source_sample,count):
    A = np.zeros((2*count,9))
    for i in range(0,count):
        xx = source_sample[i][0]
        yy = source_sample[i][1]
        x = destination_sample[i][0]
        y = destination_sample[i][1]
        A[2*i] =   [ 0,  0,  0, -x, -y, -1, x*yy, y*yy, yy]
        A[(2*i)+1]=   [x, y, 1,  0,  0,  0, -x*xx, -y*xx, -xx]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    return np.reshape(vh[8,:],(3,3))

def findHomography(dst_pts,src_pts,threshold):
    random.seed(10)
    counter = 0
    n = len(dst_pts)
    N = 100000000000
    p = 0.99
    w = 0
    while  counter <= N:
        sample = random.sample(range(0, n), k=4)
        destination_sample = np.asarray([dst_pts[sample[0]],dst_pts[sample[1]],dst_pts[sample[2]],dst_pts[sample[3]]])
        source_sample = np.asarray([src_pts[sample[0]],src_pts[sample[1]],src_pts[sample[2]],src_pts[sample[3]]])
        H = find_homographi(destination_sample,source_sample,4)
        ones = np.ones((97,1))
        dst_pts_extended = np.hstack((dst_pts,ones))
        ans = np.matmul(H,dst_pts_extended.T).T
        first= ans[:,0]/ans[:,2]
        second = ans[:,1]/ans[:,2]
        ans = np.vstack((first,second)).T
        d = np.sum((src_pts-ans)**2,axis = 1)
        boolean_mask = np.where(d>threshold,0,1)
        suport = boolean_mask.sum()
        temp_w = suport/n
        if temp_w > w :
            w = temp_w
            final_dest =[]
            final_src = []
            for i in range(0,boolean_mask.shape[0]):
                if boolean_mask[i]==1:
                    final_dest.append(dst_pts[i])
                    final_src.append(src_pts[i])
            final_suport = suport
            N = (math.log(1-p))/math.log(1-(w**4))
        counter+=1
    final_homography = find_homographi(np.asarray(final_dest),np.asarray(final_src),final_suport)
    return final_homography

img1 = cv2.imread('im03.jpg')        
img2 = cv2.imread('im04.jpg')
sift = cv2.SIFT_create()
key_points1, descriptors1 = sift.detectAndCompute(img1,None)
key_points2, descriptors2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
src_pts = []
dst_pts = []
for m in good:
    src_pts.append(np.float32(key_points1[m.queryIdx].pt))
    dst_pts.append(np.float32(key_points2[m.trainIdx].pt))

M = findHomography(dst_pts,src_pts,10.0)
print(M/M[2][2])
ans = cv2.warpPerspective(img2, M, (3000,3000))
cv2.imwrite('res20.jpg',ans)
