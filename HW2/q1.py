from typing import final
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.transform import Rotation as R

small_panorama_shape = (6100,2500)
big_panorama_shape = (6000,2000)
img_shape = (1920,1080)
x_point =   1000
y_point = 800
a = 200
patch_width = 50

def fix_frame(frame):
    h, w = frame.shape[:2]
    transform_matrix  = np.float32([
        [1,0,-70],
        [0,1,-50],
        [0,0,1]])
    frame = cv2.warpPerspective(frame,transform_matrix,(w-70,h-50))
    return frame


def extract_frames():
    os.system(f'ffmpeg -t 00:00:30 -i video.mp4 -vf fps=30 frames/img%03d.png')

def find_homogrphy(img1 , img2):
    sift = cv2.SIFT_create()
    key_points1, descriptors1 = sift.detectAndCompute(img2,None)
    key_points2, descriptors2 = sift.detectAndCompute(img1,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=2)
    src_pts = []
    dst_pts = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            src_pts.append(np.float32(key_points1[m.queryIdx].pt))
            dst_pts.append(np.float32(key_points2[m.trainIdx].pt))
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0,maxIters=1000)
    return H

def draw_rect(img1,img2,H):
    src_pts = np.array([[x_point, y_point], [(x_point+a),y_point], 
                [(x_point)+a, (y_point)+a], [x_point, (y_point)+a]],np.int32)
    src_pts = src_pts.reshape((-1, 1, 2))
    res01_450_rect = cv2.polylines(img2.copy(), [src_pts], isClosed = True, color =(0, 0, 255), thickness=3)
    dst_pts = cv2.perspectiveTransform(src_pts.astype(np.float32), np.linalg.inv(H))
    res02_270_rect = cv2.polylines(img1.copy(), [dst_pts.astype(np.int32)], isClosed = True, color =(0, 0, 255), thickness=3)
    cv2.imwrite("res01-450-rect.jpg", res01_450_rect)
    cv2.imwrite("res02-270-rect.jpg", res02_270_rect)

def combine_images(img1, img2, H):
    transform_matrix  = np.float32([
        [1,0,1680],
        [0,1,400],
        [0,0,1]])
    img1_warped = cv2.warpPerspective(img1, np.matmul(transform_matrix,H), small_panorama_shape)
    empty_mask = np.ones((img2.shape[0],img2.shape[1],3),dtype=np.uint8)
    mask_warped = cv2.warpPerspective(empty_mask, transform_matrix, small_panorama_shape)
    img2_warped = cv2.warpPerspective(img2, transform_matrix, small_panorama_shape)
    empty_image = np.zeros((small_panorama_shape[1],small_panorama_shape[0],3),dtype=np.uint8)
    empty_image = (img1_warped*(1-mask_warped)) + (img2_warped*mask_warped)
    return empty_image

def warp_image(img, H):
    transform_matrix  = np.float32([
        [1,0,1680],
        [0,1,400],
        [0,0,1]])
    img_warped = cv2.warpPerspective(img, np.matmul(transform_matrix,H), small_panorama_shape)
    return img_warped

def warp_invert_image(img, H,shape):
    transform_matrix  = np.float32([
        [1,0,1680],
        [0,1,400],
        [0,0,1]])
    img_warped = cv2.warpPerspective(img, np.linalg.inv(np.matmul(transform_matrix,H)), shape)
    return img_warped

def get_mask(left_mask,right_mask):
    y1,x1 = np.where(left_mask[:,:,0]==1)
    right = x1.max()

    y2,x2 = np.where(right_mask[:,:,0]==1)
    left = x2.min()

    x_mean =int((left+right)/2)
    mask =  np.zeros((big_panorama_shape[1],big_panorama_shape[0],3))
    mask[:,:x_mean,:]=1

    return mask


def blend(img1,img2,mask):
    img1_laplacian = img1.astype(np.float32) - cv2.GaussianBlur(img1,(3,3),0)
    img2_laplacian = img2.astype(np.float32) - cv2.GaussianBlur(img2,(3,3),0)
    img1_half = cv2.resize(img1, (0,0), fx=0.5, fy=0.5).astype(np.float32)
    img2_half = cv2.resize(img2, (0,0), fx=0.5, fy=0.5).astype(np.float32)
    mask_half = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
    mask = cv2.GaussianBlur(mask,(3,3),0)
    mask_half = cv2.GaussianBlur(mask_half,(3,3),0)
    img = (mask_half*img1_half)+(1-mask_half)*img2_half
    img = cv2.resize(img, (0,0), fx=2, fy=2)
    img = ((mask*img1_laplacian)+(1-mask)*img2_laplacian)+img
    return img

def get_f(h):
    z = math.atan(-1 * h[0,1]/h[1,1])
    a = (h[2,0]/h[0,1]) * math.sin(z)
    b = math.pow( (2*math.cos(z) - img_shape[0]*a*h[1,1]) / (2*h[1,1]) , 2)
    if b > 1 :
        return -1
    return abs(math.sqrt(1 - b) / a)

def section1(img270,img450):
    H = find_homogrphy(img270,img450)
    draw_rect(img270,img450,H)
    panorama = combine_images(img270,img450,H)
    cv2.imwrite("res03-270-450-panorama.jpg", panorama)

def section2(img90,img270,img450,img630,img810):
    empty_mask = np.ones((img450.shape[0],img450.shape[1],3),dtype=np.uint8)
    transform_matrix  = np.float32([
        [1,0,1500],
        [0,1,500],
        [0,0,1]])
    H_270_450 = find_homogrphy(img270,img450)

    img450_warped = cv2.warpPerspective(img450, transform_matrix, big_panorama_shape)
    mask450_warped = cv2.warpPerspective(empty_mask, transform_matrix, big_panorama_shape)

    img270_warped = cv2.warpPerspective(img270, np.matmul(transform_matrix,H_270_450), big_panorama_shape)
    mask270_warped = cv2.warpPerspective(empty_mask, np.matmul(transform_matrix,H_270_450), big_panorama_shape)

    mask = get_mask(mask270_warped,mask450_warped)
    background_img = blend(img270_warped,img450_warped,mask)

    H_630_450 = find_homogrphy(img630,img450)
    img630_warped = cv2.warpPerspective(img630, np.matmul(transform_matrix,H_630_450), big_panorama_shape)
    mask630_warped = cv2.warpPerspective(empty_mask, np.matmul(transform_matrix,H_630_450), big_panorama_shape)
    mask = get_mask(mask450_warped,mask630_warped)
    background_img = blend(background_img,img630_warped,mask)

    H_90_450 = np.matmul(H_270_450,find_homogrphy(img90,img270))
    img90_warped = cv2.warpPerspective(img90, np.matmul(transform_matrix,H_90_450), big_panorama_shape)
    mask90_warped = cv2.warpPerspective(empty_mask, np.matmul(transform_matrix,H_90_450), big_panorama_shape)
    mask = get_mask(mask90_warped,mask270_warped)
    background_img = blend(img90_warped,background_img,mask)

    H_810_450 = np.matmul(H_630_450,find_homogrphy(img810,img630))
    img810_warped = cv2.warpPerspective(img810, np.matmul(transform_matrix,H_810_450), big_panorama_shape)
    mask810_warped = cv2.warpPerspective(empty_mask, np.matmul(transform_matrix,H_810_450), big_panorama_shape)
    mask = get_mask(mask630_warped,mask810_warped)
    background_img = blend(background_img,img810_warped,mask)
    cv2.imwrite("res04-key-frames-panorama.jpg", background_img)

def section3_calculation(img90,img270,img450,img630,img810):
    h = np.zeros((901,3,3),np.float32)
    for i in range(270,450):
        img = cv2.imread(f'frames/img{i}.png')
        h[i] = find_homogrphy(img, img450)
    H_270_450 = find_homogrphy(img270,img450)
    for i in range(90,270):
        img = cv2.imread(f'frames/img{i:03}.png')
        h[i] = np.matmul(H_270_450,find_homogrphy(img,img270))
    H_90_450 = np.matmul(H_270_450,find_homogrphy(img90,img270))
    for i in range(1,90):
        img = cv2.imread(f'frames/img{i:03}.png')
        h[i] = np.matmul(H_90_450,find_homogrphy(img,img90))
    for i in range(450,630):
        img = cv2.imread(f'frames/img{i}.png')
        h[i] = find_homogrphy(img, img450)
    H_630_450 = find_homogrphy(img630,img450)
    for i in range(630,811):
        img = cv2.imread(f'frames/img{i}.png')
        h[i] = np.matmul(H_630_450,find_homogrphy(img,img630))
    H_810_450 = np.matmul(H_630_450,find_homogrphy(img810,img630))
    for i in range(811,901):
        img = cv2.imread(f'frames/img{i}.png')
        h[i] = np.matmul(H_810_450,find_homogrphy(img,img810))
    h = h.reshape((901,9))
    cv2.imwrite("Homography.tiff", h)

def section3():
    h = cv2.imread("Homography.tiff", cv2.IMREAD_UNCHANGED)
    h = h.reshape((901,3,3))
    for i in range(1,901):
        img = cv2.imread(f'frames/img{i:03}.png')
        panorama = warp_image(img,h[i])
        cv2.imwrite(f'new_frames/res{i:03}.png', panorama)
    os.system(f'ffmpeg -framerate 30 -i new_frames/res%03d.png  res05-reference-plane.mp4')

def section4():
    empty_image = np.zeros((small_panorama_shape[1],small_panorama_shape[0],3),dtype=np.uint8)
    for j in range(0,6100,patch_width):
        r_patch  = np.zeros((901,2500,patch_width),dtype=np.int16)
        g_patch  = np.zeros((901,2500,patch_width),dtype=np.int16)
        b_patch  = np.zeros((901,2500,patch_width),dtype=np.int16)
        final_patch = np.zeros((2500,patch_width,3),dtype=np.int16)
        for i in range(1,901):
            r_patch[i],g_patch[i],b_patch[i] = cv2.split(cv2.imread(f'new_frames/res{i:03}.png')[:,j:j+patch_width,:])
        r_patch_add  = r_patch.copy()
        r_patch_add = np.where(r_patch_add == 0,300,r_patch_add)
        g_patch_add  = g_patch.copy()
        g_patch_add = np.where(g_patch_add == 0,300,g_patch_add)
        b_patch_add  = b_patch.copy()
        b_patch_add = np.where(b_patch_add == 0,300,b_patch_add)

        r_patch = np.where(r_patch == 0,-300,r_patch)
        g_patch = np.where(g_patch == 0,-300,g_patch)
        b_patch = np.where(b_patch == 0,-300,b_patch)

        r_patch = np.vstack((r_patch,r_patch_add))
        g_patch = np.vstack((g_patch,g_patch_add))
        b_patch = np.vstack((b_patch,b_patch_add))

        final_patch[:,:,0] = np.median(r_patch,axis=0)
        final_patch[:,:,1] = np.median(g_patch,axis=0)
        final_patch[:,:,2] = np.median(b_patch,axis=0)

        empty_image[:,j:j+patch_width,:] = final_patch

    cv2.imwrite("res06-background-panorama.jpg", empty_image)

def section5():
    h = cv2.imread("Homography.tiff", cv2.IMREAD_UNCHANGED)
    h = h.reshape((901,3,3))
    panorama =  cv2.imread("res06-background-panorama.jpg")
    for i in range(1,901):
        img = warp_invert_image(panorama,h[i],img_shape)
        cv2.imwrite(f'background_frames/img{i:03}.png', img)
    os.system(f'ffmpeg -framerate 30 -i background_frames/img%03d.png  res07-background-video.mp4')

def section6():
    kernel = np.ones((7,7),np.uint8)
    for i in range(1,901):
        back = cv2.imread(f'background_frames/img{i:03}.png').astype(np.int32)
        img = cv2.imread(f'frames/img{i:03}.png').astype(np.int32)
        d = (img - back)**2
        d = np.sum(d,axis=2)
        d =  np.where(d<11000,0,1)
        d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel)
        a = np.dstack((d,d))
        img[:,:,2] = np.where(d==1,img[:,:,2]+100,img[:,:,2])
        img[:,:,:2] = np.where(a==1,img[:,:,:2]-100,img[:,:,:2])
        cv2.imwrite(f'forground_frames/img{i:03}.png', img)
    os.system(f'ffmpeg -framerate 30 -i forground_frames/img%03d.png  res08-foreground-video.mp4')

def section7():
    h = cv2.imread("Homography.tiff", cv2.IMREAD_UNCHANGED)
    h = h.reshape((901,3,3))
    panorama =  cv2.imread("res06-background-panorama.jpg")
    wider_shape=(int(1.5*img_shape[0]),img_shape[1])
    for i in range(1,901):
        img = warp_invert_image(panorama,h[i],wider_shape)
        cv2.imwrite(f'background_frames_wider/img{i:03}.png', img)
    os.system(f'ffmpeg -framerate 30 -i background_frames_wider/img%03d.png  res09-background-video-wider.mp4')

def section8():
    h = cv2.imread("Homography.tiff", cv2.IMREAD_UNCHANGED)
    h = h.reshape((901,3,3))
    f = np.zeros((901))
    for i in range(1,901):
        f[i] = get_f(h[i]/h[i][2][2])
        if f[i]==-1 :
            f[i] = f[i-1]
        # print(f[i])
    f = np.sort(f)
    f = np.delete(f, [0,1,2,3,4])
    f = np.delete(f, [-1,-2,-3,-4,-5])
    F = f.mean()
    print(F)
    # exit(0)
    print(img_shape)
    k = np.array([
        [F,0,img_shape[0]/2],
        [0,F,img_shape[1]/2],
        [0,0,1]
    ])
    x = np.zeros((900),dtype=np.float32)
    y = np.zeros((900),dtype=np.float32)
    z = np.zeros((900),dtype=np.float32)
    k2 = np.linalg.inv(k)
    for i in range(0,900):
        r = np.matmul(np.matmul(k2,h[i+1]),k)
        x[i], y[i], z[i] = R.from_matrix(r).as_euler('xyz')
    window = 150
    window_half = int((window/2))
    last_x = np.ones(window_half,dtype=np.float32)*x[899]
    last_y = np.ones(window_half,dtype=np.float32)*y[899]
    last_z = np.ones(window_half,dtype=np.float32)*z[899]
    first_x = np.ones(window_half,dtype=np.float32)*x[0]
    first_y = np.ones(window_half,dtype=np.float32)*y[0]
    first_z = np.ones(window_half,dtype=np.float32)*z[0]
    extended_x = np.hstack((x,last_x))
    extended_y = np.hstack((y,last_y))
    extended_z = np.hstack((z,last_z))
    extended_x = np.hstack((first_x,extended_x))
    extended_y = np.hstack((first_y,extended_y))
    extended_z = np.hstack((first_z,extended_z))
    x =  np.convolve(extended_x, np.ones(window+1), 'valid') / (window+1)
    y =  np.convolve(extended_y, np.ones(window+1), 'valid') / (window+1)
    z =  np.convolve(extended_z, np.ones(window+1), 'valid') / (window+1)
    for i in range(0,900):
        print(i)
        r_x = np.array([
            [1,0,0],
            [0,math.cos(x[i]),-1*math.sin(x[i])],
            [0,math.sin(x[i]),math.cos(x[i])]
        ])
        r_y = np.array([
            [math.cos(y[i]),0,math.sin(y[i])],
            [0,1,0],
            [-1*math.sin(y[i]),0,math.cos(y[i])]
        ])
        r_z = np.array([
            [math.cos(z[i]),-1*math.sin(z[i]),0],
            [math.sin(z[i]),math.cos(z[i]),0],
            [0,0,1]
        ])
        r2 = np.matmul(np.matmul(r_z,r_y),r_x)
        r2=r2/r2[2,2]
        h2 = np.matmul(np.matmul(k,r2),k2)
        img = cv2.imread(f'frames/img{(i+1):03}.png')
        final_h = np.matmul(np.linalg.inv(h2),h[i+1])
        img_warped = cv2.warpPerspective(img, final_h, (img.shape[1],img.shape[0]))
        img_warped =fix_frame(img_warped)
        cv2.imwrite(f'no_shake/img{(i+1):03}.png', img_warped)
    os.system(f'ffmpeg -framerate 30 -i no_shake/img%03d.png  res10-video-shakless.mp4')

# extract_frames()
img270 = cv2.imread('frames/img270.png')        
img450 = cv2.imread('frames/img450.png')
img90 = cv2.imread('frames/img090.png')
img630 = cv2.imread('frames/img630.png')
img810 = cv2.imread('frames/img810.png')
# section1(img270,img450)
# section2(img90,img270,img450,img630,img810)
# section3_calculation(img90,img270,img450,img630,img810)
# section3()
# section4()
# section5()
# section6()
# section7()
section8()