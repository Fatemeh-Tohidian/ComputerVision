import cv2 as cv
import numpy as np

def get_vanishing_point(edges,threshold,min_theta_co,max_theta_co):
    lines = cv.HoughLines(edges,1,np.pi/180,threshold,min_theta= min_theta_co*np.pi ,max_theta=max_theta_co*np.pi)
    A = np.zeros((lines.shape[0],3))
    for i, line in enumerate(lines):
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        A[i] = [a,b,-rho]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    return vh[2,:]/vh[2,2]

def get_h_line(img,a,b,c):
    x1 = 0
    y1 = int(-c/b)
    x2 = int(img.shape[1])
    y2 = int((-c - x2*(a))/b)
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),10)
    cv.imwrite('res01.jpg',img)
    print(a/(np.sqrt((a**2)+(b**2))))
    print(b/(np.sqrt((a**2)+(b**2))))
    print(c/(np.sqrt((a**2)+(b**2))))


original_img = cv.imread('vns.jpg')
gray = cv.cvtColor(original_img,cv.COLOR_BGR2GRAY)
original_edges = cv.Canny(gray,50,150,apertureSize = 3)
h = 15000
w = 4000
A = 2800
B = 14000

V_x = get_vanishing_point(original_edges.copy(),780,1/2 ,10/18)
V_y = get_vanishing_point(original_edges.copy(),780,8/18 ,1/2)
V_z = get_vanishing_point(original_edges.copy(),850,-1/18 ,1/18)
h_line = np.cross(V_x/10,V_y/10)
get_h_line(original_img.copy(),h_line[0],h_line[1],h_line[2])
V_x=(V_x[0:2]/10)+[A,B]
V_y=(V_y[0:2]/10)+[A,B]
V_z=(V_z[0:2]/10)+[A,B]
big_img = np.ones((h,w,3),np.uint8)*255
img_half = cv.resize(original_img, (0,0), fx=0.1, fy=0.1)
big_img[B:B+img_half.shape[0],A:A+img_half.shape[1]] = img_half
cv.circle(big_img, (int(V_x[0]),int(V_x[1])), 60, color = (0, 255, 0),thickness=-1)
cv.circle(big_img, (int(V_y[0]),int(V_y[1])), 60, color = (0, 0, 255),thickness=-1)
cv.circle(big_img, (int(V_z[0]),int(V_z[1])), 60, color = (255,0, 0),thickness=-1)
cv.line(big_img,(int(V_x[0]),int(V_x[1])),(int(V_y[0]),int(V_y[1])),(0,0,0),10)
cv.imwrite('res02.png',big_img)