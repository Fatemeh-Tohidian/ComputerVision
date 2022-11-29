import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

original_img = cv.imread('vns.jpg')
gray = cv.cvtColor(original_img,cv.COLOR_BGR2GRAY)
original_edges = cv.Canny(gray,50,150,apertureSize = 3)

def get_vanishing_point(edges,threshold,min_theta_co,max_theta_co):
    lines = cv.HoughLines(edges,1,np.pi/180,threshold,min_theta= min_theta_co*np.pi ,max_theta=max_theta_co*np.pi)
    A_z = np.zeros((lines.shape[0],3))
    for i, line in enumerate(lines):
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        A_z[i] = [a,b,-rho]
    u, s, vh = np.linalg.svd(A_z, full_matrices=True)
    return vh[2,:]/vh[2,2]

def get_priciple_point(V_x,V_y,V_z):
    A = np.array([[V_x[0]-V_z[0],V_x[1]-V_z[1]],[V_y[0]-V_z[0],V_y[1]-V_z[1]]])
    B = np.array([(V_y[0]*(V_x[0]-V_z[0]))+(V_y[1]*(V_x[1]-V_z[1])),(V_x[0]*(V_y[0]-V_z[0]))+(V_x[1]*(V_y[1]-V_z[1]))])
    x = np.linalg.solve(A, B)
    return x[0],x[1]

V_x = get_vanishing_point(original_edges.copy(),780,1/2 ,10/18)
V_y = get_vanishing_point(original_edges.copy(),780,8/18 ,1/2)
V_z = get_vanishing_point(original_edges.copy(),850,-1/18 ,1/18)
h_line = np.cross(V_x/10,V_y/10)
px,py = get_priciple_point(V_x,V_y,V_z)
f = (-(px**2)-(py**2)+((V_x[0]+V_y[0])*px)+((V_x[1]+V_y[1])*py)-(V_x[0]*V_y[0])-(V_x[1]*V_y[1]))**0.5
print(px,py)
print(f)
cv.circle(original_img, (int(px),int(py)), 50, color = (255,0, 0),thickness=-1)
plt.imshow(original_img)
plt.title(str(f))
plt.savefig('res03.jpg')
k = np.array([[f,0,px],[0,f,py],[0,0,1]])
x = np.matmul(np.linalg.inv(k),V_x)
y = np.matmul(np.linalg.inv(k),V_y)
z = np.matmul(np.linalg.inv(k),V_z)
z_angle = (np.arctan(-h_line[0]/h_line[1])/np.pi)*180
x_angle = (np.arccos(np. dot(z / np. linalg. norm(z) , [0,0,1]))/np.pi)*180
print(z_angle)
print(x_angle)
