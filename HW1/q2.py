import cv2
import numpy
import matplotlib.pyplot as plt
import math

logo = cv2.imread('logo.png')
tetha = - math.atan(40/25)
R = numpy.array([[1,0,0],[0,math.cos(tetha),math.sin(tetha)],[0,-math.sin(tetha),math.cos(tetha)]])
n = numpy.array([1,0,0]).transpose()
d = -25
c = numpy.array([0,40,0]).transpose()
t = - numpy.matmul(R,c)
K = numpy.array([[500,0,128],[0,500,128],[0,0,1]])
K_prime = numpy.array([[500,0,2000],[0,500,2000],[0,0,1]])
G = (R-(numpy.dot(t,n)/d))
A = numpy.matmul(K_prime,G)
H = numpy.matmul(A,numpy.linalg.inv(K))
new_logo =  cv2.warpPerspective(logo, H, (4000,4000))
cv2.imwrite('res12.jpg',new_logo)

