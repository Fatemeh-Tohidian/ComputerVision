import cv2
import numpy
import matplotlib.pyplot as plt
from skimage.feature import corner_peaks

def get_picks(R):
    count,res = cv2.connectedComponents(R)
    points = numpy.zeros((count,2),dtype= int)
    for i in range(0,count):
        component = numpy.where(res==i,R,0)
        point = numpy.argwhere(component==component.max())[0]
        points[i] = point
    return points

def non_maximum_supression(R,img,id):
    R = R.astype(numpy.uint8)
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    mask = get_picks(R)
    n = 20
    feature_vector = numpy.zeros((mask.shape[0],(3*(n**2))))
    for index, point in enumerate(mask):
        if point[1] in range(n,img.shape[0]-n) and point[0] in range(n,img.shape[1]-n):
            window = img[point[0]-int(n/2):point[0]+int(n/2),point[1]-int(n/2):point[1]+int(n/2),:]
            feature_vector[index]=window.reshape(-1)
            cv2.circle(img, (point[1],point[0]), 10, color = (255, 0, 0),thickness=-1)
    cv2.imwrite(f"res0{id+6}_harris.jpg",img)
    return feature_vector,mask


def get_grad(img,id,K):
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = ((Ix**2)+(Iy**2))**0.5
    abs_grad = cv2.convertScaleAbs(grad)
    cv2.imwrite(f"res0{id}_grad.jpg",abs_grad)
    Ix2 = (Ix**2)
    Iy2 = (Iy**2)
    Ixy = (Ix*Iy)
    s = 35
    Sx2 =  cv2.GaussianBlur(Ix2,(s,s),0)
    Sy2 =  cv2.GaussianBlur(Iy2,(s,s),0)
    Sxy =  cv2.GaussianBlur(Ixy,(s,s),0)
    det = (Sx2*Sy2) - (Sxy**2)
    trace = Sx2+Sy2
    R = det - K*(trace**2)
    cv2.imwrite(f"res0{id+2}_score.jpg",R)
    R = numpy.where(R >10000000, R,0)
    R = cv2.convertScaleAbs(R)
    cv2.imwrite(f"res0{id+4}_thresh.jpg",R)
    feature_vector,mask = non_maximum_supression(R,numpy.copy(img),id)
    return feature_vector,mask

def compare_points(vectors1,vectors2,treshold,labels,img1,mask1,id):
    m = vectors2.shape[0]
    for index, vector in enumerate(vectors1):
        repeted_vectors = numpy.array([vector,]*m)
        res = (repeted_vectors-vectors2)**2
        column = numpy.sum(res,axis=1)
        d1, d2 = numpy.partition(column, 1)[0:2]
        if d2/d1 >  treshold or d1 ==0:
            p = numpy.argwhere(column == d1)[0][0]
            labels[index][p] =1
            cv2.circle(img1, (mask1[index][1],mask1[index][0]), 10, (255, 10*(index), 100*((index)%2)), -1)
    cv2.imwrite(f"res0{id+8}_corres.jpg",img1)
    return labels

img1 = cv2.imread("im01.jpg")
img2 = cv2.imread("im02.jpg")
feature_vector1,mask1 = get_grad(img1,1,0.01)
feature_vector2,mask2 = get_grad(img2,2,0.01)
labels1  = numpy.zeros((feature_vector1.shape[0],feature_vector2.shape[0])) 
labels1 = compare_points(feature_vector1,feature_vector2,1.9,labels1,img1.copy(),mask1,1)
labels2  = numpy.zeros((feature_vector2.shape[0],feature_vector1.shape[0])) 
labels2 = compare_points(feature_vector2,feature_vector1,1.9,labels2,img2.copy(),mask2,2)
labels = labels1*(labels2.T)
img = cv2.hconcat([img1, img2])
lines = numpy.argwhere(labels == 1)
for index,line in enumerate(lines):
    cv2.circle(img, (mask1[line[0]][1],mask1[line[0]][0]), 15, (150-50*(index%3),40+50*(index%5),10+40*(index%7)), -1)
    cv2.circle(img, (img1.shape[1]+mask2[line[1]][1],mask2[line[1]][0]), 15, (150-50*(index%3),40+50*(index%5),10+40*(index%7)), -1)
    img = cv2.line(img, (mask1[line[0]][1],mask1[line[0]][0]), (img1.shape[1]+mask2[line[1]][1],mask2[line[1]][0]), (150-50*(index%3),40+50*(index%5),10+40*(index%7)), 7)
cv2.imwrite('res11.jpg',img)
