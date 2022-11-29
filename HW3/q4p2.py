import cv2 as cv
import numpy as np
import glob

counter = 0
histogram_train_data = np.zeros((2985,100),np.float32)
histogram_test_data = np.zeros((1500,100),np.float32)
responses = np.zeros((2985,1),np.float32)
true_responses = np.zeros((1500,1))
flag = True
for place in glob.glob('Data/Train/*'):
    for image_path in glob.glob(place+'/*'):
        print(image_path)
        image  = cv.imread(image_path,0)
        sift = cv.SIFT_create() 
        kp,ds = sift.detectAndCompute(image, None)
        if flag :
            kmeans_data = ds
            flag = False
        else:
            kmeans_data = np.concatenate((kmeans_data,ds))

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv.kmeans(kmeans_data,100,None,criteria,10,flags)

knn = cv.ml.KNearest_create()
knn.train(centers, cv.ml.ROW_SAMPLE, np.arange(100))

counter = 0
for i , place in enumerate(glob.glob('Data/Train/*')):
    for image_path in glob.glob(place+'/*'):
        print(i)
        image  = cv.imread(image_path,0)
        responses[counter] = i
        sift = cv.SIFT_create() 
        kp,ds = sift.detectAndCompute(image, None)
        ret, results, neighbours ,dist = knn.findNearest(ds, 1)
        for result in results:
            histogram_train_data[counter][int(result[0])]+=1
        counter+=1

counter = 0
for i , place in enumerate(glob.glob('Data/Test/*')):
    for image_path in glob.glob(place+'/*'):
        print(i)
        image  = cv.imread(image_path,0)
        true_responses[counter] = i
        sift = cv.SIFT_create() 
        kp,ds = sift.detectAndCompute(image, None)
        ret, results, neighbours ,dist = knn.findNearest(ds, 1)
        for result in results:
            histogram_test_data[counter][int(result[0])]+=1
        counter+=1

knn = cv.ml.KNearest_create()
knn.train(histogram_train_data, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(histogram_test_data, 1)
d = results-true_responses
false_lables = np.count_nonzero(d)
prescision = (1-(false_lables/1500))*100
print(prescision)