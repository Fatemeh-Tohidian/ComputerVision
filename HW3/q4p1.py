import cv2 as cv
import numpy as np
import glob
counter = 0
train_data = np.zeros((2985,256),np.float32)
test_data = np.zeros((1500,256),np.float32)
responses = np.zeros((2985,1),np.float32)
true_responses = np.zeros((1500,1))

for i , place in enumerate(glob.glob('Data/Train/*')):
    for image_path in glob.glob(place+'/*'):
        image  = cv.resize(cv.imread(image_path,0), (16, 16)) 
        vector = np.reshape(image,(1,-1))
        train_data[counter] = vector
        responses[counter] = i
        counter+=1

knn = cv.ml.KNearest_create()
knn.train(train_data, cv.ml.ROW_SAMPLE, responses)

counter = 0
for i , place in enumerate(glob.glob('Data/Test/*')):
    for image_path in glob.glob(place+'/*'):
        image  = cv.resize(cv.imread(image_path,0), (16, 16)) 
        vector = np.reshape(image,(1,-1))
        test_data[counter] = vector
        true_responses[counter] = i
        counter+=1

ret, results, neighbours ,dist = knn.findNearest(test_data, 3)
d = results-true_responses
false_lables = np.count_nonzero(d)
prescision = (1-(false_lables/1500))*100
print(prescision)