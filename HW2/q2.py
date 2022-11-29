import cv2
import numpy as np
import matplotlib.pyplot as plt

coordinations = np.zeros((54,3), np.float32)
coordinations[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 22

def calibrate(a,b):
    object_coordinations = []
    image_coordinations = []
    for i in range(a,b):
        image_coordinations.append(coordinations)
        img = cv2.imread(f'im{i:02}.jpg')
        h = img.shape[0]
        w = img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(img, (9,6), None)
        if retval:
            corners = cv2.cornerSubPix(gray, corners, (13,13), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.002))
            object_coordinations.append(corners)
    ret, mtx, dist, rvecs, tvecs= cv2.calibrateCamera(image_coordinations, object_coordinations,(w, h), None, None)
    input_matrix = np.array([
        [1,0,w/2],
        [0,1,h/2],
        [0,0,1]
    ])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(image_coordinations, object_coordinations,(w, h), input_matrix, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(image_coordinations, object_coordinations,(w, h), input_matrix, None, flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT)
    print(mtx)


calibrate(1,11)
calibrate(6,16)
calibrate(11,21)
calibrate(1,21)