import datetime
import numpy as np
import cv2

K = 5

headtime = datetime.datetime.now()
img = cv2.imread('candidates/102531_2.jpeg')
# img = img[60:-60, 0:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 0]) # np.array([35, 43, 46])
upper = np.array([180, 255, 220]) # np.array([99, 255, 255])
mask = cv2.inRange(img, lower, upper)
cv2.imshow('mask', mask)
cv2.waitKey(0)

Z = img.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
labels = labels.flatten()
print(labels)
centers = np.uint8(centers)
print(centers)
res = centers[labels]
print(res)
res2 = res.reshape((img.shape))
print(datetime.datetime.now() - headtime)
res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
