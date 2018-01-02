import os
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Initiate ORB detector
brisk = cv2.BRISK_create()
# fast = cv2.FastFeatureDetector_create()
# freak = cv2.xfeatures2d.FREAK_create()
orb = brisk

factor = 0.4

candidates = []
for image in os.listdir('feature_test'):
    if image.startswith('IMG_') or image.startswith('Snipaste'):
        continue
    # if not image.startswith('Snipaste_'):
    #     continue
    img = cv2.imread('feature_test/' + image, 0)
    # h, w = img.shape[:2]
    # print('img', h, w)
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    # h, w = res.shape[:2]
    # cv2.imshow('res', res)
    # print('res', h, w)
    # kp = fast.detect(img, None)
    # kp, des = freak.compute(img, kp)
    kp, des = orb.detectAndCompute(img, None)
    candidates.append({
        'kp': kp,
        'des': des,
        'img': img
    })
    # cv2.waitKey(0)
    # img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
    # plt.imshow(img2),plt.show()
    # cv2.waitKey(0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12,
                    key_size=20, multi_probe_level=2)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

starttime = datetime.datetime.now()
img = cv2.imread('feature_test/IMG_20171226_163055.jpg', 0)  # trainImage
img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
# orb = cv2.ORB_create(nfeatures=5000)
kp, des = orb.detectAndCompute(img, None)
for candidate in candidates:
    # # Match descriptors.
    matches = bf.knnMatch(candidate['des'], des, k=2)
    # matches = bf.match(candidate['des'], des)
    # matches = sorted(matches, key=lambda x: x.distance)

    # matches = flann.knnMatch(candidate['des'], des, k=2)
    print(len(matches))

    # result = cv2.drawMatches(
    #     candidate['img'], candidate['kp'], img, kp, matches[:50], None, flags=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(len(candidate['kp']), len(kp), len(good))
    # if len(good) > 10:
    #     src_pts = np.float32([ candidate['kp'][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()

    #     h,w = candidate['img'].shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)

    #     img = cv2.polylines(img, [np.int32(dst)],True,255,3, cv2.LINE_AA)
    # result = cv2.drawMatchesKnn(
    #     candidate['img'], candidate['kp'], img, kp, good, None, flags=2)
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)
    # result = cv2.drawMatches(candidate['img'],candidate['kp'],img,kp,good,None,**draw_params)
    # plt.imshow(result), plt.show()
    # cv2.waitKey(0)
    continue

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, match in enumerate(matches):
        # print(match)
        if len(match) != 2:
            continue
        m, n = match
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    result = cv2.drawMatchesKnn(
        candidate['img'], candidate['kp'], img, kp, matches, None, **draw_params)
    plt.imshow(result), plt.show()
    cv2.waitKey(0)

endtime = datetime.datetime.now()
print(endtime - starttime)
