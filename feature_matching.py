import os
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect(target_image):
    brisk = cv2.BRISK_create()
    # fast = cv2.FastFeatureDetector_create()
    # freak = cv2.xfeatures2d.FREAK_create()
    orb = brisk

    factor = 0.8

    candidates = {}
    for image in os.listdir('candidates'):
        if image.startswith('.'):
            continue
        name = image.split('_')[0]
        # if image.startswith('IMG_') or image.startswith('Snipaste'):
        #     continue
        # if not image.startswith('Snipaste_'):
        #     continue
        img = cv2.imread('candidates/' + image, 0)
        # h, w = img.shape[:2]
        # print('img', h, w)
        # img = cv2.resize(img, None, fx=factor, fy=factor,
        #                  interpolation=cv2.INTER_AREA)
        # h, w = res.shape[:2]
        # cv2.imshow('res', res)
        # print('res', h, w)
        # kp = fast.detect(img, None)
        # kp, des = freak.compute(img, kp)
        kp, des = orb.detectAndCompute(img, None)
        if name not in candidates:
            candidates[name] = []
        candidates[name].append({
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
    img = cv2.imread(target_image, 0)  # trainImage
    # img = cv2.resize(img, None, fx=factor, fy=factor,
    #                  interpolation=cv2.INTER_AREA)
    # orb = cv2.ORB_create(nfeatures=5000)
    kp, des = orb.detectAndCompute(img, None)
    for name in candidates:
        matched = False
        for candidate in candidates[name]:
            # # Match descriptors.
            matches = bf.knnMatch(candidate['des'], des, k=2)
            # matches = bf.match(candidate['des'], des)
            # matches = sorted(matches, key=lambda x: x.distance)

            # matches = flann.knnMatch(candidate['des'], des, k=2)
            # print(len(matches))

            # result = cv2.drawMatches(
            #     candidate['img'], candidate['kp'], img, kp, matches[:50], None, flags=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            # print(len(candidate['kp']), len(kp), len(good))
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
            # cv2.namedWindow('result', 16)
            # cv2.imshow('result', result)
            # cv2.waitKey(0)
            if len(good) > 10:
                matched = True
                break
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
        if matched:
            print(name + ' found')

    endtime = datetime.datetime.now()
    print(endtime - starttime)


def crop_targets(input_dir, outpu_dir):
    for image in os.listdir(input_dir):
        if image.startswith('.') or image.startswith('IMG_') or image.startswith('Snipaste'):
            continue
        img = cv2.imread(input_dir + '/' + image)
        img = img[0:1000, 300:1960]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 46]) # np.array([35, 43, 46])
        upper = np.array([180, 43, 255]) # np.array([99, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        _, mask = cv2.threshold(mask, 128, 255, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        mask = cv2.dilate(mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2 = x1 + w
        y2 = y1 + h
        for i, c in enumerate(contours):
            if i == 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            x1 = min(x, x1)
            y1 = min(y, y1)
            x2 = max(x + w, x2)
            y2 = max(y + h, y2)

        img = img[y1:y2, x1:x2]
        h, w = img.shape[:2]
        cv2.imwrite(outpu_dir + '/' + image, img)


if __name__ == '__main__':
    crop_targets('origin', 'candidates')
    detect('capture/137406563_1_20180102T114951Z.jpg')