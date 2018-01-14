import os
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import cv2
import matplotlib.pyplot as plt


brisk = cv2.BRISK_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
candidates = {}
factor = 0.8
for image in os.listdir('candidates'):
    if image.startswith('.'):
        continue
    name = image.split('_')[0]
    img = cv2.imread('candidates/' + image, 0)
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    kp, des = brisk.detectAndCompute(img, None)
    if name not in candidates:
        candidates[name] = []
    candidates[name].append({
        'kp': kp,
        'des': des,
        'img': img
    })

def init():
    factor = 0.5
    for image in os.listdir('candidates'):
        if image.startswith('.'):
            continue
        name = image.split('_')[0]
        img = cv2.imread('candidates/' + image, 0)
        img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        kp, des = brisk.detectAndCompute(img, None)
        if name not in candidates:
            candidates[name] = []
        candidates[name].append({
            'kp': kp,
            'des': des,
            'img': img
        })

def count_matches(name, img):
    kp, des = brisk.detectAndCompute(img, None)
    count = 0
    for candidate in candidates[name]:
        matches = bf.knnMatch(candidate['des'], des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # print(len(good))
        if len(good) > 1:
            count += 1
            # print(len(candidate['kp']), len(kp), len(good))
            # print(name + ' found: ' + str(len(good)))
            src_pts = np.float32([ candidate['kp'][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            contours = []
            for m in good:
                x, y = kp[m.trainIdx].pt
                contours.append([int(x), int(y)])
            img = cv2.fillConvexPoly(img, np.array(contours, dtype=np.int32), 1)
            x, y, w, h = cv2.boundingRect(np.array(contours, dtype=np.int32))
            pts = np.float32([ [x,y],[x,y+h],[x+w,y+h],[x+w,y] ]).reshape(-1,1,2)
            img = cv2.fillConvexPoly(img, np.array(pts, np.int32), 1)

            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            # h,w = candidate['img'].shape
            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv2.perspectiveTransform(pts,M)
            # img = cv2.fillConvexPoly(img, np.array(dst, np.int32), 1)
            # img = cv2.polylines(img, [np.int32(dst)],True,255,3, cv2.LINE_AA)

            # cv2.namedWindow('result', 16)
            # cv2.imshow('result', img)
            # cv2.waitKey(0)

            kp, des = brisk.detectAndCompute(img, None)
    return {name: count}


def detect_multi_cpus(img):
    # futures = []
    # with ProcessPoolExecutor() as executor:
    #     for name in candidates:
    #         futures.append(executor.submit(count_matches, name, img))
    # resultdict = {}
    # for f in as_completed(futures):
    #     resultdict.update(f.result())
    # pool = mp.Pool(processes=6)
    pool = ThreadPool(12)
    starttime = datetime.datetime.now()
    results = pool.starmap(count_matches, [(name, img) for name in candidates])
    print(datetime.datetime.now() - starttime)
    return results


def detect(candidates, target_image):
    factor=0.8
    img = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    # detector = cv2.SimpleBlobDetector_create()
    # keypoints = detector.detect(img)
    # im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
    print(detect_multi_cpus(img))
    # starttime = datetime.datetime.now()
    # for name in candidates:
    #     print(count_matches(name, img))
    # print(datetime.datetime.now() - starttime)
    return

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
        
    for name in candidates:
        # matched = False
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
                    good.append(m)
            if len(good) > 10:
                print(len(candidate['kp']), len(kp), len(good))
                print(name + ' found')
                src_pts = np.float32([ candidate['kp'][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = candidate['img'].shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                img = cv2.fillConvexPoly(img, np.array(dst, np.int32), 1)
                img = cv2.polylines(img, [np.int32(dst)],True,255,3, cv2.LINE_AA)
                kp, des = orb.detectAndCompute(img, None)
                cv2.namedWindow('result', 16)
                cv2.imshow('result', img)
                cv2.waitKey(0)
                # matched = True
                # break
            # result = cv2.drawMatchesKnn(
            #     candidate['img'], candidate['kp'], img, kp, good, None, flags=2)
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #                singlePointColor = None,
            #                matchesMask = matchesMask, # draw only inliers
            #                flags = 2)
            # result = cv2.drawMatches(candidate['img'],candidate['kp'],img,kp,good,None,**draw_params)
            
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
        # if matched:
        #     print(name + ' found')

    endtime = datetime.datetime.now()
    print(endtime - starttime)


def crop_targets(input_dir, outpu_dir):
    for image in os.listdir(input_dir):
        if image.startswith('.') or image.startswith('IMG_') or image.startswith('Snipaste'):
            continue
        img = cv2.imread(input_dir + '/' + image)
        # img = img[0:1000, 300:1960]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 46]) # np.array([35, 43, 46])
        upper = np.array([180, 43, 255]) # np.array([99, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        _, mask = cv2.threshold(mask, 128, 255, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        mask = cv2.dilate(mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2 = x1 + w
        y2 = y1 + h
        img = img[y1:y2, x1:x2]
        h, w = img.shape[:2]
        cv2.imwrite(outpu_dir + '/' + image, img)


if __name__ == '__main__':
    # crop_targets('origin', 'candidates')
    # init()
    detect(candidates, 'capture/2018_01_14_07_38_17.jpg')