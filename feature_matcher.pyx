import os
import datetime
import numpy as np
import cv2
from multiprocessing.dummy import Pool as ThreadPool


PROCESSES = 12
MATCH_DISTANCE = 0.7
GOOD_THRESHOLD = 24
RESIZE_FACTOR = 0.6
CANDIDATE_DIR = 'candidates'


pool = None
brisk = None
bf = None
candidates = {}
product = {}


difficult = {
    '102392': 10,
    '102485': 11,
    '102502': 8,
    '102531': 19,
    '102541': 11,
    # '102573': 8,
    '102590': 8,
    '102612': 7,
    '102627': 9,
    '103594': 10,
    '104370': 44,
    '107093': 22,
    '113351': 52,
    '113369': 10,
    '104709': 22,
    '112939': 11,
    '111394': 11,
    '113345': 17,
    '113349': 11,
    '113383': 12,
    '102645': 10,
    '103321': 13,
    '102579': 8,
    '102544': 8,
    '102527': 16,
    '108280': 12,
    '108185': 10,
    '113382': 20,
    '111902': 29,
    '103014': 129,
    '103055': 26,
    '102586': 20,
    '112843': 18,
}


def say_hello_to(name):
    print("Hello %s!" % name)


def init(threads=PROCESSES, candidate_dir=CANDIDATE_DIR, product_file='products.csv'):
    with open(product_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            [pid, name, barcode] = line.split(',')
            product[pid.strip()] = name
    print(len(product))
    pool = ThreadPool(threads)
    brisk = cv2.BRISK_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    for file in os.listdir(candidate_dir):
        if file.startswith('.'):
            continue
        filename, _ = _, ext = os.path.splitext(file)
        info = filename.split('_')
        if len(info) < 2 and len(info) > 3:
            continue
        threshold = '0'
        if len(info) == 3:
            [name, idx, threshold] = info
        else:
            [name, idx] = info
        img = cv2.imread(CANDIDATE_DIR + '/' + file, 0)
        if len(info) == 3 and RESIZE_FACTOR < 1.0:
            img = cv2.resize(img, None, fx=RESIZE_FACTOR,
                            fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)
        kp, des = brisk.detectAndCompute(img, None)
        if name not in candidates:
            candidates[name] = []
        candidates[name].append({
            'kp': kp,
            'des': des,
            'img': img,
            'file': file,
            'threshold': float(threshold),
            'expand': idx == '0'
        })
    print(str(len(candidates)) + ' candidates')


def count_matches(name, img):
    kp, des = brisk.detectAndCompute(img, None)
    goods = []
    for candidate in candidates[name]:
        matches = bf.knnMatch(candidate['des'], des, k=2)
        good = []
        for m, n in matches:
            if m.distance < MATCH_DISTANCE * n.distance:
                good.append(m)
        score = len(good) / len(candidate['kp'])
        print(candidate['file'] + ': ' + str(len(good)) +
              '/' + str(len(candidate['kp'])) + '=' + str(score))
        if name in difficult and len(good) >= difficult[name] * RESIZE_FACTOR \
                or name not in difficult and candidate['threshold'] > 0 and score >= candidate['threshold'] * RESIZE_FACTOR \
                or name not in difficult and candidate['threshold'] == 0 and len(good) >= GOOD_THRESHOLD * RESIZE_FACTOR:
            if candidate['expand']:
                contours = []
                for m in good:
                    x, y = kp[m.trainIdx].pt
                    contours.append([int(x), int(y)])
                x, y, w, h = cv2.boundingRect(
                    np.array(contours, dtype=np.int32))
                shape = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                pts = np.float32(shape).reshape(-1, 1, 2)
                img = cv2.fillConvexPoly(img, np.array(pts, np.int32), 1)
            else:
                src_pts = np.float32(
                    [candidate['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                transform, _ = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0)
                if transform is None:
                    continue

                h, w = candidate['img'].shape
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, transform)
                if dst is None:
                    continue

                img = cv2.fillConvexPoly(img, np.array(dst, np.int32), 1)
                shape = [point for points in dst.tolist() for point in points]

            goods.append({
                'shape': shape,
                'score': score,
                'file': candidate['file'],
                'good': len(good),
                'kp': len(candidate['kp']),
                'threshold': candidate['threshold']
            })
            kp, des = brisk.detectAndCompute(img, None)

            # contours = []
            # for m in good:
            #     x, y = candidate['kp'][m.queryIdx].pt
            #     contours.append([int(x), int(y)])
            # # img = cv2.fillConvexPoly(img, np.array(contours, dtype=np.int32), 1)
            # x, y, w, h = cv2.boundingRect(np.array(contours, dtype=np.int32))
            # print(x, y, w, h)
            # pts = np.float32([ [x,y],[x,y+h],[x+w,y+h],[x+w,y] ]).reshape(-1,1,2)
            # can_img = cv2.fillConvexPoly(candidate['img'], np.array(pts, np.int32), 1)
            # can_img = cv2.polylines(can_img, [np.array(contours, dtype=np.int32)],True,255,3, cv2.LINE_AA)
            # cv2.imshow(name + str(score), can_img)
            # cv2.waitKey(0)

            # pts = np.float32(
            #      [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]).reshape(-1, 1, 2)

            # print(dst)

            # cv2.imshow(name + str(score), img)
            # cv2.waitKey(0)
    return name, goods


def detect(img_str):
    starttime = datetime.datetime.now()
    img_np = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if RESIZE_FACTOR < 1.0:
        img = cv2.resize(img, None, fx=RESIZE_FACTOR,
                         fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)
    print(datetime.datetime.now() - starttime)

    starttime = datetime.datetime.now()
    matches = pool.starmap(count_matches, [(name, img) for name in candidates])
    print(datetime.datetime.now() - starttime)
    
    starttime = datetime.datetime.now()
    result = {}
    for label, goods in matches:
        if len(goods) == 0:
            continue
        result[label] = {
            'name': product[label] if label in product else u'未知',
            'count': len(goods),
            'detail': goods
        }
    print(result)
    print(datetime.datetime.now() - starttime)
    return result