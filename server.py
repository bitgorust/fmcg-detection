import os
import json
import datetime
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import cv2
from bottle import run, get, post, request, response


product = {}
with open('products.csv', 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        [pid, name, barcode] = line.split(',')
        product[pid.strip()] = name.strip()
print(len(product))

difficult = {
    '111471': 50,
    # '103055': 50,
    '102544': 10,
    '102573': 15,
    '113351': 10,
    '102524': 4,
    '102525': 4,
    '102586': 10,
    '105011': 10,
}

DEBUG = False
PROCESSES = 12
MATCH_DISTANCE = 0.7
GOOD_THRESHOLD = 17
RESIZE_FACTOR = 1.0
CANDIDATE_DIR = 'candidates'

pool = ThreadPool(PROCESSES)
brisk = cv2.BRISK_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


candidate_keys = []
candidates = {}
for file in os.listdir(CANDIDATE_DIR):
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
    kp, des = orb.detectAndCompute(img, None) if name not in difficult else brisk.detectAndCompute(img, None)
    if name not in candidates:
        candidate_keys.append(name)
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
candidate_keys = sorted(candidate_keys, key=lambda x: x in ('102573'))


def save_result(dir, name, img1, kp1, img2, kp2, good):
    goods = []
    for m in good:
        goods.append([m])
    img1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    img2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goods, None, flags=2)
    dir = 'debug/' + dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    cv2.imwrite(dir + '/' + name + '.png', img)


def count_matches(name, img, identity=None):
    goods = []
    for candidate in candidates[name]:
        len_good = -1
        index = 0
        while True:
            index += 1
            kp, des = orb.detectAndCompute(img, None) if name not in difficult else brisk.detectAndCompute(img, None)
            len_candidate_kp = len(candidate['kp'])

            matches = bf.knnMatch(candidate['des'], des, k=2)
            good = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < MATCH_DISTANCE * n.distance:
                        good.append(m)

            if len_good != len(good):
                len_good = len(good)
            else:
                break

            if DEBUG and identity is not None:
                save_result(identity, '_'.join([str(len_good), str(len(candidate['kp'])), str(len(kp)), candidate['file'], str(index)]), candidate['img'], candidate['kp'], img, kp, good)

            if name not in difficult and candidate['threshold'] == 0 and len_good < GOOD_THRESHOLD * RESIZE_FACTOR:
                break
            if name not in difficult and candidate['threshold'] > 0 and len_good / len_candidate_kp < candidate['threshold'] * RESIZE_FACTOR:
                break
            if name in difficult and len_good < difficult[name] * RESIZE_FACTOR:
                break

            print(candidate['file'] + ': ' + str(len_good) +
                  '/' + str(len_candidate_kp) + '=' + str(len_good / len_candidate_kp))

            if not candidate['expand']:
                src_pts = np.float32(
                    [candidate['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                transform, _ = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0)
                if transform is None:
                    break

                h, w = candidate['img'].shape
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, transform)
                if dst is None:
                    break

                shape = [point for points in dst.tolist() for point in points]
                img = cv2.fillConvexPoly(img, np.array(dst, np.int32), 1)
            else:
                contours = []
                for m in good:
                    x, y = kp[m.trainIdx].pt
                    contours.append([int(x), int(y)])
                x, y, w, h = cv2.boundingRect(
                    np.array(contours, dtype=np.int32))
                shape = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                pts = np.float32(shape).reshape(-1, 1, 2)
                img = cv2.fillConvexPoly(img, np.array(pts, np.int32), 1)

            goods.append({
                'shape': shape,
                'score': len_good / len_candidate_kp,
                'file': candidate['file'],
                'good': len_good,
                'kp': len_candidate_kp,
                'threshold': candidate['threshold']
            })
    return name, goods


def analyze(img_str, identity=None):
    headtime = datetime.datetime.now()
    img_np = np.fromstring(img_str, np.uint8)

    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if RESIZE_FACTOR < 1.0:
        img = cv2.resize(img, None, fx=RESIZE_FACTOR,
                         fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

    matches = pool.starmap(count_matches, [(name, img, identity) for name in candidate_keys])

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
    print(datetime.datetime.now() - headtime)
    return result


@get('/hello')
def hello():
    return 'welcome'


@post('/detect')
def detect():
    response.set_header('Content-Type', 'application/json; charset=utf-8')

    for key in request.headers:
        print(key, request.get_header(key))

    result = {}

    starttime = datetime.datetime.now()
    print('getting image')
    image = request.files.get('image')
    print('image got: ' + str(datetime.datetime.now() - starttime))
    if image is None:
        result['code'] = 400
        result['message'] = u'image文件为空'
        response.status = 400
        return json.dumps(result)

    _, ext = os.path.splitext(image.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
        result['code'] = 400
        result['message'] = u'只支持png/jpg格式的文件'
        response.status = 400
        return json.dumps(result)

    starttime = datetime.datetime.now()
    print('saving ' + image.filename)
    image.save('capture/', overwrite=True)
    print('finish saving: ' + str(datetime.datetime.now() - starttime))

    starttime = datetime.datetime.now()
    print('reading ' + image.filename)
    img_str = image.file.read()
    print('finish reading: ' + str(datetime.datetime.now() - starttime))
    result = analyze(img_str, image.filename)
    return json.dumps(result)


if __name__ == "__main__":
    run(server='paste', host='0.0.0.0', port=8099)
