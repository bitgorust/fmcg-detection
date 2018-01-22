import os
# import sys
import json
import datetime
from multiprocessing.dummy import Pool as ThreadPool

# from PIL import Image
# import tensorflow as tf
import numpy as np
import cv2
from bottle import run, get, post, request, response


# if tf.__version__ != '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

product = {}
with open('products.csv', 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        [pid, name, barcode] = line.split(',')
        product[pid.strip()] = name
print(len(product))

difficult = {
    # '102586': 100,
    # '105011': 100,
    # '102392': 10,
    # '102485': 11,
    # '102502': 8,
    # '102531': 19,
    # '102541': 11,
    # # '102573': 8,
    # '102590': 8,
    # '102612': 7,
    # '102627': 9,
    # '103594': 10,
    # '104370': 44,
    # '107093': 22,
    # '113351': 52,
    # '113369': 10,
    # '104709': 22,
    # '112939': 11,
    # '111394': 11,
    # '113345': 17,
    # '113349': 11,
    # '113383': 12,
    # '102645': 10,
    # '103321': 13,
    # '102579': 8,
    # '102544': 8,
    # '102527': 16,
    # '108280': 12,
    # '108185': 10,
    # '113382': 20,
    # '111902': 29,
    # '103014': 129,
    # '103055': 26,
    # '102586': 20,
    # '112843': 18,
}

# TOP_K = 6
# TOP_THRESHOLD = 0.1

PROCESSES = 12
MATCH_DISTANCE = 0.7
GOOD_THRESHOLD = 15
RESIZE_FACTOR = 1.0
CANDIDATE_DIR = 'candidates'

pool = ThreadPool(PROCESSES)
brisk = cv2.BRISK_create()
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


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
    kp, des = orb.detectAndCompute(img, None)
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


# def load_graph(model_file):
#     graph = tf.Graph()
#     graph_def = tf.GraphDef()

#     with open(model_file, "rb") as f:
#         graph_def.ParseFromString(f.read())
#     with graph.as_default():
#         tf.import_graph_def(graph_def)

#     return graph


# def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
#     file_reader = tf.read_file(file_name, "file_reader")
#     if file_name.endswith(".png"):
#         image_reader = tf.image.decode_png(
#             file_reader, channels=3, name='png_reader')
#     elif file_name.endswith(".gif"):
#         image_reader = tf.squeeze(
#             tf.image.decode_gif(file_reader, name='gif_reader'))
#     elif file_name.endswith(".bmp"):
#         image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
#     else:
#         image_reader = tf.image.decode_jpeg(
#             file_reader, channels=3, name='jpeg_reader')
#     float_caster = tf.cast(image_reader, tf.float32)
#     dims_expander = tf.expand_dims(float_caster, 0)
#     resized = tf.image.resize_bicubic(
#         dims_expander, [input_height, input_width])
#     normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
#     return tf.Session().run(normalized)


# def read_tensor_from_image_str(img_str, input_height=299, input_width=299, input_mean=0, input_std=255):
#     image_reader = tf.image.decode_jpeg(
#         tf.constant(img_str), channels=3, name='jpeg_reader')
#     float_caster = tf.cast(image_reader, tf.float32)
#     dims_expander = tf.expand_dims(float_caster, 0)
#     resized = tf.image.resize_bicubic(
#         dims_expander, [input_height, input_width])
#     normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
#     return tf.Session().run(normalized)


# def load_labels(label_file):
#     label = []
#     proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
#     for l in proto_as_ascii_lines:
#         label.append(l.rstrip())
#     return label


# model_file = "D:/label_image/outputs/10076-inception-v3-2018-01-17/retrained_graph.pb"
# label_file = "D:/label_image/outputs/10076-inception-v3-2018-01-17/retrained_labels.txt"
# input_height = 299 #224
# input_width = 299 #224
# input_mean = 0
# input_std = 255
# input_layer = "Mul" #"input"
# output_layer = "final_result"

# graph = load_graph(model_file)
# input_name = "import/" + input_layer
# output_name = "import/" + output_layer
# input_operation = graph.get_operation_by_name(input_name)
# output_operation = graph.get_operation_by_name(output_name)
# labels = load_labels(label_file)


def count_matches(name, img):
    # headtime = datetime.datetime.now()
    kp, des = orb.detectAndCompute(img, None)
    # print('detectAndCompute: ' + str(datetime.datetime.now() - headtime))
    goods = []
    for candidate in candidates[name]:
        for index in range(1, 6):
            # starttime = datetime.datetime.now()
            matches = bf.knnMatch(candidate['des'], des, k=2)
            # print('knnMatch: ' + str(datetime.datetime.now() - starttime))

            good = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < MATCH_DISTANCE * n.distance:
                        good.append(m)
            len_good = len(good)
            len_candidate_kp = len(candidate['kp'])

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
            # cv2.imwrite('test/' + name + '_' + str(index) +
            #             '_' + str(len_good) + '.png', img)
            kp, des = orb.detectAndCompute(img, None)
            # cv2.imshow(name + str(score), img)
            # cv2.waitKey(0)
        # print('count_match_iterate: ' + str(datetime.datetime.now() - starttime))
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

    # print('count_matches: ' + str(datetime.datetime.now() - headtime))
    return name, goods


def analyze(img_str):
    headtime = datetime.datetime.now()
    img_np = np.fromstring(img_str, np.uint8)

    # starttime = datetime.datetime.now()
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if RESIZE_FACTOR < 1.0:
        img = cv2.resize(img, None, fx=RESIZE_FACTOR,
                         fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)
    # print('imdecode: ' + str(datetime.datetime.now() - starttime))

    # starttime = datetime.datetime.now()
    matches = pool.starmap(count_matches, [(name, img) for name in candidates])
    # print('pool.starmap: ' + str(datetime.datetime.now() - starttime))

    # starttime = datetime.datetime.now()
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

    image = request.files.get('image')
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

    image.save('capture/', overwrite=True)

    img_str = image.file.read()
    result = analyze(img_str)
    return json.dumps(result)

    # starttime = datetime.datetime.now()
    # predicts = {}
    # t = read_tensor_from_image_str(img_str,
    #                                input_height=input_height,
    #                                input_width=input_width,
    #                                input_mean=input_mean,
    #                                input_std=input_std)
    # with tf.Session(graph=graph) as sess:
    #     objects = sess.run(output_operation.outputs[0],
    #                     {input_operation.outputs[0]: t})
    #     objects = np.squeeze(objects)
    #     top_k = objects.argsort()[-TOP_K:][::-1]
    #     for i in top_k:
    #         print(labels[i], objects[i])
    #         if objects[i] > TOP_THRESHOLD:
    #             predicts[labels[i]] = objects[i]
    # print(len(predicts))
    # print(datetime.datetime.now() - starttime)

    # with detection_graph.as_default():
    #     with tf.Session(graph=detection_graph) as sess:
    #         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    #         detection_boxes = detection_graph.get_tensor_by_name(
    #             'detection_boxes:0')
    #         detection_scores = detection_graph.get_tensor_by_name(
    #             'detection_scores:0')
    #         detection_classes = detection_graph.get_tensor_by_name(
    #             'detection_classes:0')
    #         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #         img = Image.open(image.file)
    #         img.thumbnail((300, 300), Image.BICUBIC)
    #         image_np = load_image_into_numpy_array(img)
    #         image_np_expanded = np.expand_dims(image_np, axis=0)
    #         (boxes, scores, classes, num) = sess.run(
    #             [detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    #         print(np.squeeze(boxes))
    #         print(np.squeeze(classes).astype(np.int32))
    #         print(np.squeeze(scores))
    #         print(num)
    #         vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
    #         plt.figure(figsize=(12, 8))
    #         plt.imshow(image_np)
    #         return 'content'

    # return 'error'


# def binary(pil_img):
#     thumbnail = pil_img.thumbnail((300, 300), Image.BICUBIC)
#     bgr = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)
#     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, np.array([0, 0, 46]), np.array([180, 43, 255]))
#     return mask


# @post('/freeze')
# def freeze():
#     ctrl = request.files.get('ctrl')
#     ctrl_pil = Image.open(ctrl.file)
#     ctrl_mask = binary(ctrl_pil)
#     cv2.namedWindow('ctrl_mask', cv.WINDOW_NORMAL)
#     cv2.imshow('ctrl_mask', ctrl_mask)
#     cv2.waitKey(0)
#     test = request.files.get('test')
#     test_pil = Image.open(test.file)
#     test_mask = binary(test_pil)
#     cv2.namedWindow('test_mask', cv.WINDOW_NORMAL)
#     cv2.imshow('test_mask', test_mask)
#     cv2.waitKey(0)


def test():
    with open('capture/2018_01_11_07_01_12.jpg', mode='rb') as f:
        analyze(f.read())


if __name__ == "__main__":
    # test()
    run(server='paste', host='0.0.0.0', port=8099, reloader=False)
