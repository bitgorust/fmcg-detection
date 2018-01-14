import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
from bottle import run, post, request

sys.path.append("d:/tensorflow/models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util


if tf.__version__ != '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PATH_TO_CKPT = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'
NUM_CLASSES = 143

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


@post('/detect')
def detect():
    image = request.files.get('image')
    # _, ext = os.path.splitext(image.filename)
    # if ext not in ('.png', '.jpg', '.jpeg'):
    #     return 'not allowed'

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            img = Image.open(image.file)
            img.thumbnail((300, 300), Image.BICUBIC)
            image_np = load_image_into_numpy_array(img)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            print(np.squeeze(boxes))
            print(np.squeeze(classes).astype(np.int32))
            print(np.squeeze(scores))
            print(num)
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            return 'content'

    return 'error'


def binary(pil_img):
    thumbnail = pil_img.thumbnail((300, 300), Image.BICUBIC)
    bgr = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 46]), np.array([180, 43, 255]))
    return mask
    # _, mask = cv2.threshold(cv2.GaussianBlur(mask, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)


@post('/freeze')
def freeze():
    ctrl = request.files.get('ctrl')
    ctrl_pil = Image.open(ctrl.file)
    ctrl_mask = binary(ctrl_pil)
    cv2.namedWindow('ctrl_mask', cv.WINDOW_NORMAL)
    cv2.imshow('ctrl_mask', ctrl_mask)
    cv2.waitKey(0)
    test = request.files.get('test')
    test_pil = Image.open(test.file)
    test_mask = binary(test_pil)
    cv2.namedWindow('test_mask', cv.WINDOW_NORMAL)
    cv2.imshow('test_mask', test_mask)
    cv2.waitKey(0)


if __name__ == "__main__":
    run(server='paste', host='localhost', port=8099, reloader=True)
