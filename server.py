import os
import json
import datetime
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import cv2
from bottle import run, get, post, request, response
import pprint
import logging
import pyzbar.pyzbar as pyzbar

pp = pprint.PrettyPrinter(indent=2)
logging.basicConfig(filename='debug.log', level=logging.DEBUG)
pool = ThreadPool()

STEP = 25
MAXA = 90


def decode(im, angle):
    if (angle > 0):
        matrix = cv2.getRotationMatrix2D(center=(im.shape[1] / 2, im.shape[0] / 2), angle=angle, scale=1)
        im = cv2.warpAffine(im, matrix, (im.shape[0] * 2, im.shape[1] * 2))
    barcodes = set()
    for obj in pyzbar.decode(im, symbols=[pyzbar.ZBarSymbol.CODE128]):
        barcodes.add(obj.data.decode())
    return barcodes


def analyze(img_str):
    headtime = datetime.datetime.now()
    img_np = np.fromstring(img_str, np.uint8)

    im = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    results = pool.starmap(decode, [(im, angle) for angle in range(0, MAXA, STEP)])
    logging.debug(pp.pformat(results))

    barcodes = set()
    for result in results:
        barcodes |= result
    barcodes = list(barcodes)
    logging.debug(pp.pformat(barcodes))

    logging.debug(datetime.datetime.now() - headtime)
    return barcodes


@get('/hello')
def hello():
    return 'welcome'


@post('/detect')
def detect():
    response.set_header('Content-Type', 'application/json; charset=utf-8')

    for key in request.headers:
        logging.debug(key + ': ' + request.get_header(key))

    result = {}

    starttime = datetime.datetime.now()
    logging.debug('getting image')
    image = request.files.get('image')
    logging.debug('image got: ' + str(datetime.datetime.now() - starttime))
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
    logging.debug('saving ' + image.filename)
    image.save('capture/', overwrite=True)
    logging.debug('finish saving: ' + str(datetime.datetime.now() - starttime))

    starttime = datetime.datetime.now()
    logging.debug('reading ' + image.filename)
    img_str = image.file.read()
    logging.debug('finish reading: ' + str(datetime.datetime.now() - starttime))
    result = analyze(img_str)
    return json.dumps(result)


if __name__ == "__main__":
    run(server='paste', host='0.0.0.0', port=8099)
