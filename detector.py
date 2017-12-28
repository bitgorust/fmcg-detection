import os
import sys
import numpy as np
import cv2 as cv


def h_range(bg_color='green'):
    lower = np.array([35, 43, 46])
    upper = np.array([99, 255, 255])
    if bg_color == 'white':
        lower = np.array([0, 0, 46])
        upper = np.array([180, 43, 255])
    return lower, upper


def get_mask(image, bg_color='green'):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h_lower, h_upper = h_range(bg_color)
    mask = cv.inRange(image, h_lower, h_upper)
    _, mask = cv.threshold(cv.GaussianBlur(mask, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY_INV)
    # return cv.Canny(mask, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    # return cv.dilate(cv.Canny(mask, 100, 200), kernel, iterations=5)
    # return cv.erode(mask, kernel, iterations=3)
    # return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return cv.dilate(mask, kernel, iterations=1)


def is_inside(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax1 >= bx1 and ax1 <= bx2 and ax2 >= bx1 and ax2 <= bx2 and ay1 >= by1 and ay1 <= by2 and ay2 >= by1 and ay2 <= by2:
        return True
    a_center_x, a_center_y = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    if a_center_x >= bx1 and a_center_x <= bx2 and a_center_y >= by1 and a_center_y <= by2:
        return True
    return False


def dilate(contours, extension=100):
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    boxes = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        x1, y1, x2, y2 = x - extension, y - \
            extension, x + w + extension, y + h + extension
        matched = False
        for i, box in enumerate(boxes):
            if is_inside((x1, y1, x2, y2), box):
                bx1, by1, bx2, by2 = box
                boxes[i] = (bx1 if bx1 < x1 else x1, by1 if by1 < y1 else y1,
                            bx2 if bx2 > x2 else x2, by2 if by2 > y2 else y2)
                matched = True
                break
        if not matched:
            boxes.append((x1, y1, x2, y2))
    return [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32) for x1, y1, x2, y2 in boxes]


def sub_images(image_path, output_dir, bg_color='green'):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image = cv.imread(image_path)
    height, width, depth = image.shape
    print(width, height, depth)

    # height1 = int(height / 3.0)
    # height2 = int(height * 2.0 / 3.0)
    # width1 = int(width / 3.0)
    # width2 = int(width * 2.0 / 3.0)

    # output_image = os.path.join(output_dir, image_name + '_91.jpg')
    # cv.imwrite(output_image, image[0:height1, 0:width1])
    # output_image = os.path.join(output_dir, image_name + '_92.jpg')
    # cv.imwrite(output_image, image[0:height1, width1:width2])
    # output_image = os.path.join(output_dir, image_name + '_93.jpg')
    # cv.imwrite(output_image, image[0:height1, width2:width])
    # output_image = os.path.join(output_dir, image_name + '_94.jpg')
    # cv.imwrite(output_image, image[height1:height2, 0:width1])
    # output_image = os.path.join(output_dir, image_name + '_95.jpg')
    # cv.imwrite(output_image, image[height1:height2, width1:width2])
    # output_image = os.path.join(output_dir, image_name + '_96.jpg')
    # cv.imwrite(output_image, image[height1:height2, width2:width])
    # output_image = os.path.join(output_dir, image_name + '_97.jpg')
    # cv.imwrite(output_image, image[height2:height, 0:width1])
    # output_image = os.path.join(output_dir, image_name + '_98.jpg')
    # cv.imwrite(output_image, image[height2:height, width1:width2])
    # output_image = os.path.join(output_dir, image_name + '_99.jpg')
    # cv.imwrite(output_image, image[height2:height, width2:width])

    mask = get_mask(image, bg_color)
    # cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    # cv.imshow(image_name, mask)
    # cv.waitKey(0)
    # return

    _, contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    len_contours = len(contours)
    print(str(len_contours) + ' contours found.')
    if len_contours == 0:
        return
    x1, y1, w, h = cv.boundingRect(contours[0])
    x2 = x1 + w
    y2 = y1 + h
    for i, c in enumerate(contours):
        if i == 0:
            continue
        x, y, w, h = cv.boundingRect(c)
        x1 = min(x, x1)
        y1 = min(y, y1)
        x2 = max(x + w, x2)
        y2 = max(y + h, y2)

    cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.namedWindow('mask', cv.WINDOW_NORMAL)
    cv.imshow('mask', mask)
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.imshow(image_name, image)
    cv.waitKey(0)
    return

    contours = dilate(contours, 100)
    len_contours = len(contours)
    print(str(len_contours) + ' contours found.')
    if len_contours == 0:
        return

    # image_dir = os.path.join(output_dir, image_name)
    # if not os.path.exists(image_dir):
    #     os.makedirs(image_dir)
    for i, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)
        if x < 0:
            w += x
            x = 0
        if w > width:
            w = width
        if y < 0:
            h += y
            y = 0
        if h > height:
            h = height
        print(x, y, w, h)
        output_image = os.path.join(
            output_dir, image_name + '_' + str(i) + '.jpg')
        cv.imwrite(output_image, image[y:y + h, x:x + w])

    output_image = os.path.join(output_dir, image_name + '.jpg')
    cv.imwrite(output_image, image)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('marked', image)

    cv.waitKey(0)


if __name__ == '__main__':
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    for file in os.listdir(image_dir):
        if file != 'IMG_20171226_191405.jpg':
            continue
        image_path = os.path.join(image_dir, file)
        sub_images(image_path, output_dir, 'green')
