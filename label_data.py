import os
import sys
import numpy as np
import cv2 as cv

sys.path.append("d:/tensorflow/models/research/object_detection")
from utils import label_map_util


def entity_dirs(origin_dir, offset=0, length=1000):
    dirs = []
    entities = sorted(os.listdir(origin_dir))
    print(len(entities), offset, length)
    if len(entities) <= offset:
        return dirs
    for entity in entities[offset:offset + length] if offset + length <= len(entities) else entities[offset:]:
        entity_path = os.path.join(origin_dir, entity)
        if os.path.isdir(entity_path) and not entity.startswith('.'):
            dirs.append(entity)
    return dirs


def entity_images(entity_dir):
    images = []
    for entity in os.listdir(entity_dir):
        entity_path = os.path.join(entity_dir, entity)
        if os.path.isfile(entity_path) and entity_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(entity)
    return images


def crop(image):
    return image[0:1000, 300:1960]


def h_range(bg_color='green'):
    lower = np.array([35, 43, 46])
    upper = np.array([99, 255, 255])
    return lower, upper


def get_mask(image, bg_color='green'):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h_lower, h_upper = h_range(bg_color)
    mask = cv.inRange(image, h_lower, h_upper)
    _, mask = cv.threshold(mask, 128, 255, 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 3))
    return cv.dilate(mask, kernel, iterations=1)


def find_if_close(cnt1, cnt2, dist_threshold=140):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < dist_threshold:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def unify(contours, dist_threshold=140):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2, dist_threshold)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1
    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv.convexHull(cont)
            unified.append(hull)
    return unified


def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)


def rect_overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return range_overlap(ax1, ax2, bx1, bx2) and range_overlap(ay1, ay2, by1, by2)


def get_grouped_rects(contours, reversed=True):
    contours = sorted(contours, key=cv.contourArea, reverse=reversed)
    boxes = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    clusters = []
    for box in boxes:
        matched = False
        x1, y1, x2, y2 = box
        for i, cluster in enumerate(clusters):
            cx1, cy1, cx2, cy2 = cluster
            if rect_overlaps(box, cluster):
                matched = True
                clusters[i] = (min(cx1, x1), min(cy1, y1),
                               max(cx2, x2), max(cy2, y2))
        if not matched:
            clusters.append(box)
    return clusters

def get_roi(image, bg_color='green'):
    mask = get_mask(image, bg_color)

    _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    len_contours = len(contours)
    print(str(len_contours) + ' contours found.')
    if len_contours == 0:
        return None

    unified = unify(contours)
    len_unified = len(unified)
    print(str(len_unified) + ' unified found.')
    if len_unified == 0:
        return None

    rects = get_grouped_rects(unified)
    len_rects = len(rects)
    print(str(len_rects) + ' rects found.')
    if len_rects == 0:
        return None

    return rects[0]

def get_annotation_xml(fullpath, width, height, name, rect, depth=3):
    folder_full_path, filename = os.path.split(fullpath)
    folder = os.path.basename(folder_full_path)
    xmin, ymin, xmax, ymax = rect
    return """<annotation>
	<folder>{folder}</folder>
	<filename>{filename}</filename>
	<path>{fullpath}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>{depth}</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>{name}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
</annotation>""".format(**{
        'folder': folder,
        'filename': filename,
        'fullpath': fullpath,
        'width': width,
        'height': height,
        'depth': depth,
        'name': name,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax
    })


def serialize_to_pbtxt(filepath, label_map={}):
    with open(filepath, 'w') as f:
        for label_name, label_id in sorted(label_map.items(), key=lambda kv: (-kv[1], kv[0]), reverse=True):
            print(label_name, label_id)
            f.write('item {\n  id: ' + str(label_id) + ',\n  name: \'' + str(label_name) + '\'\n}\n\n')


def annotate_images(origin_dir, output_dir, offset=0, length=1000, bg_color='green'):
    entities = entity_dirs(origin_dir, offset, length)
    len_entities = len(entities)
    print(str(len_entities) + ' entities found.')
    if len_entities == 0:
        return

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    annotations_dir = os.path.join(output_dir, 'annotations')
    xmls_dir = os.path.join(annotations_dir, 'xmls')
    if not os.path.exists(xmls_dir):
        os.makedirs(xmls_dir)

    id_index = 0
    for entity_id in entities:
        entity_path = os.path.join(origin_dir, entity_id)
        print('processing ' + entity_id + ': ' + entity_path)
        images = entity_images(entity_path)
        len_images = len(images)
        print(str(len_images) + ' images found.')
        if len_images == 0:
            continue

        for i, file in enumerate(images):
            image_path = os.path.join(entity_path, file)
            print('processing ' + file + '(' + str(i) +
                  '/' + str(len_images) + '): ' + image_path)
            image = cv.imread(image_path)
            image = crop(image)
            height, width, depth = image.shape
            print(height, width, depth)

            rect = get_roi(image, bg_color)
            if not rect:
                continue

            xmin, ymin, xmax, ymax = rect
            print('xmin', xmin, 'ymin', ymin, 'xmax', xmax, 'ymax', ymax)

            output_image = os.path.join(
                images_dir, entity_id + '_' + str(i) + '.jpg')
            cv.imwrite(output_image, image)

            annotation_file = os.path.join(
                xmls_dir, entity_id + '_' + str(i) + '.xml')
            with open(annotation_file, 'w') as f:
                f.write(get_annotation_xml(output_image, width,
                                           height, entity_id, rect, depth))


def serialize_labels(label_map_path, images_dir):
    label_map_dict = {}
    if os.path.exists(label_map_path):
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    images = os.listdir(images_dir)
    labels = set()
    for image in images:
        label = image.split('_')[0]
        labels.add(label)
    labels = sorted(list(labels))
    for label_name in labels:
        if label_name not in label_map_dict:
            label_id = len(label_map_dict) + 1
            label_map_dict[label_name] = label_id
    serialize_to_pbtxt(label_map_path, label_map_dict)


if __name__ == '__main__':
    origin_dir = sys.argv[1]
    output_dir = sys.argv[2]
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    length = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    bg_color = sys.argv[5] if len(sys.argv) > 5 else 'green'

    annotate_images(origin_dir, output_dir, offset, length, bg_color)

    label_map_path = os.path.join(output_dir, 'label_map.pbtxt')
    images_dir = os.path.join(output_dir, 'images')
    serialize_labels(label_map_path, images_dir)

    # for image in os.listdir(origin_dir):
    #     if image.startswith('.'):
    #         continue
    #     image_path = origin_dir + '/' + image
    #     img = cv.imread(image_path)
    #     img = crop(img)
    #     rect = get_roi(img, bg_color)
    #     if not rect:
    #         continue
    #     x1, y1, x2, y2 = rect
    #     cv.imwrite(output_dir + '/' + image, img[y1:y2, x1:x2])
