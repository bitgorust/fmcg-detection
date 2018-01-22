import os
import sys
import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def save_stitched(images, output_file):
    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch(tuple(images))
    cv2.imwrite(output_file, result[1])


if __name__ == "__main__":
    images_dir = sys.argv[1]
    output_file = sys.argv[2]
    images = []
    for image in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image)
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(cv2.imread(image_path))
    save_stitched(images, output_file)