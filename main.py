import argparse
import cv2
import numpy as np


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Ishihara edge detection')
    parser.add_argument('--input', default='testcases/Ishihara_23.png')
    parser.add_argument('--output', default='outputs/Ishihara_23.png')
    args = parser.parse_args()

    # Read
    image = cv2.imread(args.input)
    assert image is not None, f'Failed to read {args.input}'

    # Median blur
    image = cv2.medianBlur(image, 5)

    # HSV and K-means
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hue = np.float32(image.reshape((-1, 3)))[:, 0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(hue, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = np.uint8(labels.reshape(image.shape[:-1] + (1, )))
    if labels[0, 0, 0] == 0:
        labels ^= 1
    image = image * labels

    # Finish
    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.imwrite(args.output, image)
