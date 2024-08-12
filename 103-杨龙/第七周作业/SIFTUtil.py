# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import numpy as np


def key_points_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  # 4.4.0以上版本，可以直接这样用sift。不需要cv2.xfeatures2d
    keypoints, descriptor = sift.detectAndCompute(gray_image, None)
    result = cv2.drawKeypoints(image=image, outImage=image, keypoints=keypoints,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))
    return result


def draw_matchs_knn(image_path1, image_path2):
    image1, image2 = __get_images(image_path1, image_path2)
    detect_result1, detect_result2 = __detect_images(image1, image2)
    print(detect_result1[0])
    print(detect_result2[0])
    good_matchs = __get_good_matchs(detect_result1[1], detect_result2[1], 2)
    __draw_matchs(image1, detect_result1[0], image2, detect_result2[0], good_matchs[:20])


def __get_images(image_path1, image_path2):
    return cv2.imread(image_path1), cv2.imread(image_path2)


def __detect_images(image1, image2):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image1, None), sift.detectAndCompute(image2, None)


def __get_good_matchs(des1, des2, k):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            good_matches.append(m)
    return good_matches


def __draw_matchs(image1, kp1, image2, kp2, good_matchs):
    vis = __get_vis(image1, image2)
    post1, post2 = __get_posts(kp1, kp2, good_matchs, image1.shape[1])
    __draw_line(vis, post1, post2)
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def __get_vis(image1, image2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2
    return vis


def __get_posts(kp1, kp2, good_matchs, w1):
    p1 = [kpp.queryIdx for kpp in good_matchs]
    p2 = [kpp.trainIdx for kpp in good_matchs]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    return post1, post2


def __draw_line(vis, post1, post2):
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
