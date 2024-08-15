import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

test_imag = 'images/demo.jpg'
imread = cv2.imread(test_imag)

candidate, subset = body_estimation(imread)
copy_imread = copy.deepcopy(imread)
bodypose = util.draw_bodypose(copy_imread, candidate, subset)
hand_detect = util.handDetect(candidate, subset, imread)

all_hand_peaks = []
for x, y, w, is_left in hand_detect:

    peaks = hand_estimation(imread[y:y+w, x:x+w, :])

    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)

    all_hand_peaks.append(peaks)

hand_handpose = util.draw_handpose(bodypose, all_hand_peaks)
plt.imshow(hand_handpose[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

