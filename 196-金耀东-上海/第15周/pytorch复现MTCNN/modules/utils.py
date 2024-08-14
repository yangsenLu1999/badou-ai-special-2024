import torch,cv2
import numpy as np
from torchvision.ops import nms
from modules.CONST import STRIDE_PNET, CELL_SIZE, MIN_FACE_SIZE, IOU_THRESHOLD

def img_preprocessing(img):
    img = img.astype(np.float32)
    return (img-127.5) / 128

def get_bndboxes_pnet(scores, bbox_regress, scale, score_thres=0.9, stride=STRIDE_PNET, cellsize=CELL_SIZE):
    # 仅保留置信度大于阈值的结果
    indx = torch.where(scores>score_thres)
    if not len(indx[0]) > 0:
        return None, None, None
    scores = scores[indx]
    bbox_regress = bbox_regress[indx]

    # 构造网格坐标，shape:(h, w, 2)
    grid_xy = torch.cat([ indx[1].reshape(-1,1), indx[0].reshape(-1,1)], dim=-1)

    # 计算bboxes
    xy_min = ( grid_xy * stride ) / scale
    xy_max = ( grid_xy * stride + cellsize-1) / scale
    bboxes = torch.cat([xy_min, xy_max], dim=-1)

    return bboxes.reshape(-1, 4), scores.reshape(-1), bbox_regress.reshape(-1, 4)

def filter_bndbox(bboxes, scores, score_thres, size_thres=MIN_FACE_SIZE):
    # 第一轮筛选：框（人脸）尺寸>阈值
    keep = (bboxes[:, 3] - bboxes[:, 1] + 1) > size_thres[0]
    bboxes, scores = bboxes[keep], scores[keep]
    keep = (bboxes[:, 2] - bboxes[:, 0] + 1) > size_thres[1]
    bboxes, scores = bboxes[keep], scores[keep]

    # 第二轮筛选：置信度>阈值
    keep = scores > score_thres
    bboxes, scores = bboxes[keep] , scores[keep]

    # 第三轮筛选：非极大值抑制
    keep = nms(bboxes[:,:4], scores, IOU_THRESHOLD)
    bboxes = bboxes[keep]
    if not len(bboxes) > 0:
        return None
    return bboxes

def correct_bndbox(bboxes, x_max, y_max):
    # 修正坐标，小于0的部分变为0
    bboxes = torch.where(bboxes>0, bboxes, 0)

    # 修正坐标，大于max的部分变为max
    max = torch.tensor([x_max, y_max, x_max, y_max]).unsqueeze(0)
    bboxes[:,:4] = torch.where(bboxes[:,:4]<max, bboxes[:,:4], max)
    return bboxes

def crop_target_and_resize(img, boxes, dst_size): #imgs:(h,w,c), boxes:(num_bbx, 4) [x0, y0, x1, y1]
    crop_imgs = []
    for box in boxes:
        crop_img = img[ int(box[1]):int(box[3]), int(box[0]):int(box[2]), : ]
        h, w = crop_img.shape[:2]
        long_side = max(h,w)
        h_pad = (long_side - h) // 2
        w_pad = (long_side - w) // 2
        crop_img = cv2.copyMakeBorder(crop_img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=0)
        crop_imgs.append(cv2.resize(crop_img, dst_size, interpolation=cv2.INTER_LINEAR))
    return np.stack(crop_imgs, axis=0)


def bndbox_regress(bboxes, bbox_regress, lndmsk_regress=None):
    wh = bboxes[:, [2, 3]] - bboxes[:, [0, 1]] + 1
    if not lndmsk_regress is None:
        lndmsk = []
        for i in range(5):
            point = bboxes[..., :2] + lndmsk_regress[..., 2*i: 2*i+2] * wh
            lndmsk.append(point)
        bboxes = torch.cat([bboxes, *lndmsk], dim=-1)

    bboxes[:, [0, 1]] += (bbox_regress[:, [0, 1]] * wh)
    bboxes[:, [2, 3]] += (bbox_regress[:, [2, 3]] * wh)
    return bboxes

def draw_bndboxes_on_img(img, bndboxes, has_lndmsk=True):
    color = (0,0,255)
    thic = 1
    for bbox in bndboxes:
        cv2.rectangle(img, bbox[0:2], bbox[2:4], color, thic)
        if has_lndmsk:
            for i in range(5):
                cv2.circle(img, bbox[ i*2+4: i*2+6], 1, color, thic)
    return img

def display_img(img, winname="result"):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









