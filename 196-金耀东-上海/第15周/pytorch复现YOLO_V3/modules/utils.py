import cv2
from torch import Tensor
import torch
from torchvision.ops import nms
from modules.CONST import COCO_CLASSES

def yolo_out_sigmoid(yolo_out, num_anchor_per_grid=3, num_class=80):
    # ------------------------------------------------------#
    #   对yolo输出结果进行处理
    #   使用sigmoid函数对tx, tw, conf, classes进行归一化
    # ------------------------------------------------------#
    out = []
    for ele in yolo_out:
        b , c , h, w = ele.shape
        ele = ele.view(-1, num_anchor_per_grid, (num_class+5), h, w).permute(0, 1, 3, 4, 2) # b, 3, 13, 13, 5+80
        ele[...,[0,1,4]] = ele[...,[0,1,4]].sigmoid() # tx, ty, confidence
        ele[..., 5:] = ele[..., 5:].sigmoid() # classes
        out.append(ele)
    return out

def decode_yolo_out(yolo_out, base_anchors, input_shape):
    # ------------------------------------------------------#
    #   对yolo输出进行解码
    #   获取boxes,scores,classes
    # ------------------------------------------------------#
    boxes, scores, classes = [], [], []
    for l, ele in enumerate(yolo_out):
        batch_size, num_anchor_per_grid, num_grid_h, num_grid_w, _= ele.shape
        anchors_tensor = Tensor(base_anchors[l]).reshape(1,num_anchor_per_grid, 1, 1, -1)

        tx = ele[..., 0]
        ty = ele[..., 1]
        tw = ele[..., 2]
        th = ele[..., 3]
        conf = ele[..., 4]
        p_cls, cls = torch.max(ele[..., 5:], dim=-1)

        # 生成每个网格的左上角坐标
        grid_x = Tensor(range(num_grid_w)).reshape(1, num_grid_w).tile(num_grid_h, 1)
        grid_y = Tensor(range(num_grid_h)).reshape(num_grid_h, 1).tile(1, num_grid_w)
        grid_x = grid_x.reshape(1,1,num_grid_h, num_grid_w)
        grid_y = grid_y.reshape(1,1,num_grid_h, num_grid_w)

        # 计算box中心点坐标（以input_shape为参照）
        x = ((tx + grid_x) * (input_shape[1] / num_grid_w)).unsqueeze(dim=-1)
        y = ((ty + grid_y) * (input_shape[0] / num_grid_h)).unsqueeze(dim=-1)

        # 计算box中心点坐标（以input_shape为参照）
        w = (torch.exp(tw) * anchors_tensor[...,0]).unsqueeze(dim=-1)
        h = (torch.exp(th) * anchors_tensor[...,1]).unsqueeze(dim=-1)

        # 计算box左上/右下坐标
        min_x = x - w/2
        min_y = y - h/2
        max_x = x + w/2
        max_y = y + h/2

        boxes.append(torch.cat([min_x,min_y,max_x,max_y],dim=-1).reshape(-1, 4))
        scores.append((conf*p_cls).reshape(-1))
        classes.append(cls.reshape(-1))

    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)
    classes = torch.cat(classes, dim=0)
    return boxes, scores, classes

def filter_prediction(boxes, scores, classes, num_class, score_threshold=0.5, iou_threshold=0.5):
    # ------------------------------------------------------#
    #   对候选框进行筛选
    # ------------------------------------------------------#
    # 第一轮筛选：保留score>【阈值】的预测
    keep = scores>score_threshold
    boxes , scores , classes = boxes[keep,...] , scores[keep,...] , classes[keep,...]

    # 第二轮筛选：非极大值抑制
    boxes_out, scores_out, classes_out = [], [], []
    for cls_i in range(num_class):
        mask_clsi = (classes==cls_i)
        boxes_clsi, scores_clsi, classes_clsi = boxes[mask_clsi] , scores[mask_clsi] , classes[mask_clsi]
        if not boxes_clsi.shape[0] > 0:
            continue
        keep_clsi = nms(boxes_clsi, scores_clsi, iou_threshold)
        boxes_out.append(boxes_clsi[keep_clsi])
        scores_out.append(scores_clsi[keep_clsi])
        classes_out.append(classes_clsi[keep_clsi])
    if not len(boxes_out) > 0:
        return None, None, None
    return torch.cat(boxes_out) , torch.cat(scores_out) , torch.cat(classes_out)

def get_real_box(boxes, src_img_size, input_img_size):
    # ------------------------------------------------------#
    #   获取真实尺寸的标定框
    # ------------------------------------------------------#
    scal_h = src_img_size[0] / input_img_size[0]
    scal_w = src_img_size[1] / input_img_size[1]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scal_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scal_h
    return boxes.to(torch.int)

def draw_bndbox_on_img(img, boxes, scores, classes):
    # ------------------------------------------------------#
    #   将标定框画回到原图
    # ------------------------------------------------------#
    img_h, img_w, _ = img.shape
    for bndbox, score, cls in zip(boxes, scores, classes):
        label = f"{COCO_CLASSES[cls]}:{score:.2f}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX # 标签字体
        font_scal = 0.7 # 字体缩放比例
        font_thic = 1 # 字体粗细
        (font_w, font_h), baseline = cv2.getTextSize(label, font_face, font_scal, font_thic)
        font_org = bndbox[0], bndbox[1]-baseline # 标签文本起始位置：左下坐标
        labelbox = bndbox[0], bndbox[1]-font_h-baseline, bndbox[0]+font_w, bndbox[1] # 标签框坐标

        # 画标定框
        cv2.rectangle(img, bndbox[:2], bndbox[2:4], (0,0,255), 1)
        # 画标签框
        cv2.rectangle(img, labelbox[:2], labelbox[2:4], (0,0,255), -1)
        # 画标签
        cv2.putText(img, label, font_org, font_face, font_scal, (255,255,255), font_thic)
    return img

def display_img(img, winname="result"):
    # ------------------------------------------------------#
    #   展示图像
    # ------------------------------------------------------#
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


