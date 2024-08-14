from torch import softmax
from modules.CONST import DEVICE, SCALE_FACTOR, MIN_SIZE
from modules.transfer_learning import load_pretrained_weight
from modules.utils import *

def get_model(net, weight_path, device=DEVICE):
    model = net().to(device)
    if (not load_pretrained_weight(model, weight_path, device=DEVICE)):
        raise Exception("failed to load model!")
    return model

def predict(model, x):
    x = torch.from_numpy(x).permute(0, 3, 1, 2).type(torch.float32).to(DEVICE)
    cls, bbox_regress, lndmsk_regress = model(x)
    scores = softmax(cls, dim=1)[:,1]
    return scores, bbox_regress, lndmsk_regress

def detect_face_region(model, img): # Pnet
    all_bboxes, all_scores, all_bbox_regress = [] , [], []
    current_img = img_preprocessing(img)
    while not min(current_img.shape[0], current_img.shape[1]) < MIN_SIZE: # 图像金字塔
        x = np.expand_dims(current_img, axis=0)
        scores, bbox_regress, _ = predict(model, x)
        scores, bbox_regress = scores.squeeze() , bbox_regress.permute(0, 2, 3, 1).squeeze()
        bboxes, scores, bbox_regress = get_bndboxes_pnet(scores, bbox_regress, scale=current_img.shape[0]/img.shape[0])
        if not bboxes is None:
            all_bboxes.append(bboxes) , all_scores.append(scores) , all_bbox_regress.append(bbox_regress)
        dst_size = ( int(current_img.shape[1] * SCALE_FACTOR) , int(current_img.shape[0] * SCALE_FACTOR) )
        current_img = cv2.resize(current_img, dst_size, interpolation=cv2.INTER_LINEAR)

    all_bboxes = torch.cat(all_bboxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_bbox_regress = torch.cat(all_bbox_regress, dim=0)
    all_bboxes = bndbox_regress(all_bboxes,all_bbox_regress)
    if not len(all_bboxes) > 0:
        return None

    return all_bboxes, all_scores

def detect_face(model, imgs, bboxes): # Rnet & Onet
    imgs = img_preprocessing(imgs)
    scores, bbox_regress, lndmsk_regress = predict(model, imgs)
    bboxes = bndbox_regress(bboxes, bbox_regress, lndmsk_regress)
    return bboxes, scores

def detect(model, img_input, bboxes=None):
    if bboxes is None:
        return detect_face_region(model, img_input)
    return detect_face(model, img_input, bboxes)