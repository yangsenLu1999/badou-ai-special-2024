import torch, cv2, os
from modules.CONST import *
from modules.models import PNet, RNet, ONet
from modules.detect import *
from modules.utils import draw_bndboxes_on_img, display_img

def run_detect_face(img):
    src_h, src_w = img.shape[:2]
    nets = [PNet, RNet, ONet]
    weight_paths = [PATH_PNET_WEIGHT, PATH_RNET_WEIGHT, PATH_ONET_WEIGHT]
    score_thresholds = [SCORE_THRES_PNET, SCORE_THRES_RNET, SCORE_THRES_ONET]
    img_sizes = [IMG_SIZE_RNET, IMG_SIZE_ONET]

    img_input, bboxes = img, None
    with torch.no_grad():
        for i in range(3):
            model = get_model(nets[i], weight_paths[i])
            bboxes, scores = detect(model, img_input, bboxes)
            bboxes = correct_bndbox(bboxes, x_max=src_w - 1, y_max=src_h - 1)
            bboxes = filter_bndbox(bboxes, scores, score_thres=score_thresholds[i])
            if bboxes is None:
                return None
            if not nets[i] is ONet:
                img_input = crop_target_and_resize(img, bboxes, dst_size=img_sizes[i])
        bboxes = bboxes.type(torch.int).tolist()
    return bboxes

if __name__ == "__main__":
    img_path = os.path.join(PATH_SAMPLES, "img.jpg")
    img = cv2.imread(img_path)
    bboxes = run_detect_face(img)
    if not bboxes is None:
        result_img = draw_bndboxes_on_img(img, bboxes)
        display_img(result_img)
        save_path = os.path.join(PATH_SAMPLES, "result.jpg")
        cv2.imwrite(save_path, result_img)

