from modules import model
from torchsummary import summary
from modules.transfer_learning import *
from modules.utils import *

def detect(img_path):
    # 加载图片
    src_img_arr = cv2.imread(img_path)
    img = cv2.resize(src_img_arr, dsize=INPUT_SHAPE[:2])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255., (2, 0, 1))  # c, h, w

    with torch.no_grad(): # 临时关闭对张量的自动微分
        # 创建模型
        yolo = model.YoloV3(in_channels=3, num_achors_per_grid=3, num_cls=NUM_CLASS).to(DEVICE)
        # 生成输入
        x = torch.from_numpy(img).type(torch.float32).unsqueeze(dim=0).to(DEVICE)
        # 加载参数
        load_pretrained_model(yolo, PATH_LAST_MODEL)
        # yolo正向传播
        yolo_out = yolo(x)
        # 将yolo正向传播结果转换为预测结果
        y_pred = yolo_out_sigmoid(yolo_out)
        # 对预测结果解码
        boxes, scores, classes = decode_yolo_out(y_pred, BASE_ACHORS, INPUT_SHAPE)
        # 对预测结果进行筛选
        boxes, scores, classes = filter_prediction(boxes, scores, classes, NUM_CLASS, SCORE_THRESHOLD, IOU_THRESHOLD)

        img_with_bndbox = None
        if boxes is None:
            print("没有检测到目标！")
        else:
            # 将预测框映射回原图
            boxes = get_real_box(boxes, src_img_size=src_img_arr.shape[:2], input_img_size=INPUT_SHAPE[:2])
            # 将预测框画到原图上
            img_with_bndbox = draw_bndbox_on_img(src_img_arr, boxes.tolist(), scores.tolist(), classes.tolist())
            display_img(img_with_bndbox)
        return img_with_bndbox


if __name__ == "__main__":
    img_path = os.path.join(PATH_SAMPLES, "img4.jpg")
    resutl_img = detect(img_path)

    if resutl_img is not None:
        save_path = os.path.join(PATH_SAMPLES,"result4.jpg")
        cv2.imwrite(save_path, resutl_img)



