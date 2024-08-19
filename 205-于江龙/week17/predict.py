import glob
import numpy as np
import torch
import cv2
from model.unet_model import UNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_imgs = glob.glob("205-于江龙/week17/data/test/image/*.png")
    for test_path in test_imgs[:5]:
        save_output_path = test_path.split(".")[0] + "_output.png"
        image = cv2.imread(test_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = torch.from_numpy(image).to(device, dtype=torch.float32)
        pred = model(image).cpu().detach().numpy()
        pred[pred > 0.5] = 255
        pred[pred <= 0.5] = 0
        # pred: [batch_size, channel, height, width]
        cv2.imwrite(save_output_path, pred[0][0])
        