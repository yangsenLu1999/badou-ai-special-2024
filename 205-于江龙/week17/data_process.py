import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
import random

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs = glob.glob(os.path.join(data_path, "image", "*.png"))
    
    def augment(self, image, flipCode):
        # flipCode: 1 means flip horizontally, 0 means flip vertically, 
        # -1 means flip both horizontally and vertically
        return cv2.flip(image, flipCode)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        label_path = image_path.replace("image", "label")
        # get image and label
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255

        # data augmentation
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = MyDataset("205-于江龙/week17/data/train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (image, label) in enumerate(dataloader):
        print(image.shape, label.shape)
        break
