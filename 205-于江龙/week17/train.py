from model.unet_model import UNet
from data_process import MyDataset
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(net, device, data_path, epochs=40, batch_size=4, lr=1e-5):
    dataset = MyDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    pbar = tqdm(total=epochs, desc="epoch")
    for epoch in range(epochs):
        net.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            predict = net(image)
            loss_value = loss(predict, label)
            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(net.state_dict(), "best_model.pth")
            loss_value.backward()
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch}, Step: {i}, Loss: {loss_value.item():.4f}")
            pbar.refresh()
        pbar.update(1) 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(1, 1).to(device)
    train(net, device, "205-于江龙/week17/data/train")
