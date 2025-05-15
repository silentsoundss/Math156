import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class YOLOv1(nn.Module):
    def __init__(self, S=13, B=1, C=1):
        super(YOLOv1, self).__init__()
        self.S = S  #Grid size
        self.B = B  #boxes
        self.C = C  #classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 208x208

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 104x104

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 52x52

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 26x26

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 13x13
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.S * self.S, 4096), nn.ReLU(),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


def yolo_loss(pred, target):
    lambda_coord = 5
    lambda_noobj = 0.5

    obj_mask = target[..., 4] == 1
    noobj_mask = target[..., 4] == 0

    mse = nn.MSELoss(reduction='sum')

    loss_coord = lambda_coord * mse(pred[obj_mask][..., :4], target[obj_mask][..., :4])
    loss_obj = mse(pred[obj_mask][..., 4], target[obj_mask][..., 4])
    loss_noobj = lambda_noobj * mse(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])
    loss_class = mse(pred[obj_mask][..., 5:], target[obj_mask][..., 5:])

    total_loss = loss_coord + loss_obj + loss_noobj + loss_class
    return total_loss


def train(model, dataloader, optimizer, epochs=5):
    model.train() 
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()            
            outputs = model(images)            
            loss = yolo_loss(outputs, targets) 
            loss.backward()                    
            optimizer.step()                  
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def detect(model, image_tensor, conf_thresh=0.5):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        output = output.squeeze(0).cpu() 
        detections = []
        S = 13
        for i in range(S):
            for j in range(S):
                cell = output[i, j]
                x, y, w, h, conf, cls_score = cell
                if conf > conf_thresh:
                    box_x = (j + x.item()) / S * 416
                    box_y = (i + y.item()) / S * 416
                    box_w = w.item() * 416
                    box_h = h.item() * 416
                    detections.append([box_x, box_y, box_w, box_h, conf.item()])
        return detections

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    dataset = putdatasethere(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, dataloader, optimizer, epochs=10)
    test_img, _ = dataset[0]
    boxes = detect(model, test_img, conf_thresh=0.3)
    print("Detections:", boxes)