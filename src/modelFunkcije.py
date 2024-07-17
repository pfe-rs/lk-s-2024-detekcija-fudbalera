import matplotlib.pyplot as plt
import torchvision
import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Funkcija za kreiranje modela
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Trening funkcija
def train_model(model, data_loader, device, num_epochs):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {epoch_loss}")

#plotovanje slika
def plot_image_with_boxes(image, boxes, labels, scores, CLASSES):
    img = image.permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # ovaj threshold moze da se namesta
            if label < len(CLASSES):  # provera da li je label u opsegu
                box = box.cpu().numpy().astype(int)
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                label_str = CLASSES[label]
                img = cv2.putText(img, f'{label_str}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                print(f"Label index {label} out of range for CLASSES list.")
                print(f"Labels: {labels}")
                print(f"Classes length: {len(CLASSES)}")

    plt.imshow(img)
    plt.axis('off')
    plt.show()