from src.utils import *
import torch
from tqdm import tqdm
import torchvision.models as models

def train(model, optimizer, data_loader, device, criterion):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc="Training Epoch"):
        images = torch.stack([image.to(device) for image in images])
        print(images.shape)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)

        # Pozovi YOLOv8 gubitak funkciju
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


