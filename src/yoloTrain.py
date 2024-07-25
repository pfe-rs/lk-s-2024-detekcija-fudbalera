from src.utils import *
import torch
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train(model, optimizer, data_loader:DataLoader, device, criterion):
    model.train()
    epoch_loss = 0


    # iterate through the data loader
    for images, targets in tqdm(data_loader, desc="Training Epoch"):
        images = torch.stack([image.to(device) for image in images])

        optimizer.zero_grad()
        outputs = model(images)

        # Pozovi YOLOv8 gubitak funkciju
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)
