from src.utils import *

def train(model, optimizer, data_loader, device):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc="Training Epoch"):
        images = torch.stack([image.to(device) for image in images])
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        for target in targets:
            target['bbox'] = target.pop('boxes')
            target['label'] = target.pop('labels')
        

        optimizer.zero_grad()
        outputs = model(images)

        pred_tensor = outputs
        

        loss = criterion(pred_tensor,target_tensor)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)