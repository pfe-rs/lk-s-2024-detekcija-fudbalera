import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Funkcija koja cita anotacije iz XML-a
def read_voc_annotations(xml_path, CLASSES):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in CLASSES:
            continue
        label_idx = CLASSES.index(label)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        if xmax <= xmin or ymax <= ymin:
            print(f"Skipping invalid bounding box: {[xmin, ymin, xmax, ymax]}")
            continue

        box = [xmin, ymin, xmax, ymax]
        boxes.append(box)
        labels.append(label_idx)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    return boxes, labels

#ucitavanje slika i njihovih anotacija
def load_image_and_annotations(img_path, xml_path, CLASSES, transform=None):
    # ucitamo sliku
    img = Image.open(img_path).convert("RGB")

    # ucitamo anotaciju
    boxes, labels = read_voc_annotations(xml_path, CLASSES)

    # Primenimo transformations ako ih ima
    if transform:
        img = transform(img)

    target = {
        "boxes": boxes,
        "labels": labels
    }

    return img, target

# Klasa za dataset
class CustomVOCDataset(Dataset):
    def __init__(self, root, CLASSES, transforms=None, return_image_id=False):
        self.root = root
        self.transforms = transforms
        self.CLASSES = CLASSES
        self.image_dir = os.path.join(root, 'images')
        self.annotation_dir = os.path.join(root, 'annotations')
        self.image_ids = [f[:-4] for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.return_image_id = return_image_id  

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        img, target = load_image_and_annotations(img_path, ann_path, self.CLASSES, self.transforms)

        if self.return_image_id:
            return img, target, image_id
        else:
            return img, target  

        