import wget
import zipfile
import os
import shutil
from torch.utils.data import Dataset
from src.utils import *
from PIL import Image
import torch

def srediDataset(skinuti_na_ovaj_path, naziv_fajla):
    output_file = downloadDataset(skinuti_na_ovaj_path, naziv_fajla)
    unzipDataset(output_file, output_file+"_unzip")


def downloadDataset(skinuti_na_ovaj_path, naziv_fajla):

    # Kreira direktorijum ako ne postoji
    os.makedirs(skinuti_na_ovaj_path, exist_ok=True)

    #Path za novi fajl
    output_file = os.path.join(skinuti_na_ovaj_path, naziv_fajla)

    # Download
    wget.download(url, output_file)
    
    return output_file


def unzipDataset(zip_file_path, extract_to):

    # Kreira direktorijum ako ne postoji
    os.makedirs(extract_to, exist_ok=True)

    # Unzip 
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def organize_dataset_folder(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'images'), exist_ok=True)

    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        if filename.endswith('.xml'):  # anotacije su u XML formatu
            shutil.copy(src_file, os.path.join(dest_folder, 'annotations', filename))
        elif filename.endswith(('.jpg', '.jpeg', '.png')):  # slike su u jpg, jpeg, png
            shutil.copy(src_file, os.path.join(dest_folder, 'images', filename))
   

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class YoloV8Dataset(Dataset):
    def __init__(self, img_dir, labels_dir, grid_size=7, num_boxes=3, num_classes=10, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")

        # namesti na 0
        box_tensor = torch.zeros((self.grid_size, self.grid_size, self.num_boxes, 5))
        class_tensor = torch.zeros((self.grid_size, self.grid_size, self.num_classes))

        boxes = []
        labels = []

        # ucitavanje
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # koordinati celije u gridu
                grid_x = int(x_center * self.grid_size)
                grid_y = int(y_center * self.grid_size)
                
                # tensori
                box_tensor[grid_y, grid_x, 0] = torch.tensor([x_center, y_center, width, height, 1])
                class_tensor[grid_y, grid_x, label] = 1

        if self.transform:
            image = self.transform(image)

        return image, {'boxes': box_tensor, 'labels': class_tensor}
