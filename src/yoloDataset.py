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
        if filename.endswith('.txt'):  # anotacije su u TXT formatu
            shutil.copy(src_file, os.path.join(dest_folder, 'annotations', filename))
        elif filename.endswith(('.jpg', '.jpeg', '.png')):  # slike su u jpg, jpeg, png
            shutil.copy(src_file, os.path.join(dest_folder, 'images', filename))
   

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

import matplotlib.pyplot as plt

class YoloV8Dataset(Dataset):
    def __init__(self, img_dir, labels_dir, grid_size=7, num_boxes=1, num_classes=4, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")

        # Initialize output tensors with zeros
        box_tensor = torch.zeros((self.grid_size, self.grid_size, self.num_boxes, 5))
        class_tensor = torch.zeros((self.grid_size, self.grid_size, self.num_classes))

        # Process labels
        with open(label_path, 'r') as f:
            for line in f.readlines():

                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Calculate the grid cell coordinates
                grid_x = int(x_center * self.grid_size)
                grid_y = int(y_center * self.grid_size)

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    for i in range(self.num_boxes):
                        if box_tensor[grid_y, grid_x, i, 4] == 0:  # Check if the box slot is empty
                            box_tensor[grid_y, grid_x, i] = torch.tensor([x_center, y_center, width, height, 1])
                            class_tensor[grid_y, grid_x, label] = 1
                            break
                else:
                    print(f"Bounding box is out of grid bounds: {grid_x}, {grid_y}")

        if self.transform:
            image = self.transform(image)
        

        # Print tensors for debugging
        #print(f"Box tensor shape: {box_tensor.shape}")
        #print(f"Class tensor shape: {class_tensor.shape}")
        #print("Box tensor:", box_tensor)
        #print("Class tensor:", class_tensor)

        return image, (box_tensor, class_tensor)
