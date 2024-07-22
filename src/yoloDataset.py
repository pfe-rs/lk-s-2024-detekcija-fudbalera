import wget
import zipfile
import os
import shutil
from src.utils import *
    
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
   

class YoloV8Dataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                labels.append(int(parts[0]))
                #anotcije su centar, dimenzije
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([x_center, y_center, width, height])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target