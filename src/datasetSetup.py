import wget
import zipfile
import os
import shutil
from src.utils import *
    
def srediDataset(skinuti_na_ovaj_path, naziv_fajla):
    output_file = downloadDataset(skinuti_na_ovaj_path, naziv_fajla)
    unzipDataset(output_file, output_file+"_unzip")
    organizeDataset(output_file+"_unzip", output_file+"_sredjen")


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

# Funkcija koja kreira strukturu direktorijuma i prerasporedjuje fajlove
def organizeDataset(dataset_path, new_dataset_path):
    #za svaki od foldera pokrenemo funkciju
    for folder in ['train', 'valid', 'test']:
        folder_path = os.path.join(dataset_path, folder)
        new_folder_path = os.path.join(new_dataset_path, folder)
        if os.path.exists(folder_path):
            organize_dataset_folder(folder_path, new_folder_path)

    print("Dataset je uspesno organizovan")
