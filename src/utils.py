import torch.nn as nn
from torchvision import transforms

# Link za dataset
url = "https://universe.roboflow.com/ds/tE2lVUrLjS?key=ROqhczt6Fq"
urlyolo= "https://universe.roboflow.com/ds/u6psVS4gwh?key=MufkhbxejQ"

# Broj klasa
num_classes = 5

# Klase za detekciju
CLASSES = ['__background__', 'player', 'referee', 'goalkeeper', 'ball']
CLASSES_YOLO = ['ball', 'goalkeeper','player', 'referee']
# Klase za base model (pre treniranja)
classes = [
    '__background__',  # index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# Za trening funkciju
lr = 0.005
momentum = 0.9
weight_decay = 0.0005
step_size = 3
gamma = 0.1

# Broj epoha
num_epochs = 10

# Threshold za metrike (compare_predictions_with_annotations)
thrcompare = 0.01

# Broj iteracija za generisanje slika
N = 3

#Features za YOLOv8

features_yolo = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2) 
)



#putanje za slike i anotacije yolo trening


batch_size_yolo = 8
num_epochs_yolo = 50
learning_rate_yolo = 0.001

transform_yolo = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_size_yolo = 7  
num_bboxes_yolo = 2 
num_classes_yolo = 4 
lambda_coord = 5.0
lambda_noobj = 0.5


learning_rate = 1e-3
