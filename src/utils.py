
# Link za dataset
url = "https://universe.roboflow.com/ds/tE2lVUrLjS?key=ROqhczt6Fq"

# Broj klasa
num_classes = 5

# Klase za detekciju
CLASSES = ['__background__', 'player', 'referee', 'goalkeeper', 'ball']

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
