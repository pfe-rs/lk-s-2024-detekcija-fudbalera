import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Ekstrakcija dresa
def extract_jersey(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]

# Ekstraktovanje histograma
def extract_histogram(image, mask=None):
    hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Dobijanje boja timova
def find_team_colors(image, k=2):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    team_colors = kmeans.cluster_centers_.astype(int)
    return team_colors

# Proverava da li je objekat u jednoj od boja timova
def is_team_color(object_hist, team_histograms):
    distances = [euclidean(object_hist, team_hist) for team_hist in team_histograms]
    min_distance = min(distances)
    threshold = 0.5
    return min_distance < threshold

# Definisemo labele
def label_map(label):
    if label == 1:
        return 'player'
    elif label == 3:
        return 'goalkeeper'
    elif label == 2:
        return 'referee'
    elif label == 4:
        return 'ball'
    else:
        return 'unknown'

# Ispravljanje labela na osnovu boja
def correct_labels(detections, image, team_colors):
    corrected_labels = []
    team_histograms = [extract_histogram(np.uint8([[color]])) for color in team_colors]

    for detection in detections:
        x, y, w, h, label, score = detection
        object_image = image[int(y):int(y+h), int(x):int(x+w)]
        object_hist = extract_histogram(object_image)

        if is_team_color(object_hist, team_histograms):
            corrected_labels.append((x, y, w, h, 'player'))
        else:
            corrected_labels.append((x, y, w, h, label_map(label)))

    return corrected_labels

def convert_bbox(bbox):
    return [int(coord) for coord in bbox]

# Racunanje histograma
def compute_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Klasterovanje igraca
def cluster_players(histograms, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(histograms)
    labels = kmeans.labels_
    return labels

# Racunanje prosecnog histograma
def compute_average_histograms(histograms, labels, n_clusters=2):
    cluster_histograms = [[] for _ in range(n_clusters)]
    
    for hist, label in zip(histograms, labels):
        cluster_histograms[label].append(hist)
    
    average_histograms = []
    for cluster_hist in cluster_histograms:
        average_hist = np.mean(cluster_hist, axis=0)
        average_histograms.append(average_hist)
    
    return average_histograms

# Crtanje histograma
def plot_histogram(histogram, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

# Detektovanje timova i ispravljanje labela
def detect_and_correct_labels(image_path, model, confidence_threshold=0.5):

    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            detections.append((x1.item(), y1.item(), width.item(), height.item(), label.item(), score.item()))

    team_colors = find_team_colors(np.array(image))
    corrected_detections = correct_labels(detections, np.array(image), team_colors)

    image_np = np.array(image)

    histograms = []
    for detection in corrected_detections:
        x, y, w, h, label = detection[:5]
        if label == 'player':
            bbox = convert_bbox([x, y, x+w, y+h])
            jersey = extract_jersey(image_np, bbox)
            hist = compute_histogram(jersey)
            histograms.append(hist)

    histograms = np.array(histograms)
    cluster_labels = cluster_players(histograms)
    average_histograms = compute_average_histograms(histograms, cluster_labels)

    for i, avg_hist in enumerate(average_histograms):
        plot_histogram(avg_hist, f'Average Histogram for Team {i + 1}')

    for detection, cluster_label in zip(corrected_detections, cluster_labels):
        x, y, w, h, label = detection[:5]
        if label == 'player':
            color = (0, 255, 0) if cluster_label == 0 else (0, 0, 255)
            cv2.rectangle(image_np, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            label_text = f'Player\nTeam {cluster_label + 1}'
            for i, line in enumerate(label_text.split('\n')):
                cv2.putText(image_np, line, (int(x), int(y - 10 - (i*15))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (255, 255, 255)
            cv2.rectangle(image_np, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            label_text = label
            cv2.putText(image_np, label_text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return corrected_detections


