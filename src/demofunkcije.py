import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import numpy as np

from src.customDataset import *
from src.modelFunkcije import *
from src.metrike import *

def process_video(video_path, output_video_path, model, device, fps=30, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Uzmemo dimenzije videa
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertujemo frejm u PIL image, a zatim u tensor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)

        # Detekcija
        with torch.no_grad():
            predictions = model(img_tensor)

        # Crtamo bounding boxes i labele na frejmu
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, str(label), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Procesuirani frejm stavimo u video
        out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed {frame_count} frames and saved the video to {output_video_path}")