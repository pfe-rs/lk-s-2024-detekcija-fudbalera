# Soccer Player Detection Project

## Introduction

This project focuses on developing a machine learning model for detecting soccer players, balls, and referees on the pitch using a finetuned Faster R-CNN. We experimented with different datasets and compared results of different models in order to see the progress of our work.

## Dataset

### Selection and Preparation

1. **Dataset Source**: We chose a dataset from Roboflow for football player detection. [Link to dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1).

2. **Preparation**:
    - Unzipped the dataset directly on Google Colab.
    - Reformatted the dataset to separate annotations and images into different folders.
    - Split the dataset into train, validation, and test sets.

### Custom Dataset Class

- Created a custom VOC dataset class to work with the Faster R-CNN model, ensuring the dataset was properly formatted for training.

## Model Training and Evaluation

### Initial Implementation

1. **Model Choice**: We selected Faster R-CNN based on extensive literature review and discussions with our mentor.
2. **Initial Testing**: Implemented and tested Faster R-CNN on built-in PyTorch datasets to understand its performance.

### Data Loader

- Developed a data loader to handle the custom dataset format, enabling proper loading and augmentation of training data.

### Training

1. **Initial Training**: Trained the model on a small subset of images to verify the implementation.
2. **Metrics Calculation**: 
    - Implemented code to calculate confusion matrix elements (True Positive, False Positive, False Negative) for each class.
    - Evaluated the initial model’s performance using these metrics.

### Fine-Tuning and Performance

1. **Fine-Tuning**: Trained the model on the complete dataset using Paperspace, saving intermediate models after each epoch.
2. **Results**: Achieved a high detection accuracy (90-100%) for players, goalkeepers, and referees. The model also successfully detected balls when visible.
3. **Issues**: Identified issues with false positives, especially misclassifying goalkeepers and referees as players.

## Visualization and Demo

1. **Confusion Matrix**: Plotted confusion matrices to visualize model performance.
2. **Demo Creation**: Created a demo by running a video through the trained model to visualize real-time performance. Identified several issues:
    - Misclassification of advertisements as players.
    - Overlapping players losing bounding boxes.
    - Smear frames causing detection failures.

## Future Work

1. **Improved Visualization**: Develop a program to create graphical representations of class overlaps.
2. **Team Detection**: Plan to implement a method to detect different teams based on dominant colors within bounding boxes.
3. **YOLO Exploration**: Investigate YOLO models for potential real-time detection improvements, focusing on versions better suited for detecting small objects and handling multiple bounding boxes per grid cell.

## Challenges with YOLO

1. **Initial Attempts**: Attempted to implement YOLOv1 but faced issues with small object detection and grid cell limitations.
2. **Future Plans**: Explore newer YOLO versions with pre-configured datasets to simplify the process and improve detection capabilities, most likey YOLOv8.

## Conclusion

This project successfully developed a model for detecting soccer players, balls, and referees using Faster R-CNN. While initial results were promising, further improvements and exploration of alternative models like YOLO are planned to enhance performance and address current limitations.
