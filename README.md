# Automatic License Plate Detection

This repository contains the implementation of an automatic license plate detection system using PyTorch and YOLOv8. The project aims to detect license plates in images with high accuracy and deploys the model via a Streamlit web application.

## Tech Stack
Frameworks: PyTorch, YOLOv8 <br>
Languages: Python
Libraries: OpenCV, NumPy, Pandas
Deployment: Streamlit

## Dataset
Link : https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
Total images: 433
The dataset consists of pre-annotated images of vehicles with corresponding XML files.
The annotations were processed and converted into the YOLO format for training purposes.

## Model Training
Model: YOLOv8
Epochs: 100
The XML annotations were extensively processed and transformed into YOLOv8-compatible format.
Trained for 100 epochs with optimal hyperparameters to maximize detection performance.
Performance
mAP@0.5-0.7: Achieved satisfactory accuracy levels for plate detection.
mAP@0.5:0.95-0.5: Demonstrates robustness in detecting plates under varying conditions.

## Deployment
The trained model is deployed using Streamlit, providing a simple web interface where users can upload images and receive real-time detection results.
The web interface is lightweight, allowing for quick detection and easy integration into broader applications.
