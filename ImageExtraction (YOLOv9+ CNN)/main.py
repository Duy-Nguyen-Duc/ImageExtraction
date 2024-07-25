import numpy as np
import ultralytics
import os
import pandas as pd
import random
import cv2
import timm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from ultralytics import YOLO
from .text_recognition import CRNN, decode
text_det_model_path = 'runs/detect/train/weights/best.pt'
yolo = YOLO(text_det_model_path)

chars = '0123456789abcdefghijklmnopqrstuvwxyz-'
vocab_size = len(chars)

char_to_idx= {char: idx + 1 for idx, char in enumerate(sorted(chars))} 
idx_to_char= {index: char for char, index in char_to_idx.items()}

hidden_size = 256
n_layers = 3
dropout_prob = 0.2
unfreeze_layers = 3
model_path = 'ocr_crnn_base.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crnn_model = CRNN(vocab_size=vocab_size, hidden_size=hidden_size,n_layers=n_layers, dropout=dropout_prob, unfreeze_layers=unfreeze_layers).to(device)
crnn_model.load_state_dict(torch.load(model_path))

def text_detection(img_path, text_det_model):
    text_det_results = text_det_model(img_path, verbose=False)[0]
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()

    return bboxes,classes, names,confs

def text_recognition(img, data_transforms, text_reg_model, idx_to_char, device):
    transformed_image = data_transforms(img)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu()
    text = decode(logits.permute(1,0,2).argmax(2), idx_to_char)
    return text


def visualize_detections (img, detections):
    plt.figure(figsize=(12, 8)) 
    plt.imshow(img)
    plt.axis ('off')
    for bbox, detected_class, confidence, transcribed_text in detections:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill = False, edgecolor='red', linewidth=2))
        plt.text(x1, y1- 10, f"{detected_class} ({confidence: .2f}): {transcribed_text}",
                 fontsize=9, bbox=dict (facecolor='red', alpha=0.5)
)
    plt.show()


data_transform = {
    'train': transforms.Compose([
        transforms.Resize ((100,420)),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5), 
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(degrees=1, shear=1),
        transforms.RandomPerspective(distortion_scale =0.3, p=0.5, interpolation=3),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]),
    'val': transforms.Compose ([
        transforms.Resize ((100, 420)),
        transforms.Grayscale (num_output_channels=1), 
        transforms.ToTensor (),
        transforms.Normalize ((0.5,), (0.5,))]),
}

def predict(img_path, data_transform, text_det_model, text_reg_model, idx_to_char, device):
    bboxes,classes, names,confs = text_detection(img_path, text_det_model)

    img = Image.open(img_path)
    predictions = []
    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1,y1,x2,y2 = bbox
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        cropped_image = img.crop((x1,y1,x2,y2))

        transcribed_text = text_recognition(cropped_image,data_transform, text_reg_model, idx_to_char, device)
        predictions.append((bbox, name, confidence, transcribed_text))
    visualize_detections(img,predictions)
    return predictions
