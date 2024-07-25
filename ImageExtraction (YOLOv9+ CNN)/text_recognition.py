
import os
import shutil
import xml.etree. ElementTree as ET 
import numpy as np
import pandas as pd
import random
import cv2
import timm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn. model_selection import train_test_split
from .data_preprocess import extract_data_from_xml

dataset_dir = "icdar2003\SceneTrialTrain"
img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(dataset_dir)


def split_bounding_boxes (img_paths, img_labels, bboxes, save_dir): 
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    labels = [] # List to store labels
    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes): 
        img  = Image.open(img_path)
        for label, bb in zip (img_label, bbs): 
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
    # filter out if 90% of the cropped image is black or white 
        if np.mean(cropped_img) < 35 or np.mean (cropped_img) > 220: 
            continue
        if cropped_img.size[0] < 10 or cropped_img.size [1] < 10:
            continue
    # Save image
        filename = f"{count:06d}.jpg"
        cropped_img.save(os.path.join(save_dir, filename))
        new_img_path = os.path.join(save_dir, filename)
        label = new_img_path + '\t' + label
    # Append label to the list
        labels.append(label)
        count += 1
    print (f" Created {count} images")
    # Write labels to a text file
    with open(os.path.join(save_dir, 'labels.txt'), 'w') as f: 
        for label in labels:
            f.write(f" {label}\n")

save_dir='ocr_dataset '
split_bounding_boxes (img_paths, img_labels, bboxes, save_dir)

root_dir = save_dir
img_paths = []
labels = []
with open(os.path.join(root_dir,'labels.txt'), "r") as f:
    for label in f:
        labels.append(label.strip().split('\t')[1])
        img_paths.append(label.strip().split('\t')[0])


letters =[char.split(".")[0].lower() for char in labels]
letters = "".join(letters)
letters = sorted (list(set(list(letters))))
# create a string of all characters in the dataset
chars = "".join(letters)
# for "blank" character
blank_char = '-'
chars += blank_char
vocab_size = len(chars)


char_to_idx= {char: idx + 1 for idx, char in enumerate(sorted(chars))} 
idx_to_char= {index: char for char, index in char_to_idx.items()}


max_label_len = max ([len (label) for label in labels])
def encode(label, char_to_idx, max_label_len): 
    encoded_labels = torch.tensor(
        [char_to_idx [char] for char in label], dtype=torch.int32)
    label_len = len(encoded_labels)
    lengths  = torch.tensor(label_len,dtype=torch.int32)
    padded_labels = F.pad(encoded_labels, (0, max_label_len- label_len),value=0)
    return padded_labels, lengths


def decode(encoded_sequences, idx_to_char, blank_char='-'):
    decoded_sequences = []
    for seq in encoded_sequences: 
        decoded_label = []
        for idx, token in enumerate (seq):
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)
        decoded_sequences.append(''.join(decoded_label))
    return decoded_sequences

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

seed = 42
val_size = 0.2
test_size = 0.125
is_shuffle = True

X_train, X_val, Y_train, Y_val = train_test_split(img_paths,labels, test_size=val_size, random_state=seed, shuffle = is_shuffle)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=test_size, random_state=seed, shuffle = is_shuffle)

class STRDataset(Dataset):
    def __init__ (self, X,y, char_to_idx, max_label_len, label_encoder = None, transform = None ):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(
                label, 
                self.char_to_idx,
                self.max_label_len
            )
        return img,encoded_label, label_len
    


train_dataset = STRDataset ( X_train, Y_train,char_to_idx=char_to_idx,max_label_len = max_label_len,label_encoder = encode, transform= data_transform['train'])
val_dataset = STRDataset ( X_val, Y_val,char_to_idx=char_to_idx,max_label_len = max_label_len,label_encoder = encode, transform= data_transform['val'])
test_dataset = STRDataset ( X_test, Y_test,char_to_idx=char_to_idx,max_label_len = max_label_len,label_encoder = encode, transform= data_transform['val'])


train_batch_size = 32
test_batch_size = 8
train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=train_batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=train_batch_size,shuffle=False)


class CRNN (nn. Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2,unfreeze_layers =3):
        super(CRNN, self).__init__()
        
        backbone  = timm.create_model('resnet101',in_chans =1,pretrained=True)
        modules = list (backbone.children()) [: -2]
        modules.append(nn.AdaptiveAvgPool2d((1, None))) 
        self.backbone = nn. Sequential (* modules)
        # Unfreeze the last few layers
        for parameter in self.backbone [-unfreeze_layers:]. parameters(): 
            parameter.requires_grad= True
        self.mapSeq = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))
        self.lstm = nn.LSTM(512, hidden_size,n_layers, bidirectional =True, batch_first=True, dropout  = dropout if n_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size* 2)
        self.out= nn.Sequential(nn.Linear(hidden_size *2, vocab_size), nn.LogSoftmax(dim=2))
    def forward(self,x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1) # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.lstm (x)
        x = self.layer_norm(x)
        x = self.out (x)
        x = x.permute(1, 0, 2) # Based on CTC
        return x
    

hidden_size = 256
n_layers = 3
dropout_prob = 0.2
unfreeze_layers = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(vocab_size=vocab_size, hidden_size=hidden_size,n_layers=n_layers, dropout=dropout_prob, unfreeze_layers=unfreeze_layers).to(device)


def evaluate(model, dataloader, criterion, device):
  model.eval()
  losses = []
  with torch.no_grad():
    for inputs, labels, labels_len in  dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      labels_len= labels_len.to(device)

      outputs = model(inputs)
      logits_lens = torch.full(
          size = (outputs.size(1),),
          fill_value = outputs.size(0),
          dtype= torch.long
      ).to(device)

      loss = criterion(outputs, labels, logits_lens, labels_len)
      losses.append(loss.item())
  
  loss = sum(losses) / len(losses)
  return loss

def train(model, train_dataloader, val_dataloader, criterion, optimizer,scheduler, device, epochs):
  train_losses = []
  val_losses = []
  for epoch in range(epochs):
    batch_train_losses = []
    model.train()
    for idx, (inputs, labels, labels_len) in enumerate(train_dataloader):
      inputs = inputs.to(device)
      labels = labels.to(device)
      labels_len= labels_len.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      logits_lens = torch.full(
          size = (outputs.size(1),),
          fill_value = outputs.size(0),
          dtype= torch.long
      ).to(device)

      loss = criterion(outputs, labels, logits_lens, labels_len)
      batch_train_losses.append(loss.item())
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
      optimizer.step()
      batch_train_losses.append(loss.item())
    train_loss = sum(batch_train_losses) / len(batch_train_losses)
    train_losses.append(train_loss)
    val_loss = evaluate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    scheduler.step()
  return train_losses, val_losses

lr = 1e-3
epochs =20
weight_decay = 1e-5
scheduler_step_size = epochs * 0.6
criterion = nn.CTCLoss(blank_char = char_to_idx[blank_char], zero_infinity = True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs)

save_model_path = 'ocr_crnn_base.pth'
torch.save(model.state_dict(), save_model_path)