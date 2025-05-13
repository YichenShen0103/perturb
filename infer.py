import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# Import our custom models and dataset
from libs.CelebADataset import ImageDataset
from facenet_pytorch import InceptionResnetV1
from libs.BinaryClassifier import BinaryClassifier
from libs.DiT import *

from utils.evaluate import evaluate_models

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    id_model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=10177).to(device)
    task_model = BinaryClassifier().to(device)
    noise_generator = DiT_S_8(
        input_size=input_size,
        in_channels=3,
        scale=0.1,  # Control the strength of the perturbation
    ).to(device)

    # load state dicts
    try:
        id_model.load_state_dict(torch.load('models/facenet_celeba_finetuned.pth', map_location=device))
        task_model.load_state_dict(torch.load('models/task_classifier.pth', map_location=device))
        noise_generator.load_state_dict(torch.load('models/noise_generator.pth', map_location=device))
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset using the custom dataset class
    print("Loading dataset...")
    csv_path = 'data/attr_celeba_facenet.csv'  # Update with actual CSV filename
    img_dir = 'data/faces'
    attr = "Wearing_Hat"

    dataset = ImageDataset(csv_path=csv_path, img_dir=img_dir, attr=attr, transform=transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Evaluate models
    evaluate_models(noise_generator, id_model, task_model, val_loader, device)

if __name__ == "__main__":
    main()