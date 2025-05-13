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

def main() -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    input_size = 256  # Image size
    batch_size = 32
    num_epochs = 20
    lr = 2e-4
    
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
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize models
    print("Initializing models...")
    
    # ID classifier
    id_model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=10177).to(device)
    
    # Attr classifier using ResNet50
    task_model = BinaryClassifier().to(device)
    
    # DiT noise generator - using the provided implementation
    noise_generator = DiT_S_8(
        input_size=input_size,
        in_channels=3,
        scale=0.1,  # Control the strength of the perturbation
    ).to(device)
    
    # Training phases
    print("Starting training phases...")
    os.makedirs('models', exist_ok=True)
    
    # Phase 1: Train the ID classifier
    print("\nPhase 1: Loading ID classifier...")
    if os.path.exists('models/facenet_celeba_finetuned.pth'):
        id_model.load_state_dict(torch.load('models/facenet_celeba_finetuned.pth'))
        print("Loaded existing ID classifier model.")
    else:
        print("State-dict file doesn\'t exist! Train Facenet model first.")
        return
    
    # Phase 2: Train the task classifier
    print("\nPhase 2: Loading task classifier...")
    if os.path.exists('models/task_classifier.pth'):
        task_model.load_state_dict(torch.load('models/task_classifier.pth'))
        print("Loaded existing task classifier model.")
    else:
        print("State-dict file doesn\'t exist. Training task classifier...")
        train_task_classifier(task_model, train_loader, val_loader, device, epochs=10)
        torch.save(task_model.state_dict(), 'models/task_classifier.pth')
    
    # Phase 3: Train the DiT noise generator
    print("\nPhase 3: Training DiT noise generator...")
    train_noise_generator(
        noise_generator, 
        id_model, 
        task_model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=num_epochs,
        lr=lr
    )
    torch.save(noise_generator.state_dict(), 'models/noise_generator.pth')
    
    # Final evaluation
    print("\nFinal evaluation...")
    evaluate_models(noise_generator, id_model, task_model, val_loader, device)

def train_task_classifier(model, train_loader, val_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, attr_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            attr_labels = attr_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, attr_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validate
        val_loss, val_accuracy = validate_task_classifier(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def validate_task_classifier(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, attr_labels, _ in val_loader:
            images = images.to(device)
            attr_labels = attr_labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, attr_labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += attr_labels.size(0)
            correct += (predicted == attr_labels).sum().item()
    
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy

def identity_confusion_loss(id_preds, id_labels):
    """
    Maximizes confusion in ID predictions
    """
    return -F.cross_entropy(id_preds, id_labels)

def train_noise_generator(noise_generator, id_model, task_model, train_loader, val_loader, device, epochs=50, lr=0.0001):
    # Freeze ID and task models
    for param in id_model.parameters():
        param.requires_grad = False
    for param in task_model.parameters():
        param.requires_grad = False
    
    id_model.eval()
    task_model.eval()
    
    # Configure optimizer for noise generator
    optimizer = optim.AdamW(noise_generator.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    task_criterion = nn.CrossEntropyLoss()
    
    # Loss weights
    lambda_id = 1.0      # Weight for ID confusion loss
    lambda_task = 1.0  # Weight for task preservation loss
    lambda_visual = 0.5  # Weight for visual similarity loss
    
    for epoch in range(epochs):
        noise_generator.train()
        running_loss = 0.0
        running_id_loss = 0.0
        running_task_loss = 0.0
        running_visual_loss = 0.0
        
        for images, attr_labels , id_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            facenet_images = F.interpolate(images, size=(160, 160), mode='bilinear', align_corners=False)
            facenet_images = facenet_images.to(device)
            id_labels = id_labels.to(device)
            attr_labels = attr_labels.to(device)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            
            # Generate noise using DiT
            perturbed_images = noise_generator(images)
            facenet_perturbed_images = F.interpolate(
                perturbed_images, 
                size=(160, 160), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Get predictions from ID and task models
            id_pred_orig = id_model(facenet_images)
            id_pred_pert = id_model(facenet_perturbed_images)
            
            tast_pred_orig = task_model(images)
            task_pred_pert = task_model(perturbed_images)
            
            # Calculate losses
            # 1. ID confusion loss - maximize entropy of ID predictions
            id_loss = identity_confusion_loss(id_pred_pert, id_labels)
            # id_loss = task_criterion(id_pred_pert, id_labels)
            
            # 2. Task preservation loss
            task_loss = task_criterion(task_pred_pert, attr_labels)
            
            # 3. Visual similarity loss (L1 loss)
            visual_loss = nn.L1Loss()(perturbed_images, images)
            
            # Total loss
            total_loss = lambda_id * id_loss + lambda_task * task_loss + lambda_visual * visual_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            running_loss += total_loss.item()
            running_id_loss += id_loss.item()
            running_task_loss += task_loss.item()
            running_visual_loss += visual_loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate_models(noise_generator, id_model, task_model, val_loader, device)
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"ID Loss: {running_id_loss/len(train_loader):.4f}, "
              f"Task Loss: {running_task_loss/len(train_loader):.4f}, "
              f"Visual Loss: {running_visual_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(noise_generator.state_dict(), f'models/noise_generator_epoch{epoch+1}.pth')

def evaluate_models(noise_generator, id_model, task_model, val_loader, device, save_samples=True):
    noise_generator.eval()
    id_model.eval()
    task_model.eval()
    
    id_correct_orig = 0
    id_correct_pert = 0
    task_correct_orig = 0
    task_correct_pert = 0
    total = 0
    
    # For visualization
    if save_samples and len(val_loader) > 0:
        first_batch = next(iter(val_loader))
        sample_images = first_batch[0][:8].to(device)  # Take first 8 images
    
    with torch.no_grad():
        for images, task_labels, id_labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            facenet_images = F.interpolate(images, size=(160, 160), mode='bilinear', align_corners=False)
            facenet_images = facenet_images.to(device)
            task_labels = task_labels.to(device)
            id_labels = id_labels.to(device)
            batch_size = images.size(0)
            total += batch_size
            
            # Generate perturbed images
            perturbed_images = noise_generator(images)
            facenet_perturbed_images = F.interpolate(
                perturbed_images, 
                size=(160, 160), 
                mode='bilinear', 
                align_corners=False
            )
            
            # ID classification
            id_preds_orig = id_model(facenet_images).argmax(1)
            id_preds_pert = id_model(facenet_perturbed_images).argmax(1)
            
            id_correct_orig += (id_preds_orig == id_labels).sum().item() 
            id_correct_pert += (id_preds_pert == id_labels).sum().item()
            
            # Task classification
            task_preds_orig = task_model(images).argmax(1)
            task_preds_pert = task_model(perturbed_images).argmax(1)
            
            task_correct_orig += (task_preds_orig == task_labels).sum().item()
            task_correct_pert += (task_preds_pert == task_labels).sum().item()
    
    # Calculate accuracies
    id_acc_orig = 100 * id_correct_orig / total
    id_acc_pert = 100 * id_correct_pert / total
    task_acc_orig = 100 * task_correct_orig / total
    task_acc_pert = 100 * task_correct_pert / total
    
    print("\nEvaluation Results:")
    print(f"ID Recognition Accuracy: Original={id_acc_orig:.2f}%, Perturbed={id_acc_pert:.2f}%")
    print(f"Task Classification Accuracy: Original={task_acc_orig:.2f}%, Perturbed={task_acc_pert:.2f}%")
    
    # Visual comparison
    if save_samples:
        with torch.no_grad():
            perturbed_samples = noise_generator(sample_images)
            
            # Denormalize
            def denormalize(x):
                return x * 0.5 + 0.5
            
            # Convert to numpy for plotting
            orig_np = denormalize(sample_images).cpu().permute(0, 2, 3, 1).numpy()
            pert_np = denormalize(perturbed_samples).cpu().permute(0, 2, 3, 1).numpy()
            
            # Create figure
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(8):
                axes[0, i].imshow(np.clip(orig_np[i], 0, 1))
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(np.clip(pert_np[i], 0, 1))
                axes[1, i].set_title("Perturbed")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            os.makedirs('samples', exist_ok=True)
            plt.savefig('samples/comparison.png', dpi=150)
            plt.close()

if __name__ == "__main__":
    main()