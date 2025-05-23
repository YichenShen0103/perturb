import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

# Import our custom models and dataset
from libs.CelebADataset import ImageDataset
from facenet_pytorch import InceptionResnetV1, MTCNN
from libs.BinaryClassifier import BinaryClassifier
from libs.SSIMLoss import SSIMLoss
from libs.DiT import *

def main() -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    input_size = 256  # Image size
    batch_size = 32
    num_epochs = 5
    lr = 2e-4
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset using the custom dataset class
    print("Loading dataset...")
    data_dir = 'data/'
    attr = "Male"
    
    dataset = ImageDataset(data_dir=data_dir, attr=attr, transform=transform)
    
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
        train_task_classifier(task_model, train_loader, val_loader, device, epochs=3)
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
    print("DiT noise generator training completed.")

def train_task_classifier(model, train_loader, val_loader, device, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, _, attr_labels, __ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        for images, _, attr_labels, __ in val_loader:
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
    # return -F.cross_entropy(id_preds, id_labels)
    margin = math.log(10177) * 0.8
    return torch.clamp(margin - F.cross_entropy(id_preds, id_labels), min=0)

def train_noise_generator(noise_generator, id_model, task_model, train_loader, val_loader, device, epochs=50, lr=0.0001):
    mtcnn = MTCNN(keep_all=False, device=device)

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
    # visual_criterion = nn.L1Loss()
    # visual_criterion = nn.MSELoss()
    # visual_criterion = nn.SmoothL1Loss()
    visual_criterion = SSIMLoss(data_range=1.0)
    
    # Loss weights
    lambda_id = 1.0      # Weight for ID confusion loss
    lambda_task = 1.0    # Weight for task preservation loss
    lambda_visual = 10.0  # Weight for visual similarity loss
    
    for epoch in range(epochs):
        noise_generator.train()
        running_loss = 0.0
        running_id_loss = 0.0
        running_task_loss = 0.0
        running_visual_loss = 0.0
        
        for images, _, attr_labels , id_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            id_labels = id_labels.to(device)
            attr_labels = attr_labels.to(device)
            
            optimizer.zero_grad()
            
            # Generate noise using DiT
            added_noise = noise_generator(images)
            perturbed_images = images + added_noise
            # perturbed_images = noise_generator(images)

            stack = []
            for img in perturbed_images: 
                img_pil = to_pil_image(img.cpu().clamp(0, 1))
                face = mtcnn(img_pil)
                if face is not None:
                    stack.append(face)
                else:
                    stack.append(torch.zeros(3, 160, 160))
            facenet_perturbed_images = torch.stack(stack).to(device)
            
            # Get predictions from ID and task models
            id_pred_pert = id_model(facenet_perturbed_images)
            
            # tast_pred_orig = task_model(images)
            task_pred_pert = task_model(perturbed_images)
            
            # Calculate losses
            # 1. ID confusion loss - maximize entropy of ID predictions
            id_loss = identity_confusion_loss(id_pred_pert, id_labels)
            # id_loss = task_criterion(id_pred_pert, id_labels)
            
            # 2. Task preservation loss
            task_loss = task_criterion(task_pred_pert, attr_labels)
            
            # 3. Visual similarity loss (L1 loss)
            visual_loss = visual_criterion(perturbed_images, images)
            
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

        # Save model every epoch
        torch.save(noise_generator.state_dict(), f'.cache/noise_generator_epoch{epoch+1}.pth')
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"ID Loss: {running_id_loss/len(train_loader):.4f}, "
              f"Task Loss: {running_task_loss/len(train_loader):.4f}, "
              f"Visual Loss: {running_visual_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()
