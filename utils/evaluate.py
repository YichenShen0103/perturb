import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

import torch
from torchvision.transforms.functional import to_pil_image

# Import our custom models and dataset
from libs.DiT import *

def evaluate_models(noise_generator, id_model, task_model, val_loader, device, save_samples=True):
    mtcnn = MTCNN(keep_all=False, device=device)
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
        for origin_images, facenet_images, task_labels, id_labels in tqdm(val_loader, desc="Evaluating"):
            origin_images = origin_images.to(device)
            facenet_images = facenet_images.to(device)
            id_labels = id_labels.to(device)
            task_labels = task_labels.to(device)
            batch_size = origin_images.size(0)
            total += batch_size
            
            # Generate noise using DiT
            perturbed_images = noise_generator(origin_images)
            stack = []
            for img in perturbed_images: 
                img_pil = to_pil_image(img.cpu().clamp(0, 1))
                face = mtcnn(img_pil) 
                if face is not None:
                    stack.append(face)
                else:
                    stack.append(torch.zeros(3, 160, 160))
            facenet_perturbed_images = torch.stack(stack).to(device)
            
            # ID classification
            id_preds_orig = id_model(facenet_images).argmax(1)
            id_preds_pert = id_model(facenet_perturbed_images).argmax(1)
            
            id_correct_orig += (id_preds_orig == id_labels).sum().item() 
            id_correct_pert += (id_preds_pert == id_labels).sum().item()
            
            # Task classification
            task_preds_orig = task_model(origin_images).argmax(1)
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