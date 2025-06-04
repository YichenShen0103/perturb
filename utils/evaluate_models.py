import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def evaluate_models(noise_generator, id_model, task_model_pipeline, val_loader, device):
    mtcnn = MTCNN(keep_all=False, device=device)
    noise_generator.eval()
    id_model.eval()
    for task_model in task_model_pipeline:
        task_model.eval()
    
    id_correct_orig = 0
    id_correct_pert = 0
    task_correct_orig = [0 for _ in task_model_pipeline]
    task_correct_pert = [0 for _ in task_model_pipeline]
    total = 0
    
    with torch.no_grad():
        for origin_images, facenet_images, labels, id_labels in tqdm(val_loader, desc="Evaluating"):
            origin_images = origin_images.to(device)
            facenet_images = facenet_images.to(device)
            facenet_images = F.interpolate(facenet_images, size=(160, 160),
                                           mode='bilinear', align_corners=False)
            id_labels = id_labels.to(device)
            batch_size = origin_images.size(0)
            total += batch_size
            
            # Generate noise using DiT
            # perturbed_images = noise_generator(origin_images)
            added_noise = noise_generator(origin_images)
            perturbed_images = origin_images + added_noise
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
            for i, task_model in enumerate(task_model_pipeline):
                task_labels = labels[i].to(device)
                task_preds_orig = task_model(origin_images).argmax(1)
                task_preds_pert = task_model(perturbed_images).argmax(1)
                
                task_correct_orig[i] += (task_preds_orig == task_labels).sum().item()
                task_correct_pert[i] += (task_preds_pert == task_labels).sum().item()
    
    # Calculate accuracies
    id_acc_orig = 100 * id_correct_orig / total
    id_acc_pert = 100 * id_correct_pert / total
    task_acc_orig = [0 for _ in task_model_pipeline]
    task_acc_pert = [0 for _ in task_model_pipeline]
    for i in range(len(task_model_pipeline)):
        task_acc_orig[i] = 100 * task_correct_orig[i] / total
        task_acc_pert[i] = 100 * task_correct_pert[i] / total
    
    print("\nEvaluation Results:")
    print(f"ID Recognition Accuracy: Original={id_acc_orig:.2f}%, Perturbed={id_acc_pert:.2f}%")
    for i in range(len(task_model_pipeline)):
        print(f"Task {i+1} Classification Accuracy: Original={task_acc_orig[i]:.2f}%, Perturbed={task_acc_pert[i]:.2f}%")