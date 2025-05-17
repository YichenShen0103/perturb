import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

# Import our custom models and dataset
from libs.CelebADataset import ImageDataset
from facenet_pytorch import InceptionResnetV1, MTCNN
from libs.BinaryClassifier import BinaryClassifier
from libs.DiT import *

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    print("Loading models...")
    id_model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=10177).to(device)
    task_model = BinaryClassifier().to(device)
    noise_generator = DiT_S_8(
        input_size=256,
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
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset using the custom dataset class
    print("Loading dataset...")
    data_dir = 'data/'
    attr = "Male"
    dataset = ImageDataset(data_dir=data_dir, attr=attr, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    gender_dict = {0: 'Female', 1: 'Male'}
    mtcnn = MTCNN(keep_all=False, device=device)
    noise_generator.eval()
    id_model.eval()
    task_model.eval()
    
    first_batch = next(iter(data_loader))
    sample_images = first_batch[0][:4].to(device)  # Take first 8 images
    id_labels = first_batch[3][:4]    # Take first 8 labels
    task_labels = first_batch[2][:4]  # Take first 8 labels
    # perturbed_images = noise_generator(sample_images)
    added_noise = noise_generator(sample_images)
    perturbed_images = sample_images + added_noise
    stack = []
    for img in perturbed_images: 
        img_pil = to_pil_image(img.cpu().clamp(0, 1))
        face = mtcnn(img_pil) 
        if face is not None:
            stack.append(face)
        else:
            stack.append(torch.zeros(3, 160, 160))
    facenet_perturbed_images = torch.stack(stack).to(device)
            
    id_preds_pert = id_model(facenet_perturbed_images).argmax(1)
    task_preds_pert = task_model(perturbed_images).argmax(1)
    
    with torch.no_grad():
        # Denormalize
        def denormalize(x):
            return x * 0.5 + 0.5
        
        # Convert to numpy for plotting
        orig_np = denormalize(sample_images).cpu().permute(0, 2, 3, 1).numpy()
        pert_np = denormalize(perturbed_images).cpu().permute(0, 2, 3, 1).numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        for i in range(4):
            axes[0, i].imshow(np.clip(orig_np[i], 0, 1))
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')
            axes[0, i].text(
                0.5, -0.1, 
                f"ID: {id_labels[i].item()}, " +
                f"Gender: {gender_dict[task_labels[i].item()]}", 
                ha='center', va='top', transform=axes[0, i].transAxes
            )
            
            axes[1, i].imshow(np.clip(pert_np[i], 0, 1))
            axes[1, i].set_title("Perturbed")
            axes[1, i].axis('off')
            axes[1, i].text(
                0.5, -0.1, 
                f"ID: {id_preds_pert[i].item()}, " +
                f"Gender: {gender_dict[task_preds_pert[i].item()]}", 
                ha='center', va='top', transform=axes[1, i].transAxes
            )
        
        plt.tight_layout()
        os.makedirs('samples', exist_ok=True)
        plt.savefig('samples/comparison.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    main()