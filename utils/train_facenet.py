import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

celeba_img_dir = '../data/faces'
identity_file = '../data/attr_celeba_facenet.csv' 

batch_size = 32
num_epochs = 5
num_classes = 10177
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_id = self.data_frame.iloc[idx]['image_id']
        img_name = os.path.join(self.img_dir, f"{img_id}")
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder black image if the image can't be loaded
            image = Image.new('RGB', (160, 160), (0, 0, 0))

        # Get ID label
        id_label = self.data_frame.iloc[idx]['person_id'] - 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, id_label

dataset = ImageDataset(img_dir=celeba_img_dir, csv_path=identity_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练
print("Start training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), '../models/facenet_celeba_finetuned.pth')
print("Model saved to models/facenet_celeba_finetuned.pth")