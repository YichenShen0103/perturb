import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from libs.CelebADataset import ImageDataset
from tqdm import tqdm

data_dir = 'data/'
batch_size = 32
num_epochs = 5
num_classes = 10177
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageDataset(data_dir='data/', attr='Male', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Start training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, inputs, __, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = F.interpolate(inputs, size=(160, 160), mode='bilinear', align_corners=False)

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
torch.save(model.state_dict(), 'models/facenet_celeba_finetuned.pth')
print("Model saved to models/facenet_celeba_finetuned.pth")