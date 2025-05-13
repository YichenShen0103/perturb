import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    """
    Custom dataset for loading images with labels from CSV file
    """
    def __init__(self, csv_path, img_dir, attr, transform=None):
        """
        Args:
            csv_path: Path to the CSV file with image_id and attr columns
            img_dir: Directory containing the images named as {image_id}.jpg
            transform: Optional transform to be applied on the images
        """
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.attr = attr
        
        # Map attr to numerical labels (0, 1)
        # Assuming attr is stored as strings (e.g., 'male', 'female') or as binary values
        unique_vals = self.data_frame[attr].unique()
        if len(unique_vals) > 2:
            raise ValueError(f"Expected binary values, found {len(unique_vals)} different values")
        
        self.attr_to_idx = {attribute: idx for idx, attribute in enumerate(unique_vals)}
        print(f"Attribute mapping: {self.attr_to_idx}")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_id = self.data_frame.iloc[idx]['image_id']
        img_name = os.path.join(self.img_dir, f"{img_id}")
        
        # Open image and convert to RGB (in case of grayscale)
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder black image if the image can't be loaded
            image = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # Get attr label
        attrbt = self.data_frame.iloc[idx][self.attr]
        attrbt_idx = self.attr_to_idx[attrbt]

        # Get ID label
        id_label = self.data_frame.iloc[idx]['person_id'] - 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, attrbt_idx, id_label