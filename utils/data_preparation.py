from facenet_pytorch import MTCNN
from PIL import Image
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

import pandas as pd

def preprocess_faces(csv_path, img_dir, output_dir, new_csv_path):
    df = pd.read_csv(csv_path)
    mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(df)-1, -1, -1)):
        img_id = df.iloc[i]['image_id']
        img_path = os.path.join(img_dir, img_id)
        save_path = os.path.join(output_dir, img_id)

        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                face_img = transforms.ToPILImage()(face)
                face_img.save(save_path)
            else:
                print(f"No face detected in {img_id}")
                df.drop(i, inplace=True)
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
    df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    csv_path = '../data/attr_celeba.csv'
    img_dir = '../data/imgs'
    output_dir = '../data/faces'
    new_csv_path = '../data/attr_celeba_facenet.csv'
    preprocess_faces(csv_path, img_dir, output_dir, new_csv_path)