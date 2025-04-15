import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from torchvision import transforms

class PointDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(224, 224), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图片
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 加载标签（单行 16 个数字）
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            points = np.array(list(map(float, line.split()))).reshape(8, 2)  # [8, 2]
        
        # 将归一化坐标转换为像素坐标（基于目标尺寸 224, 224）
        points[:, 0] *= self.target_size[1]  # x 坐标，224
        points[:, 1] *= self.target_size[0]  # y 坐标，224
        
        # 转换为 tensor
        points = torch.tensor(points, dtype=torch.float32).view(-1)  # [16]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return image, points

def get_dataloaders(data_root='dataset', batch_size=4, num_gpus=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 确保一致性
        transforms.ToTensor(),
    ])
    
    train_dataset = PointDataset(
        image_dir=os.path.join(data_root, 'train', 'images'),
        label_dir=os.path.join(data_root, 'train', 'labels'),
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * num_gpus,
        shuffle=True,
        num_workers=4
    )
    
    val_dataset = PointDataset(
        image_dir=os.path.join(data_root, 'val', 'images'),
        label_dir=os.path.join(data_root, 'val', 'labels'),
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * num_gpus,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

#test
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(data_root='/localdata/szhoubx/rm/data/dataset', batch_size=4, num_gpus=1)
    for images, points in train_loader:
        print(f"Points shape: {points.shape}")
        print(f"Sample points (pixels): {points[0]}")
        print(f"Points range: {points.min().item()} to {points.max().item()}")
        break