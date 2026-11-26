
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(training=True):
    if training:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


class CCTVImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, image_name


def create_dataloaders(
    image_dir,
    batch_size=16,
    train_ratio=0.8,
    seed=42
):
    train_full = CCTVImageDataset(
        image_dir, get_transforms(training=True)
    )

    val_full = CCTVImageDataset(
        image_dir, get_transforms(training=False)
    )

    total = len(train_full)
    train_size = int(total * train_ratio)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()

    train_dataset = Subset(train_full, indices[:train_size])
    val_dataset = Subset(val_full, indices[train_size:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
