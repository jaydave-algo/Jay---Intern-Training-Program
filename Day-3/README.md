# Day 3 - PyTorch Data Pipeline

## Objective
Build a reusable PyTorch Dataset and DataLoader pipeline using real office CCTV image frames.

## Dataset
- Total images: 700
- Training images: 560 (80%)
- Validation images: 140 (20%)
- Image format: JPG
- Original image dimensions: 360 x 640

## Implementation
- Custom PyTorch Dataset
- PyTorch DataLoader
- Albumentations image augmentation
- Image resizing to 224 x 224
- Image normalization
- Training and validation split
- Batch visualization

## Files
- images/: CCTV image frames
- dataset.py: Reusable Dataset and DataLoader implementation
- requirements.txt: Required Python packages

## Verification
- Dataset import successful
- Training and validation DataLoaders created successfully
- Test batch shape: [16, 3, 224, 224]

## Note
This exercise uses unlabelled CCTV images for data pipeline practice.
It does not train an object detection model.
