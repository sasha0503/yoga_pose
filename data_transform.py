import os
import shutil

import numpy as np
from PIL import Image
from torchvision import transforms

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.RandomRotation(9),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.5, interpolation=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # run main to create augmented images and review them

    augment_path = "data/augment"
    os.makedirs(augment_path, exist_ok=True)

    data_path = "data/ukraine-ml-bootcamp-2023/images/train_images"
    train_csv = "data/ukraine-ml-bootcamp-2023/train.csv"

    csv_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)
    images = [os.path.join(data_path, image) for image in csv_data[:, 0]]
    random_images = np.random.choice(images, 10)
    for i, image_path in enumerate(random_images):
        image = Image.open(image_path).convert("RGB")
        image_tensor = augmentation_transform(image)
        pil_image = transforms.ToPILImage()(image_tensor)
        pil_image.save(os.path.join(augment_path, f"{i}_aug.jpg"))
        shutil.copy(image_path, os.path.join(augment_path, f"{i}_orig.jpg"))
