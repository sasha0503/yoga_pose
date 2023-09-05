import cv2
import os
import numpy as np
from rembg import remove
from PIL import Image, ImageEnhance


def remove_background(image_path):
    input = Image.open(image_path)


    # Removing the background from the given Image
    output = remove(input, threshold=50)

    return output


if __name__ == '__main__':
    # get examples

    no_bg_path = "data/no_bg"
    os.makedirs(no_bg_path, exist_ok=True)

    data_path = "data/ukraine-ml-bootcamp-2023/images/train_images"
    train_csv = "data/ukraine-ml-bootcamp-2023/train.csv"

    csv_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)
    images = [os.path.join(data_path, image) for image in csv_data[:, 0]]
    random_images = images[30:40]
    for i, image_path in enumerate(random_images):
        res = remove_background(image_path).convert('RGB')
        res.save(os.path.join(no_bg_path, f'{i}_no_bg.jpg'))
        cv2.imwrite(os.path.join(no_bg_path, f'{i}_original.jpg'), cv2.imread(image_path))
