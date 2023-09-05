import os
from remove_bg import remove_background
from tqdm import tqdm
import multiprocessing as mp


def process_image(dataset_path, dest_path, i):
    des_img_path = os.path.join(dest_path, i)
    image_path = os.path.join(dataset_path, i)
    if os.path.exists(des_img_path):
        return
    res = remove_background(image_path).convert('RGB')
    res.save(des_img_path)

if __name__ == '__main__':
    dataset_path = "/home/oleksandr/yoga/data/ukraine-ml-bootcamp-2023/images/test_images/"
    dest_path = "/home/oleksandr/yoga/data/ukraine-ml-bootcamp-2023/images/test_images_no_bg/"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    imgs = os.listdir(dataset_path)
    for i in tqdm(imgs):
        process_image(dataset_path, dest_path, i)



