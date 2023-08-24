import json
import os
import shutil

import cv2


data_folder = 'data/ukraine-ml-bootcamp-2023/examples'
reference_folder = 'data/ukraine-ml-bootcamp-2023/reference'
subcat_folder = 'data/ukraine-ml-bootcamp-2023/subcat'
if not os.path.exists(subcat_folder):
    os.makedirs(subcat_folder)
if not os.path.exists(reference_folder):
    os.makedirs(reference_folder)

CLASS_ID = '3'

class_path = os.path.join(data_folder, CLASS_ID)
images = os.listdir(class_path)

already_categorized = []
for subcat in os.listdir(subcat_folder):
    already_categorized.extend(os.listdir(os.path.join(subcat_folder, subcat)))

images = [image for image in images if image not in already_categorized]

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
save_path = os.path.join(class_path, 'save.json')
if not os.path.exists(save_path):
    image2cat = {}
    j = 0
else:
    with open(save_path, 'r') as f:
        image2cat = json.load(f)
        last_image = list(image2cat.keys())[-1]
    j = images.index(last_image)

while True:
    print(j, '/', len(images))
    image = images[j]
    image_path = os.path.join(class_path, image)
    img = cv2.imread(image_path)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    elif k == ord('z'):
        j -= 1
    elif chr(k).isdigit():
        j = (j + 1) % len(images)
        sub_cat = chr(k)
        category = CLASS_ID + '_' + sub_cat
        image2cat[image] = sub_cat
        if not os.path.exists(os.path.join(reference_folder, category + '.jpg')):
            shutil.copy(image_path, os.path.join(reference_folder, category + '.jpg'))
        with open(save_path, 'w') as f:
            json.dump(image2cat, f)
    else:
        continue

for image, sub_cat in image2cat.items():
    image_path = os.path.join(class_path, image)
    category = CLASS_ID + '_' + sub_cat
    dest = os.path.join(subcat_folder, category, image)
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    shutil.copy(image_path, dest)
    j = (j + 1) % len(images)
