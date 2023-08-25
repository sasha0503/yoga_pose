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

CLASS_ID = '5'

class_path = os.path.join(data_folder, CLASS_ID)
images = os.listdir(class_path)

already_categorized = []
for subcat in os.listdir(subcat_folder):
    already_categorized.extend(os.listdir(os.path.join(subcat_folder, subcat)))

images = [image for image in images if image not in already_categorized]

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
image2cat = {}
j = 0
while True:
    try:
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
            if not os.path.exists(os.path.join(subcat_folder, category)):
                os.makedirs(os.path.join(subcat_folder, category))
            if image in image2cat and image2cat[image] != sub_cat:
                shutil.move(os.path.join(subcat_folder, CLASS_ID + '_' + image2cat[image], image),
                            os.path.join(subcat_folder, category, image))
            else:
                shutil.copy(image_path, os.path.join(subcat_folder, category, image))
            image2cat[image] = sub_cat
            if not os.path.exists(os.path.join(reference_folder, category + '.jpg')):
                shutil.copy(image_path, os.path.join(reference_folder, category + '.jpg'))
        else:
            continue
    except Exception as e:
        print(e)
        continue
