import os
import shutil

import cv2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

subcat_path = 'data/ukraine-ml-bootcamp-2023/subcat'

CLASS_ID = '4_2'
ADD_CLASS_ID = '4_12'

class_path = os.path.join(subcat_path, CLASS_ID)
add_class_path = os.path.join(subcat_path, ADD_CLASS_ID)
if not os.path.exists(add_class_path):
    os.makedirs(add_class_path, exist_ok=True)

images_path = os.listdir(class_path)

j = 0
while True:
    print(j, '/', len(images_path))
    image_path = os.path.join(class_path, images_path[j])
    img = cv2.imread(image_path)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    elif k == ord('z'):
        j -= 1
    elif k == ord('s'):
        try:
            shutil.move(image_path, os.path.join(add_class_path, images_path[j]))
        except Exception as e:
            print(e)
        j = (j + 1) % len(images_path)
    elif k == ord('u'):
        try:
            shutil.move(os.path.join(add_class_path, images_path[j]), image_path)
        except Exception as e:
            print(e)
        j = (j + 1) % len(images_path)
    else:
        j = (j + 1) % len(images_path)
