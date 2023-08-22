import os
import random
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data/ukraine-ml-bootcamp-2023'

train_images = os.listdir(os.path.join(data_path, 'images/train_images'))

train_size = len(train_images)
print('Train size:', train_size)

train_csv = os.path.join(data_path, 'train.csv')
train_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)
print('Train data shape:', train_data.shape)
print('Train data sample:', train_data[0])

# train_data format: image_id, class_id

# examples distribution
class_ids = train_data[:, 1]
unique_class_ids, counts = np.unique(class_ids, return_counts=True)
print('Unique class ids:', unique_class_ids)
for class_id, count in zip(unique_class_ids, counts):
    print('Class id:', class_id, 'Count:', count)

# plot distribution
plt.bar(unique_class_ids, counts)
plot_path = os.path.join(data_path, 'plot.png')
plt.savefig(plot_path)

# create examples for each class
os.makedirs(os.path.join(data_path, 'examples'), exist_ok=True)
for class_id in unique_class_ids:
    class_examples = np.random.choice(train_data[train_data[:, 1] == class_id][:, 0], size=9, replace=False)
    os.makedirs(os.path.join(data_path, 'examples', class_id), exist_ok=True)
    for example in class_examples:
        shutil.copy(os.path.join(data_path, 'images/train_images', example),
                   os.path.join(data_path, 'examples', class_id, example))

print('Examples created')
