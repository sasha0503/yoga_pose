"""run the model on the test set and save wrong predictions"""

import os
import shutil

import torch
import numpy as np

from train import reversed_groups, CustomClassifier, CustomDataset
from create_submission import get_res, MODEL_PATHS, submission_name, DATA_TRANSFORMS

train_csv = "data/ukraine-ml-bootcamp-2023/train.csv"
train_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)
img_paths = [os.path.join('data/ukraine-ml-bootcamp-2023/images/train_images', i) for i in train_data[:, 0]]
ground_truth = train_data[:, 1]

res = []
for i, (model_path, data_transform) in enumerate(zip(MODEL_PATHS, DATA_TRANSFORMS)):
    test_data = CustomDataset(img_paths, targets=None, transform=data_transform)
    res.append(get_res(model_path, test_data))
    print(f'Error predictions {i} done')
final_res = torch.stack(res).mean(dim=0)
final_res = [r.argmax().item() for r in final_res]
final_res = [reversed_groups[r] for r in final_res]

error_path = os.path.join(os.path.dirname(submission_name), os.path.basename(submission_name)[:-4] + '__errors')
os.makedirs(error_path, exist_ok=True)
wrong_preds = []
for i, (pred, gt) in enumerate(zip(final_res, ground_truth)):
    if int(pred) != int(gt):
        shutil.copy(img_paths[i], os.path.join(error_path, f'{i}_{gt}_{pred}.jpg'))
