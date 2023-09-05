"""run the model on the test set and save wrong predictions"""

import os
import shutil

import torch
import numpy as np

from train import reversed_groups, CustomClassifier, CustomDataset
from create_submission import get_res, MODEL_PATHS, submission_name, DATA_TRANSFORMS

train_csv = "data/ukraine-ml-bootcamp-2023/train.csv"
subcat_path = "/home/oleksandr/yoga/data/ukraine-ml-bootcamp-2023/subcat_no_bg"
subcats = os.listdir(subcat_path)

train_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1, dtype=str)

img_paths = []
for i in subcats:
    img_paths.extend([os.path.join(subcat_path, i, j) for j in os.listdir(os.path.join(subcat_path, i))])

imgs_ids = [os.path.basename(i) for i in img_paths]
indexes = [np.where(train_data[:, 0] == i)[0][0] for i in imgs_ids]

ground_truth = train_data[:, 1]
ground_truth = ground_truth[indexes]


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
        img_id = os.path.basename(img_paths[i])
        shutil.copy(img_paths[i], os.path.join(error_path, f'{img_id}_{gt}_{pred}.jpg'))
