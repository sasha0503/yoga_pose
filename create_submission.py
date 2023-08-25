import os

import torch
from torch.utils.data import DataLoader

from data_transform import base_transform
from train import CustomDataset, load_model, reversed_groups

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda', 'CUDA is not available'

test_data_path = 'data/ukraine-ml-bootcamp-2023/images/test_images'
images = [os.path.join(test_data_path, i) for i in os.listdir(test_data_path)]


def get_res(model_path):
    model_path = os.path.join('runs', model_path, 'model.pth')
    model = load_model(from_scratch=False, model_path=model_path)
    model = model.to(device)
    model.eval()
    test_data = CustomDataset(images, targets=None, transform=base_transform)
    data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    local_res = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            local_res.extend(outputs)

    assert len(local_res) == len(images), 'Wrong predictions length'
    tensor_res = torch.cat(local_res)
    tensor_res = tensor_res.view(len(images), -1)
    return tensor_res

MODEL_PATHS = [
    'train-08-25_17:35-WIN',
    'train-08-25_17:53',
    ]
res = []

print('Test size:', len(images))

for i, model_path in enumerate(MODEL_PATHS):
    res.append(get_res(model_path))
    print(f'Predictions {i} done')

final_res = torch.stack(res).mean(dim=0)
final_res = [r.argmax().item() for r in final_res]
final_res = [reversed_groups[r] for r in final_res]


submission_name = "submissions/" + '~'.join(MODEL_PATHS) + '~~submission.csv'
if not os.path.exists('submissions'):
    os.makedirs('submissions')
with open(submission_name, 'w') as f:
    f.write('image_id,class_6\n')
    for image, pred in zip(images, final_res):
        f.write(f'{os.path.basename(image)},{pred}\n')
