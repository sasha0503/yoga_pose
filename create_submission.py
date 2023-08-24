import os

import torch
from torch.utils.data import DataLoader

from train import CustomDataset, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda', 'CUDA is not available'

MODEL_PATH = 'runs/train-08-23_21:49/model.pth'
model = load_model(from_scratch=False, model_path=MODEL_PATH)
model = model.to(device)
model.eval()
print('Model loaded')

test_data_path = 'data/ukraine-ml-bootcamp-2023/images/test_images'
submission_csv = os.path.join(os.path.dirname(MODEL_PATH), 'submission.csv')

images = [os.path.join(test_data_path, i) for i in os.listdir(test_data_path)]
print('Test size:', len(images))

test_data = CustomDataset(images, targets=None)
data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

res = []
for batch in data_loader:
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        res.extend(preds.cpu().numpy())

assert len(res) == len(images), 'Wrong predictions length'

print('Predictions done')

with open(submission_csv, 'w') as f:
    f.write('image_id,class_6\n')
    for image, pred in zip(images, res):
        f.write(f'{os.path.basename(image)},{pred}\n')
