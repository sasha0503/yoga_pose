import os

import torch
from torch.utils.data import DataLoader

from data_transform import base_transform, base_transform_no_norm, augmentation_transform
from train import CustomDataset, load_model, reversed_groups, CustomClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert str(device) == 'cuda', 'CUDA is not available'


def get_res(model_path, test_data: CustomDataset):
    weights_path = os.path.join('runs', model_path, 'model.pth')
    pickle_path = os.path.join('runs', model_path, 'model.pkl')
    assert os.path.exists(pickle_path), 'No model found'
    if os.path.exists(pickle_path):
        model = load_model(pickle_path=pickle_path)
        print('Model loaded from pickle')
    elif os.path.exists(weights_path):
        model = load_model(from_scratch=False, weights_path=weights_path)
        print('Model loaded from weights')
    else:
        raise ValueError(f'No model found at {weights_path} or {pickle_path}')
    model = model.to(device)
    model.eval()
    data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    local_res = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            local_res.extend(outputs)

    tensor_res = torch.cat(local_res)
    tensor_res = tensor_res.view(len(test_data), -1)
    return tensor_res


MODEL_PATHS = [
    'train-08-26_14:43_new_59_classes',
    'train-08-26_15:14_bigger_classifier'
]

DATA_TRANSFORMS = [
    base_transform,
    base_transform,
]

assert len(MODEL_PATHS) == len(DATA_TRANSFORMS), 'Wrong paths length'

res = []

submission_name = "submissions/" + '~'.join(MODEL_PATHS) + '~~submission.csv'

if __name__ == '__main__':
    test_data_path = 'data/ukraine-ml-bootcamp-2023/images/test_images'
    images = [os.path.join(test_data_path, i) for i in os.listdir(test_data_path)]

    print('Test size:', len(images))
    for i, (model_path, data_transform) in enumerate(zip(MODEL_PATHS, DATA_TRANSFORMS)):
        test_data = CustomDataset(images, targets=None, transform=data_transform)
        res.append(get_res(model_path, test_data))
        print(f'Predictions {i} done')

    final_res = torch.stack(res).mean(dim=0)
    final_res = [r.argmax().item() for r in final_res]
    final_res = [reversed_groups[r] for r in final_res]

    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    with open(submission_name, 'w') as f:
        f.write('image_id,class_6\n')
        for image, pred in zip(images, final_res):
            f.write(f'{os.path.basename(image)},{pred}\n')
