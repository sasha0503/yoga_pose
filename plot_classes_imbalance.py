import os

import matplotlib.pyplot as plt

subcat_path = 'data/ukraine-ml-bootcamp-2023/subcat'

subcats = os.listdir(subcat_path)
subcats = [subcat for subcat in subcats if os.path.isdir(os.path.join(subcat_path, subcat))]
class_instances = {}

for subcat in subcats:
    class_id = subcat
    class_instances[class_id] = class_instances.get(class_id, 0) + len(os.listdir(os.path.join(subcat_path, subcat)))

plt.bar(class_instances.keys(), class_instances.values(), width=0.6)
plt.xticks(rotation=90, ha='right', fontsize=5)  # Rotate and adjust font size
plt.subplots_adjust(bottom=0.3)  # Add more space at the bottom for labels
plt.tight_layout()  # Adjust padding
plt.savefig('data/ukraine-ml-bootcamp-2023/subcat_plot.png')
