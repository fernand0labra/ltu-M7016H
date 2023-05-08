import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils import char_plot


#############################################
#               Data Analysis               #
#############################################

data_path = "./data"
set_path = ["train", "test"]
labels = ['covid', 'normal', 'viral']
sizes = np.zeros(3, dtype=np.int32)

for set in set_path:
    fig = plt.figure(figsize=(12, 3.5))
    fig.suptitle(set.capitalize() + " Data")
    
    for ix, l in enumerate(labels):
        directory_path = data_path + "/" + set + "/" + l
        sizes[ix] = len(os.listdir(directory_path))
        plt.subplot(1, 3, ix + 1)
        plt.imshow(mpimg.imread(directory_path + '/' + str(ix + 1) + ".jpeg"))
        plt.title(l.capitalize() + " Patient")
    plt.show() 

    char_plot(set, sizes, labels)


#############################################
#             Data Augmentation             #
#############################################

augmentation_transform = transforms.Compose([
    transforms.RandomEqualize(1.0),  # Histogram equalization (./docs/papers/article-3.pdf)
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 3
augmentation_dataset = datasets.ImageFolder('./data/train/', transform=augmentation_transform)
augmentation_dataloader = DataLoader(augmentation_dataset, batch_size=batch_size, shuffle=True)

iterable = iter(augmentation_dataloader)
fig = plt.figure(figsize=(batch_size * 4, batch_size * 4))

for bix in range(0, batch_size):
    image_list, label_list = next(iterable)
    
    for ix, (image, label) in enumerate(zip(image_list, label_list), 1):
        plt.subplot(batch_size, batch_size, bix * batch_size + ix)  # [n_rows, n_cols, index]
        plt.imshow(image.numpy().transpose((1, 2, 0)))  # (C, H, W) -> (H, W, C)
        plt.title(augmentation_dataset.classes[label].capitalize() + " Patient")

plt.show()