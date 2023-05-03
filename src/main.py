import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import char_plot

if not os.path.isfile('./data'):
   with zipfile.ZipFile('./data.zip', 'r') as zip_ref:
       zip_ref.extractall('./')

directory_path = "./data/train/"
labels = ['covid', 'normal', 'viral']
sizes = np.zeros(3, dtype=np.int32)

for i, l in enumerate(labels):
    sizes[i] = len(os.listdir(directory_path + l))
    plt.imshow(mpimg.imread(directory_path + l + '/' + str(i+1) + ".jpeg"))
    plt.title(l.capitalize() + " Patient")
    plt.show()

char_plot(sizes, labels)