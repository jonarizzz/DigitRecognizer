# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# plot some examples
train_data = pd.read_csv("input/train.csv", header=0)
train_label = train_data['label']
train_img = train_data.iloc[:, 1:] / 255  # scale
train_img = train_img.to_numpy()
train_img.resize(42000, 28, 28, 1)

samples = np.random.randint(0, 10, 2)
for i in samples:
    plt.imshow(np.resize(train_img[i], (28, 28)))
    plt.show()
