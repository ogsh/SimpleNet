from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np

from util.make_gaussian import make_gaussian_map
from util.util import img2tensor

class BBDataSet(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.bb_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bb_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.bb_frame.iloc[idx, 0])

        if not os.path.exists(img_name):
            raise ValueError('image file not found: ' + img_name)

        image = cv2.imread(img_name)
        image = image[:, :, ::-1].copy()
        image = image.transpose(2, 0, 1)
        bb_data = self.bb_frame.iloc[idx, 1:].values
        bb_data = bb_data.astype('float').reshape(-1, 2)
        center = np.sum(bb_data, 0) * 0.5
        size = bb_data[1, :] - bb_data[0, :]
        size = size[::-1]
        bb_map = make_gaussian_map(image.shape[1:], center, size)
        bb_map = bb_map[np.newaxis, :]

        sample = {'image': image, 'bb': bb_map}

        if self.transform:
            sample = self.transform(sample)

        return sample

