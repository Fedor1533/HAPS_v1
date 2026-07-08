import sys, os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, dataset_path, dist_col='dist', class_col='class', return_meta=False, meta_cols=("pname", "patch_id")):
        """
        :param dataset_dir: Путь к таблицам с метаданными (пути к файлам, ошибка регистрации, класс)
        """
        super().__init__()
        self.base_dir = "/beegfs/home/f.gubanov/f.gubanov/bimai_lab" # path to folder with data
        self.dataset_path = dataset_path
        self.annotations = pd.read_csv(self.dataset_path)
        self.dist_col = dist_col
        self.class_col = class_col
        
        self.return_meta = return_meta
        self.meta_cols = tuple(meta_cols)

    def __len__(self):
        return len(self.annotations)
    
    @staticmethod
    def unpack_item(item):
        """
        Унифицированная распаковка meta data.
        """
        if len(item) == 4:
            f_patch, w_patch, dist, target = item
            return f_patch, w_patch, dist, target, None
        if len(item) == 5:
            f_patch, w_patch, dist, target, meta = item
            return f_patch, w_patch, dist, target, meta
        raise ValueError(f"Unexpected dataset item length: {len(item)}")

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # пути до изображений
        fixed_path = os.path.join(self.base_dir, row['fixed_path'])
        warped_path = os.path.join(self.base_dir, row['warped_path'])

        # reg. error and class
        dist, target = row[self.dist_col], row[self.class_col]

        # Загружаем патчи
        f_patch = np.array(Image.open(fixed_path).convert("RGB"))
        w_patch = np.array(Image.open(warped_path).convert("RGB"))
        
        if not self.return_meta:
            return f_patch, w_patch, dist, target

        meta = {k: row[k] for k in self.meta_cols}
        return f_patch, w_patch, dist, target, meta