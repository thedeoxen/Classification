import os

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

folder = "dataset/train_images/"


# folder = "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images/"

class RSNADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.patient_ids = data['patient_id']
        self.labels = data.drop(columns=['patient_id'])
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __normalize_tensor(self, tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        min = torch.min(tensor)
        max = torch.max(tensor)
        # normalized_data = (tensor - mean) / std
        normalized_data = (tensor - min) / max
        return normalized_data

    def __select_indices(self, arr_length, target_length):
        target_indices = np.linspace(0, arr_length - 1, target_length)
        selected_indices = np.round(target_indices).astype(int)
        return selected_indices

    def __get_images(self, patient_id):
        if patient_id in self.cache:
            return self.cache[patient_id]

        image_paths = [
            f'{folder}/{patient_id}/{dir}/{file}'
            for dir in os.listdir(f'{folder}/{patient_id}')
            for file in
            os.listdir(f'{folder}/{patient_id}/{dir}')
        ]

        image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        indexes = self.__select_indices(len(image_paths), 128)
        selected_images = [image_paths[i] for i in indexes]
        image_batch = [cv2.resize(pydicom.dcmread(file).pixel_array, (128, 128)) for file in selected_images]
        images = np.array(image_batch, dtype=np.float32)
        images = torch.tensor(images)
        images = self.__normalize_tensor(images)
        self.cache[patient_id] = images
        return images

    def __getitem__(self, idx):
        patient_ids = int(self.patient_ids.iloc[idx])
        labels = self.labels.iloc[idx]
        images = self.__get_images(patient_ids)

        return images, torch.tensor(labels.to_numpy(dtype=np.float32))
