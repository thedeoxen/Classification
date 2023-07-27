import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from fruits.train import image_size


def main():
    pass


if __name__ == '__main__':
    main()


class FruitImageDataset(Dataset):
    def __init__(self, folder, train=True):
        self.filenames, self.fruit, self.fresh = [], [], []
        for file in tqdm(os.listdir(folder)):
            for img in os.listdir(os.path.join(folder, file)):
                self.fresh.append(0 if file[0] == 'f' else 1)
                self.fruit.append(file[5:] if file[0] == 'f' else file[6:])
                self.filenames.append(os.path.join(folder, file, img))

        self.fruits_classes = sorted(list(set(self.fruit)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.fruits_classes)}

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=10)
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(image_size, image_size))
                ]
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(Image.fromarray(img))
        fruit = self.class_to_idx[self.fruit[idx]]
        return image, fruit, self.fresh[idx]
