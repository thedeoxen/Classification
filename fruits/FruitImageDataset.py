import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm


def main():
    pass


if __name__ == '__main__':
    main()


class FruitImageDataset(Dataset):
    def __init__(self, folder, image_size, train=True):
        # super().__init__(self)
        self.images, self.fruit, self.fresh = [], [], []
        for file in tqdm(os.listdir(folder)):
            for img in os.listdir(os.path.join(folder, file))[:100]:
                self.fresh.append(0 if file[0] == 'f' else 1)
                self.fruit.append(file[5:] if file[0] == 'f' else file[6:])
                path = os.path.join(folder, file, img)

                self.images.append(path)

        self.classes = sorted(list(set(self.fruit)))
        self.targets = self.fruit
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

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
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(Image.fromarray(image))
        fruit = self.class_to_idx[self.fruit[idx]]
        return image, fruit, self.fresh[idx]
