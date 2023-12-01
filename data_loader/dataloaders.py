from torchvision import transforms
from torch.utils.data import Dataset
from base_data_loader import BaseDataLoader
from datasets import *

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, input_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize([*input_size]),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = CustomMNISTDataset(self.data_dir, transform = trsfm, target)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class RGBSimpleShapeDataLoader(BaseDataLoader):
    def __init__(self, data_dir, input_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize([*input_size]),
            ConvertImageDtype(torch.float)
        ])
        self.data_dir = data_dir
        self.dataset = RGBSimpleShapeDataset(self.data_dir, transform = trsfm, target)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MalariaDataLoader(BaseDataLoader):
    def __init__(self, data_dir, input_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ColorJitter(brightness=0.5),  # 更改明亮度
            transforms.RandomRotation(
                degrees=45, expand = True
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=train_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)