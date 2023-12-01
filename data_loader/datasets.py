import os
import numpy as np

class CustomMNISTDataset(Dataset):
    def __init__(
        self,
        root: str,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.augmentation = augmentation
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        image_file = f"images.idx3-ubyte"
        label_file = f"labels.idx1-ubyte"
        data, targets = loadlocal_mnist(
            images_path=os.path.join(self.root, image_file),
            labels_path=os.path.join(self.root, label_file)
        )
        data = data.reshape(len(data), 28, 28)
        all_masked_data = data.copy()
        all_targets = targets.copy()
        
        if self.augmentation:
            all_masked_data, all_targets = self.data_argument(all_masked_data, all_targets)
        return all_masked_data, all_targets 

    def data_argument(self, all_masked_data, all_targets):
        all_masked_data = all_masked_data[0:1, :, :]
        all_targets = all_targets[0:1]
        mask_value = 0
        positions = [
            [(0, 0), (14, 14)], # 左上
            [(0, 14), (14, 28)], # 右上
            [(14, 0), (28, 14)], # 左下
            [(14, 14), (28, 28)], # 右下
            [(0, 0), (14, 28)], # 上半
            [(14, 0), (28, 28)], # 下半
        ]
        
        for position in positions:
            left, right = position
            mask = np.zeros_like(data)
            mask[:, left[0]:right[0], left[1]:right[1]] = 1.0
            masked_data = np.where(mask == 1.0, mask_value, data)
            all_masked_data = np.concatenate((all_masked_data, masked_data), axis=0)
            all_targets = np.concatenate((all_targets, targets), axis=0)
        np.delete(all_masked_data, 0, axis=0)
        np.delete(all_targets, 0, axis=0)
        return all_masked_data, all_targets
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], int(self.targets[index])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)

class RGBSimpleShapeDataset(Dataset):
    def __init__(self,
        root: str,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.augmentation = augmentation
        labels = ['red_circle', 'green_circle', 'blue_circle', 'red_ellipse', 'green_ellipse', 'blue_ellipse', 
                    'red_rectangle', 'green_rectangle', 'blue_rectangle', 'red_square', 'green_square', 'blue_square',
                    'red_triangle', 'green_triangle', 'blue_triangle']
        self.label_to_num = {k:i for i,k in enumerate(labels)}
        self.num_to_label = {i:k for i,k in enumerate(labels)}
        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = self.root
        image_dataset = []
        label_dataset = []
        for root, dirs, files in os.walk(image_file):
            for name in files:
                name_split = '_'.join(name.split('_')[:2])
                label = self.label_to_num[name_split]

                y_onehot = np.eye(15)[label]
                y_onehot = torch.from_numpy(y_onehot)
                label_dataset.append(y_onehot)
                image = read_image(os.path.join(root, name))
                image_dataset.append(image)

        return image_dataset, label_dataset
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.data)

