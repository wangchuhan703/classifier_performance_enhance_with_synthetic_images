import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any

from torch.utils.data import ConcatDataset

cifar_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class MyCustomDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        super().__init__(root, transform=transform, target_transform=target_transform)

        # 设置指向训练或测试目录的路径
        self.data_folder = root
        self.dataset = datasets.ImageFolder(root=self.data_folder, transform=self.transform)
        class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }
        self.dataset.class_to_idx = class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.dataset[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.dataset)


class CIFAR10:
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):


        cifar10_dataset = PyTorchCIFAR10(
            root=location, download=True, train=True, transform=preprocess
        )

        custom_dataset = MyCustomDataset(
            root='/ocean/projects/cis220031p/hchen10/chw_research/transfer_learning/Diffusion/StableDiffusionImages',
            transform=preprocess
        )

        self.train_dataset = ConcatDataset([cifar10_dataset, custom_dataset])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)

class CIFAR101:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        data_root = os.path.join(location, "CIFAR-10.1")
        data = np.load(os.path.join(data_root, 'cifar10.1_v6_data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(data_root, 'cifar10.1_v6_labels.npy'), allow_pickle=True)

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.train_loader = None

        self.test_dataset = BasicVisionDataset(
            images=data, targets=torch.Tensor(labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames


class CIFAR102:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):

        train_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_train.npy'), allow_pickle=True).item()
        test_data = np.load(os.path.join(location, "CIFAR-10.2", 'cifar102_test.npy'), allow_pickle=True).item()


        train_data_images = train_data['images']
        train_data_labels = train_data['labels']

        test_data_images = test_data['images']
        test_data_labels = test_data['labels']

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.test_dataset = BasicVisionDataset(
            images=test_data_images, targets=torch.Tensor(test_data_labels).long(),
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.classnames = cifar_classnames
