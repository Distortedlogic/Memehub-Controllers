from typing import Callable, List, cast

import torch
from torchvision import transforms
from torchvision.transforms import (
    ColorJitter,
    RandomAffine,
    RandomHorizontalFlip,
    ToTensor,
)
from torchvision.transforms.transforms import RandomVerticalFlip

toTensorOnly: Callable[..., torch.Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trainingTransforms: Callable[..., torch.Tensor] = transforms.Compose(
    cast(
        List[Callable[..., torch.Tensor]],
        [
            RandomHorizontalFlip(),
            # RandomVerticalFlip(0.1),
            RandomAffine(5, shear=5, scale=(0.95, 1.05)),
            ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
)
