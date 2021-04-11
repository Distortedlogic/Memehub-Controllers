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
            RandomVerticalFlip(0.3),
            RandomAffine(20, shear=20, scale=(0.8, 1.2)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
)
