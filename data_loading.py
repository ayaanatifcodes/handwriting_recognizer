from sklearn.model_selection import train_test_split  # split datasets into training and testing sets
import os  # interact with the operating system (paths, directories, files)
import torch  # PyTorch for building and training neural networks
import numpy as np  # numerical computations and array operations
import cv2  # OpenCV for image processing
from torchvision import transforms  # image transformations and preprocessing
from torch.utils.data import Dataset  # base class for custom datasets
from torch.utils.data import DataLoader  # batch loading and iteration over datasets
from preprocessor import Preprocessor  # custom preprocessing logic
from typing import List, Tuple, Optional, Callable  # type annotations


class HandwritingDataset(Dataset):  # custom dataset for handwriting recognition

    def __init__(
        self,
        data: List[Tuple[str, str]],  # list of (image_path, label) pairs
        vocabulary: str = "",  # string containing all unique characters
        max_len: int = 0,  # maximum label sequence length
        transform: Optional[Callable] = None,  # optional torchvision transform pipeline
        augmentations: bool = False  # toggle data augmentation
    ):
        self.data: List[Tuple[str, str]] = data  # store dataset samples
        self.transform: Optional[Callable] = transform  # store optional transforms
        self.vocab: str = vocabulary  # store vocabulary for encoding
        self.max_len: int = max_len  # store max sequence length

        self.data_preprocessor: Preprocessor = Preprocessor(  # initialize preprocessing logic
            image=None,  # image is provided later in __getitem__
            vocab=self.vocab,  # vocabulary for text-to-index encoding
            augmentation=augmentations  # enable/disable augmentation
        )

    def __len__(self) -> int:  # return dataset size
        return len(self.data)  # number of samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:  # get one sample
        image_path, label = self.data[idx]  # extract image path and label

        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # load grayscale image

        image, label_indices = self.data_preprocessor(  # preprocess image and encode label
            image, label, self.max_len
        )

        if self.transform:  # apply optional torchvision transforms
            image = self.transform(image)

        image = torch.from_numpy(image).float()  # convert NumPy array to float32 tensor
        image = image / 255.0  # normalize pixel values to [0, 1]

        return image, label_indices  # return processed image and encoded label
