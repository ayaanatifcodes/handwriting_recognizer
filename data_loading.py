from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessor import Preprocessor
from typing import List, Tuple, Optional, Callable

class HandwritingDataset(Dataset):
   
    def __init__(
        self, 
        data: List[Tuple[str, str]], 
        vocabulary: str = "", 
        max_len: int = 0, 
        transform: Optional[Callable] = None, 
        augmentations: bool = False
    ):


        self.data: List[Tuple[str, str]] = data
        self.transform: Optional[Callable] = transform
        self.vocab: str = vocabulary
        self.max_len: int = max_len
        self.data_preprocessor: Preprocessor = Preprocessor(
            image=None, 
            vocab=self.vocab, 
            augmentation=augmentations
        )
   
    def __len__(self) -> int:
        return len(self.data)
   
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image_path: str
        label: str
        image_path, label = self.data[idx]
        
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image, label_indices = self.data_preprocessor(image, label, self.max_len)
       
        if self.transform:
            image = self.transform(image)
       
        image = torch.from_numpy(image).float()
        image = image / 255.0
       
        return image, label_indices
