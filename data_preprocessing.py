import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, Dict

class Preprocessor:
    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 244),
            augmentation: bool = False,
            vocab: str = ""
    ):
        self.image_size = image_size
        self.augment = augmentation
        self.vocab = vocab
        self.vocab_dict = {}
        for i, c in enumerate(vocab):
            self.vocab_dict[c] = i

        self.affine_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15,
                shear=10,
                scale=(0.8, 1.2),
                translate=(0.2, 0.4)
            )
        ])

    def __call__(
            self,
            image: np.ndarray,
            label: str,
            max_len: int = 40
    ):
        img, label = self.preprocess_img(image, label)
        label = label.lower().strip()
        label = self.label_indexer(self.vocab_dict, label)
        label = self.label_padding(len(self.vocab), max_len, label)

        if self.augment:
            img = self.apply_augmentation(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float() / 255.0
        return img, label
    
    def preprocess_img(
            self,
            img: np.ndarray,
            text: str
    ):
        target_width, target_height = self.image_size
        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_width, new_height = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_width, new_height))

        pad_w = target_width - new_width
        pad_h = target_height - new_height
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=255
        )
        return img, text
    
    def apply_augmentation(
            self,
            img: np.ndarray
    ) -> np.ndarray:
        
        if np.random.rand() < 0.5:
            img = self.random_sharpen(img)

        if np.random.rand() < 0.5:
            img = np.array(self.affine_transform(img))
        return img
    
    @staticmethod
    def random_sharpen(
        image: np.ndarray,
        strength_range: Tuple[float, float] = (0.25, 1.0),
        lightness_range: Tuple[float, float] = (0.75, 2.0)
    ) -> np.ndarray:
        alpha = np.random.uniform(*strength_range)
        lightness = np.random.uniform(*lightness_range)

        kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]], dtype=np.float32)
        kernel = (1 - alpha) * kernel_identity + alpha * kernel_sharpen
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def label_indexer(vocab_dict: Dict[str, int], label: str) -> np.ndarray:
        return np.array([vocab_dict[c] for c in label if c in vocab_dict], dtype=np.int64)

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray) -> np.ndarray:
        label = label[:max_len]
        return np.pad(label, (0, max_len - len(label)), mode="constant", constant_values=padding_value)

    def single_image_preprocessing(self, img: np.ndarray) -> torch.Tensor:
        img, _ = self.preprocess_img(img, "")
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float() / 255.0
        return img
