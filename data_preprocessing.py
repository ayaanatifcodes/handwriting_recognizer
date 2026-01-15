import cv2
import torch
from torchvision import transforms
import numpy as np
from typing import Tuple

# Importing all the necessary technologies

class Preprocessor: # Class defined
    def __init__( # Initialization of a few objects
            self,
            image_size: Tuple[int, int] =  (244, 244), # Tuple ensures values stay stable
            augmentation: bool = False, # Flagged as not needed during processing of real data
            vocab: str = "" # Null due to no vocab being passed
    ):
       
        self.image_size = image_size
        self.augment = augmentation
        self.vocab = vocab

        self.affine_transform = transforms.Compose([ # Composing a series of transformations
            transforms.ToPILImage(), # Torch or NumPy array to PIL image
            transforms.RandomAffine( # Random values will be set for data
                degrees = 25,
                translate = (0.1, 0.1),
                shear = 10,
                scale = (0.8, 1.2)
                # The upper values are RANGES rather than absolutes
            )
        ])

    def __call__(self, img: np.ndarray, label: str, max_len: int = 35): # Image set to a NumPy array
        img, label = self.preprocess_img(img, label)
        # In the preprocess function, the image is resized to what all data should be in dimensions
        # Label will be cleaned, meaning that all the extra spaces will be cleared and the text will be made lowercase
        label = self.label_indexer(self.vocab, label)
        # The label will be converted into indices based on the vocab provided
        label = self.label_padding(len(self.vocab), max_len, label)
        # The label will be padded to ensure that all labels are of the same length

        if self.augment: # If set to true, augment the image
            img = self.apply_augmnetation(img) # Apply augmnetation on the image for preprocessing
        return img, label # Return the processed image and label

    def preprocess_img(self, img: np.ndarray, text: str):
        target_w, target_h = self.image_size
        h, w = img.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = cv2.resize(img, (new_w, new_h))

        pad_w = target_w - new_w
        pad_h = target_h - new_h

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

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img = self.random_sharpen(img)

        if np.random.rand() < 0.5:
            img = np.array(self.affine_transform(img))

        return img

    @staticmethod
    def random_sharpen(
        image: np.ndarray,
        alpha_range: Tuple[float, float] = (0.25, 1.0),
        lightness_range: Tuple[float, float] = (0.75, 2.0),
    ) -> np.ndarray:
        alpha = np.random.uniform(*alpha_range)
        lightness = np.random.uniform(*lightness_range)

        kernel_identity = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            dtype=np.float32
        )

        kernel_sharpen = np.array(
            [[-1, -1, -1],
             [-1,  8 + lightness, -1],
             [-1, -1, -1]],
            dtype=np.float32
        )

        kernel = (1 - alpha) * kernel_identity + alpha * kernel_sharpen
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def label_indexer(vocab: str, label: str) -> np.ndarray:
        return np.array([vocab.index(c) for c in label if c in vocab])

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray) -> np.ndarray:
        label = label[:max_len]
        return np.pad(
            label,
            (0, max_len - len(label)),
            mode="constant",
            constant_values=padding_value
        )

    def single_image_preprocessing(self, img: np.ndarray) -> torch.Tensor:
        img, _ = self.preprocess_img(img, "")
        img = torch.from_numpy(img).float()
        img = img / 255.0
        return img





